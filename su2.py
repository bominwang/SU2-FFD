"""Read a serial SU2 mesh surface and append native, single-volume FFD data."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np

from .lattice import ControlLattice


@dataclass
class MeshSurface:
    dimension: int
    points: np.ndarray
    markers: dict[str, np.ndarray]
    mesh_bytes: bytes


def read_surface(path: str | Path) -> MeshSurface:
    """Read 2D/3D serial SU2 ASCII coordinates and line/triangle/quad markers.

    Volume connectivity and the original mesh body are preserved verbatim.
    Existing FFD metadata is replaced only in the newly exported file.
    """
    raw = Path(path).read_bytes()
    physical_lines = raw.splitlines(keepends=True)
    for i, line in enumerate(physical_lines):
        if line.split(b"%", 1)[0].strip().startswith(b"FFD_NBOX"):
            raw = b"".join(physical_lines[:i])
            break
    lines = iter(v for line in raw.decode("utf-8-sig").splitlines()
                 if (v := line.split("%", 1)[0].strip()))
    dimension, points, markers = None, None, {}
    for line in lines:
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if key == "NZONE" and int(value) != 1:
            raise ValueError("Only single-zone serial SU2 meshes are supported")
        if key == "NDIME":
            if dimension is not None:
                raise ValueError("Multiple mesh zones are unsupported")
            dimension = int(value)
            if dimension not in (2, 3):
                raise ValueError("Expected NDIME=2 or 3")
        elif key == "NELEM":
            for _ in range(int(value)):
                next(lines)
        elif key == "NPOIN":
            if dimension is None:
                raise ValueError("NDIME must precede NPOIN")
            rows = [next(lines).split() for _ in range(int(value.split()[0]))]
            points = np.zeros((len(rows), 3), dtype=np.float64)
            for i, row in enumerate(rows):
                points[i, :dimension] = [float(v) for v in row[:dimension]]
                if len(row) > dimension and int(row[dimension]) != i:
                    raise ValueError("Expected serial, zero-based point numbering")
        elif key == "NMARK":
            for _ in range(int(value)):
                marker_key, _, tag = next(lines).partition("=")
                count_key, _, count = next(lines).partition("=")
                tag = tag.strip()
                if marker_key.strip() != "MARKER_TAG" or count_key.strip() != "MARKER_ELEMS" or tag in markers:
                    raise ValueError("Invalid or duplicate boundary marker")
                ids = []
                for _ in range(int(count)):
                    row = [int(v) for v in next(lines).split()]
                    width = {3: 2, 5: 3, 9: 4}.get(row[0])
                    if width is None or len(row) != width+1:
                        raise ValueError("Expected line, triangle or quadrilateral boundary elements")
                    ids.extend(row[1:])
                markers[tag] = np.unique(ids).astype(np.int64)
    if points is None or not np.isfinite(points).all() or not markers:
        raise ValueError("Incomplete SU2 mesh")
    for ids in markers.values():
        if len(ids) and (ids.min() < 0 or ids.max() >= len(points)):
            raise ValueError("Boundary references an invalid point")
    return MeshSurface(dimension, points, markers, raw)


def _metadata(lattice: ControlLattice, dimension: int, surfaces: list[tuple]) -> str:
    ni, nj, nk = lattice.shape
    lines = ["FFD_NBOX= 1", "FFD_NLEVEL= 1", f"FFD_TAG= {lattice.tag}", "FFD_LEVEL= 0",
             f"FFD_DEGREE_I= {ni-1}", f"FFD_DEGREE_J= {nj-1}"]
    if dimension == 3:
        lines.append(f"FFD_DEGREE_K= {nk-1}")
    lines.append("FFD_BLENDING= BSPLINE_UNIFORM")
    for axis, order in zip("IJK"[:dimension], lattice.orders):
        lines.append(f"BSPLINE_ORDER_{axis}= {order}")
    lines += ["FFD_PARENTS= 0", "FFD_CHILDREN= 0", f"FFD_CORNER_POINTS= {2**dimension}"]
    corners = [(0, 0, 0), (ni-1, 0, 0), (ni-1, nj-1, 0), (0, nj-1, 0)]
    if dimension == 3:
        corners += [(i, j, nk-1) for i, j, _ in corners.copy()]
    for index in corners:
        lines.append(" ".join(f"{v:.17g}" for v in lattice.points[index][:dimension]))
    lines.append(f"FFD_CONTROL_POINTS= {ni*nj*nk}")
    for index in np.ndindex(lattice.shape):
        lines.append(" ".join(map(str, index))+" "+" ".join(
            f"{v:.17g}" for v in lattice.points[index]))
    lines.append(f"FFD_SURFACE_POINTS= {len(surfaces)}")
    for tag, node, uvw in surfaces:
        lines.append(f"{tag} {node} "+" ".join(f"{v:.17g}" for v in uvw))
    return "\n".join(lines)+"\n"


def export_su2(source: str | Path, output: str | Path, lattice: ControlLattice, *,
               markers: tuple[str, ...], tolerance: float = 1e-10,
               overwrite: bool = False) -> dict:
    """Embed selected surfaces and export a baseline for native SU2_DEF.

    No design-variable ranges, fixed columns, CFD settings or optimizations
    are inferred here. These remain in the caller's case configuration.
    """
    source, output = Path(source), Path(output)
    if source.resolve() == output.resolve():
        raise ValueError("Export must not overwrite its input mesh")
    if output.exists() and not overwrite:
        raise FileExistsError(output)
    mesh = read_surface(source)
    markers = tuple(markers)
    if not markers or len(set(markers)) != len(markers) or set(markers)-mesh.markers.keys():
        raise ValueError("Select existing, distinct mesh boundary markers")
    if any(not len(mesh.markers[tag]) for tag in markers):
        raise ValueError("Selected boundary markers must not be empty")
    if mesh.dimension == 2:
        cp = lattice.points
        if (lattice.shape[2] != 2 or lattice.orders[2] != 2
                or not np.allclose(cp[:, :, 0, :2], cp[:, :, 1, :2], atol=1e-14, rtol=0)
                or not np.allclose(cp[..., 2], np.array([-.5, .5]), atol=1e-14, rtol=0)):
            raise ValueError("2D SU2 requires paired xy controls on z=-0.5 and +0.5")
    ids = np.unique(np.concatenate([mesh.markers[tag] for tag in markers]))
    uvw, error = lattice.embed(mesh.points[ids], tolerance)
    rows = [(tag, int(node), uvw[np.searchsorted(ids, node)])
            for tag in markers for node in mesh.markers[tag]]
    payload = mesh.mesh_bytes
    if payload and not payload.endswith(b"\n"):
        payload += b"\n"
    payload += _metadata(lattice, mesh.dimension, rows).encode()
    # All validation finishes before an existing output is opened.
    with output.open("wb" if overwrite else "xb") as stream:
        stream.write(payload)
    shape, orders = lattice.shape, lattice.orders
    options = {
        "FFD_BLENDING": "BSPLINE_UNIFORM",
        "FFD_BSPLINE_ORDER": ", ".join(map(str, orders)),
        "FFD_DEGREE": "( "+", ".join(map(str, [shape[0]-1, shape[1]-1,
                                                    shape[2]-1 if mesh.dimension == 3 else 0]))+" )",
    }
    return {
        "source": str(source.resolve()), "output": str(output.resolve()),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(payload).hexdigest(),
        "mesh_dimension": mesh.dimension, "tag": lattice.tag,
        "control_grid": list(shape), "spline_orders": list(orders),
        "unique_surface_nodes": len(ids),
        "marker_nodes": {tag: len(mesh.markers[tag]) for tag in markers},
        "maximum_embedding_error": error, "embedding_tolerance": tolerance,
        "su2_config_options": options, "mesh_body_preserved": True,
        "design_domain_defined": False,
    }
