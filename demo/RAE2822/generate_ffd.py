"""Build a fitted FFD from the existing blunt-TE RAE2822 geometry."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pygeo
import pyspline
import scipy

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from FFD import ControlLattice, generate_airfoil_plot3d


def file_sha256(path: Path) -> str:
    """Return the digest of the file's original bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plot_lattice(name: str, contour: np.ndarray, lattice: ControlLattice,
                 directory: Path) -> dict:
    """Plot the emitted lattice with undistorted chord-normalized coordinates."""
    font_path = Path("/mnt/c/Windows/Fonts/times.ttf")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        font_family = font_manager.FontProperties(fname=str(font_path)).get_name()
    else:
        font_family = "DejaVu Serif"
    blue = "#002FA7"
    style = {
        "font.family": font_family, "font.size": 18,
        "mathtext.fontset": "stix", "axes.labelsize": 23,
        "xtick.labelsize": 18, "ytick.labelsize": 18,
        "axes.linewidth": .9, "svg.fonttype": "none",
        "savefig.facecolor": "white",
    }
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(10, 3.6))
        fig.subplots_adjust(left=.095, right=.985, bottom=.22, top=.80)
        # Both virtual span planes have the same xy coordinates: show one.
        points = lattice.points[:, :, 0, :2]
        for row in range(2):
            ax.plot(points[:, row, 0], points[:, row, 1], color=blue,
                    lw=1.5, marker="o", ms=5.5, mfc="white", mew=1.25)
        for pair in points:
            ax.plot(pair[:, 0], pair[:, 1], color=blue, lw=1.15)
        ax.plot(contour[:, 0], contour[:, 1], color="black", lw=1.9, zorder=4)
        ax.set(xlim=(-.035, 1.035), ylim=(-.115, .135),
               xlabel=r"$x/c$", ylabel=r"$y/c$")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.set_yticks([-.1, 0, .1])
        ax.tick_params(which="both", direction="in", length=5)
        handles = [Line2D([], [], color="black", lw=1.9, label=name),
                   Line2D([], [], color=blue, lw=1.5, marker="o", mfc="white",
                          ms=5.5, label="FFD control lattice")]
        ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, 1.04),
                  ncol=2, frameon=False, columnspacing=1.4, handlelength=2.0)
        for suffix in ("png", "svg"):
            fig.savefig(directory / f"ffd.{suffix}", dpi=300)
        plt.close(fig)
    return {
        "files": ["ffd.png", "ffd.svg"], "size_inches": [10, 3.6],
        "png_dpi": 300, "font": font_family, "lattice_color": blue,
        "equal_coordinate_scale": True,
        "shown_plane": "z/c=-0.5; the +0.5 plane has identical xy coordinates",
    }


def run_demo(name: str, directory: Path, lower: np.ndarray, upper: np.ndarray,
             contour: np.ndarray, *, n_chord: int, source: dict,
             expected_hashes: dict[str, str]) -> dict:
    """Generate one example and verify all supplied boundary coordinates."""
    input_hashes = {file: file_sha256(directory / file) for file in expected_hashes}
    if input_hashes != expected_hashes:
        raise ValueError("Geometry input differs from the documented source")
    parameters = dict(n_chord=n_chord, x_margin=.001, thickness_margin=.02,
                      relative_thickness_margin=.05, chord_inset_fraction=1e-4)
    output = directory / "ffd.xyz"
    generate_airfoil_plot3d(lower, upper, output, **parameters, overwrite=True)
    lattice = ControlLattice.from_plot3d(output, orders=(4, 2, 2), tag=f"{name}_FFD")
    xyz = np.column_stack([contour, np.zeros(len(contour))])
    uvw, error = lattice.embed(xyz, tolerance=1e-10)

    parameters_sample = np.stack(np.meshgrid(np.linspace(0, 1, 65),
                                            np.linspace(0, 1, 9),
                                            [0., .5, 1.], indexing="ij"), axis=-1)
    jacobian = lattice.jacobian(parameters_sample.reshape(-1, 3))
    determinants = np.linalg.det(jacobian)
    scaled = determinants / np.prod(np.linalg.norm(jacobian, axis=-2), axis=-1)
    checks = {
        "all_boundary_points_embedded": bool(error <= 1e-10),
        "all_parameters_inside": bool(np.all((uvw >= 0) & (uvw <= 1))),
        "positive_sampled_jacobians": bool(np.all(determinants > 0)),
        "nonsingular_sampled_jacobians": bool(np.min(np.abs(scaled)) > 1e-10),
        "upper_controls_above_lower": bool(np.all(
            lattice.points[:, 1, :, 1] > lattice.points[:, 0, :, 1])),
        "identical_xy_in_virtual_planes": bool(np.array_equal(
            lattice.points[:, :, 0, :2], lattice.points[:, :, 1, :2])),
        "su2_virtual_z_planes": bool(np.array_equal(
            lattice.points[..., 2], np.broadcast_to([-.5, .5], lattice.shape))),
        "geometry_inputs_unchanged": input_hashes == {
            file: file_sha256(directory / file) for file in expected_hashes},
    }
    if not all(checks.values()):
        raise ValueError(f"{name}: failed geometric checks: {checks}")
    figure = plot_lattice(name, contour, lattice, directory)
    report = {
        "case": name, "status": "passed", "scope": "baseline geometry and FFD embedding",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": source, "input_sha256": input_hashes,
        "generation_backend": "pygeo.geo_utils.createFittedWingFFD",
        "versions": {"pygeo": pygeo.__version__, "pyspline": pyspline.__version__,
                     "numpy": np.__version__, "scipy": scipy.__version__,
                     "matplotlib": matplotlib.__version__},
        "python_executable": sys.executable, "generation_parameters": parameters,
        "control_shape": list(lattice.shape), "control_point_count": int(np.prod(lattice.shape)),
        "spline_orders": list(lattice.orders),
        "lower_surface_input_points": len(lower), "upper_surface_input_points": len(upper),
        "boundary_input_points_tested": len(xyz),
        "unique_boundary_points_tested": len(np.unique(xyz, axis=0)),
        "trailing_edge_thickness_over_chord": float(upper[-1, 1] - lower[-1, 1]),
        "maximum_embedding_error_over_chord": error,
        "parameter_min": uvw.min(axis=0).tolist(), "parameter_max": uvw.max(axis=0).tolist(),
        "jacobian_samples": len(determinants),
        "minimum_sampled_jacobian_determinant": float(determinants.min()),
        "minimum_sampled_scaled_jacobian": float(np.abs(scaled).min()),
        "checks": checks, "figure": figure,
        "output_sha256": {file: file_sha256(directory / file)
                          for file in ("ffd.xyz", "ffd.png", "ffd.svg")},
    }
    (directory / "validation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{name}: passed; shape={lattice.shape}; "
          f"boundary_points={len(xyz)}; max_error={error:.3e} c", flush=True)
    return report


def main() -> None:
    directory = Path(__file__).resolve().parent
    lower = np.loadtxt(directory / "lower.txt", usecols=(0, 1))
    upper = np.loadtxt(directory / "upper.txt", usecols=(0, 1))
    contour = np.vstack([upper[::-1], lower[1:], upper[-1:]])
    run_demo("RAE2822", directory, lower, upper, contour, n_chord=14,
             source={
                 "geometry_files": [
                     "Engineering/RAE2822/MESH/geometry/RAE2822_lower.txt",
                     "Engineering/RAE2822/MESH/geometry/RAE2822_upper.txt"],
                 "geometry_variant": "Existing normalized blunt-trailing-edge geometry",
                 "local_transformations": "Copy the existing geometry files unchanged; read x,y and assemble a closed plotting boundary.",
             },
             expected_hashes={
                 "lower.txt": "6eee473bac7233fb3de653c17b9c3644bb5689a3a73e8cc9330c505d62d042d6",
                 "upper.txt": "44bff8c527d7e06cf71339d59a031a16b33f21d56e2bcaaca6f202cca3d5c1d2",
             })


if __name__ == "__main__":
    main()
