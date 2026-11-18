"""Generate fitted FFD control lattices with MDOLab pyGeo.

Airfoil coordinate pairs and triangulated wing surfaces are converted to
single-volume ASCII PLOT3D lattices. These are geometric control points, not
CFD volume meshes. SU2 initialization is provided separately by export_su2.
"""
from pathlib import Path
import tempfile

import numpy as np

from .lattice import ControlLattice


def generate_wing_plot3d(surface, output: str | Path, *, surface_format: str,
                         leading_edge, trailing_edge, n_span, n_chord: int,
                         absolute_margins, relative_margins, lift_index: int,
                         overwrite: bool = False) -> Path:
    """Fit one FFD volume to a wing using pyGeo.createFittedWingFFD.

    ``surface_format`` is 'point-point' for triangle vertices or 'point-vector'
    for a list [p0, edge1, edge2]. Leading/trailing reference lines are (N,3)
    arrays inside the surface's span. ``n_span`` is an integer or counts per
    reference-line segment. Both margin vectors use pyGeo's ordering:
    [chord, span, thickness]. Absolute margins have geometry coordinate units;
    relative margins are fractions. ``lift_index`` is 2 for y or 3 for z.

    Conventional M6 and CRM wing surfaces can use this interface after their
    reference lines, axes and margins are supplied. The geometry must admit
    pyGeo's two-sided thickness projections and a nonfolding single volume.
    """
    try:
        from pygeo.geo_utils import createFittedWingFFD
    except ImportError as exc:
        raise ImportError(
            "MDOLab pyGeo and compiled pySpline are required for fitted-lattice "
            "generation. Install them in the active Python environment. "
            "https://mdolab-pygeo.readthedocs-hosted.com/en/latest/install.html"
        ) from exc
    output = Path(output)
    if output.exists() and not overwrite:
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    createFittedWingFFD(surface, surface_format, str(output), leading_edge,
                        trailing_edge, n_span, n_chord, absolute_margins,
                        relative_margins, lift_index)
    return output


def generate_airfoil_plot3d(lower: np.ndarray, upper: np.ndarray,
                            output: str | Path, *, n_chord: int,
                            x_margin: float, thickness_margin: float,
                            relative_thickness_margin: float = 0.0,
                            chord_inset_fraction: float = 1e-4,
                            overwrite: bool = False) -> Path:
    """Generate a 2D airfoil lattice through pyGeo's fitted-wing algorithm.

    ``lower`` and ``upper`` contain (x,y) coordinates ordered from leading to
    trailing edge, with strictly increasing x and the same chord endpoints.
    RAE2822, NACA0012 and other single-valued airfoil surfaces are accepted;
    a sharp or finite-thickness trailing edge is allowed.

    A temporary straight extrusion supplies triangles to pyGeo. Reference
    rays are inset slightly from the chord endpoints to avoid the coincident
    upper/lower intersection at a sharp edge. ``x_margin`` must exceed that
    inset distance. Output rows are lower then upper, with two identical xy
    planes at z=-0.5 and +0.5 for SU2's 2D FFD convention. No shape fitting,
    mesh deformation or design-variable bounds are applied after pyGeo.
    """
    sides = [np.asarray(side, dtype=np.float64) for side in (lower, upper)]
    for side in sides:
        if (side.ndim != 2 or side.shape[1] != 2 or len(side) < 2
                or not np.isfinite(side).all() or np.any(np.diff(side[:, 0]) <= 0)):
            raise ValueError("Airfoil sides must be finite (N,2) arrays with increasing x")
    lo, hi = sides[0][[0, -1], 0]
    if not np.allclose(sides[1][[0, -1], 0], [lo, hi], atol=1e-12, rtol=0):
        raise ValueError("Upper and lower surfaces must have the same chord endpoints")
    x_check = np.unique(np.concatenate([side[:, 0] for side in sides]))
    if np.any(np.interp(x_check, sides[1][:, 0], sides[1][:, 1])
              < np.interp(x_check, sides[0][:, 0], sides[0][:, 1])):
        raise ValueError("Upper and lower airfoil surfaces cross")
    if not isinstance(n_chord, int) or n_chord < 2:
        raise ValueError("n_chord must be an integer >=2")
    if (not np.isfinite([x_margin, thickness_margin, relative_thickness_margin,
                         chord_inset_fraction]).all()
            or not 0 < chord_inset_fraction < .5
            or x_margin <= chord_inset_fraction*(hi-lo)
            or thickness_margin <= 0 or relative_thickness_margin < 0):
        raise ValueError("Use positive margins, x_margin greater than chord inset, and nonnegative relative margin")
    output = Path(output)
    if output.exists() and not overwrite:
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    triangles = []
    for side in sides:
        a = np.column_stack([side, -np.ones(len(side))])
        b = np.column_stack([side, np.ones(len(side))])
        triangles.extend(np.stack([a[:-1], a[1:], b[1:]], axis=1))
        triangles.extend(np.stack([a[:-1], b[1:], b[:-1]], axis=1))
    refs = []
    for x in [lo+chord_inset_fraction*(hi-lo), hi-chord_inset_fraction*(hi-lo)]:
        y = .5*sum(np.interp(x, side[:, 0], side[:, 1]) for side in sides)
        refs.append(np.array([[x, y, -.5], [x, y, .5]]))
    with tempfile.TemporaryDirectory(prefix="pygeo_airfoil_") as temporary:
        raw = generate_wing_plot3d(
            np.asarray(triangles), Path(temporary)/"raw.xyz", surface_format="point-point",
            leading_edge=refs[0], trailing_edge=refs[1], n_span=2, n_chord=n_chord,
            absolute_margins=[x_margin, 0., thickness_margin],
            relative_margins=[0., 0., relative_thickness_margin], lift_index=2,
        )
        lattice = ControlLattice.from_plot3d(raw, orders=(2, 2, 2))
    # Normalize pyGeo's upper/lower ordering to lower/upper in the 2D interface.
    lattice.points = lattice.points[:, ::-1, :, :].copy()
    lattice.write_plot3d(output)
    return output
