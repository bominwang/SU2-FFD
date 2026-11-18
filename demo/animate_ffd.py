"""Animate the four FFD examples using SU2's uniform B-spline mapping.

Read existing geometry and ffd.xyz files; write ffd_deformation.gif and
animation.json in each selected example directory. This is a geometry demo:
the motion amplitudes do not define an engineering optimization design domain.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))
from FFD import ControlLattice

CASES = ("NACA0012", "RAE2822", "M6", "CRM")
BLUE = "#002FA7"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_geometry(case: str) -> tuple[np.ndarray, np.ndarray | None, list[np.ndarray], list[str]]:
    """Reuse each independent example's geometry without running its main()."""
    directory = ROOT / case
    if case == "NACA0012":
        contour = np.loadtxt(directory / "geometry.dat")
        xyz = np.column_stack([contour, np.zeros(len(contour))])
        return xyz, None, [np.arange(len(xyz))], ["geometry.dat"]
    if case == "RAE2822":
        lower = np.loadtxt(directory / "lower.txt", usecols=(0, 1))
        upper = np.loadtxt(directory / "upper.txt", usecols=(0, 1))
        contour = np.vstack([upper[::-1], lower[1:], upper[-1:]])
        xyz = np.column_stack([contour, np.zeros(len(contour))])
        return xyz, None, [np.arange(len(xyz))], ["lower.txt", "upper.txt"]

    spec = importlib.util.spec_from_file_location(f"ffd_demo_{case}", directory / "generate_ffd.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    patches = module.read_cad(directory / module.CAD_FILENAME)
    _, _, quads, _, _, geometry_report = module.prepare_geometry(patches)
    curves = module.geometry_outlines(patches, geometry_report)
    lengths = [len(quads) * 4, *[len(curve) for curve in curves]]
    joined = np.concatenate([quads.reshape(-1, 3), *curves])
    unique, inverse = np.unique(joined, axis=0, return_inverse=True)
    offsets = np.cumsum([0, *lengths])
    faces = inverse[:lengths[0]].reshape(-1, 4)
    outlines = [inverse[a:b] for a, b in zip(offsets[1:-1], offsets[2:])]
    return unique, faces, outlines, [module.CAD_FILENAME]


def motion_modes(lattice: ControlLattice, is_wing: bool) -> tuple[np.ndarray, list[str]]:
    """Define two smooth demonstration motions in control-point coordinates."""
    controls = lattice.points
    ni, nj, nk = lattice.shape
    chord = np.linspace(0., 1., ni)
    modes = np.zeros((2, *controls.shape))
    envelope = np.sin(np.pi * chord) ** 2
    envelope[[0, -1]] = 0.
    if not is_wing:
        # Both virtual span planes move identically; leading/trailing columns stay fixed.
        modes[0, ..., 1] = (.032 * envelope * (1. - .35 * chord))[:, None, None]
        signs = np.array([-1., 1.])[None, :, None]
        modes[1, ..., 1] = .010 * envelope[:, None, None] * signs
        names = ["camber", "thickness"]
    else:
        span = np.linspace(0., 1., nj)
        local_chord = controls[-1, :, :, 0].mean(1) - controls[0, :, :, 0].mean(1)
        # The root control plane and chord-end columns stay fixed for these shape modes.
        bending = .22 * span ** 1.4
        modes[0, ..., 2] = envelope[:, None, None] * bending[None, :, None]
        camber = .085 * envelope[:, None] * (2. * chord[:, None] - .65)
        modes[1, ..., 2] = (camber * local_chord[None, :] * span[None, :])[..., None]
        gap = controls[:, :, 1, 2] - controls[:, :, 0, 2]
        modes[1, ..., 2] += (.12 * envelope[:, None] * gap * span[None, :])[..., None] * np.array([-1., 1.])
        names = ["span-varying camber", "chordwise shape and thickness"]
    return modes, names


def control_connectivity(shape: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Return edge and enclosure-panel indices into the flattened lattice."""
    ids = np.arange(np.prod(shape)).reshape(shape)
    edges, panels = [], []
    for axis in range(3):
        for index in np.ndindex(shape):
            other = list(index)
            other[axis] += 1
            if other[axis] < shape[axis]:
                edges.append([ids[index], ids[tuple(other)]])
        for end in (0, shape[axis] - 1):
            face = np.take(ids, end, axis=axis)
            for i in range(face.shape[0] - 1):
                for j in range(face.shape[1] - 1):
                    panels.append(face[[i, i + 1, i + 1, i], [j, j, j + 1, j + 1]])
    return np.array(edges), np.array(panels)


def camera_matrix() -> np.ndarray:
    """Use the same rigid, equal-scale orthographic camera as the static figures."""
    elev, azim, roll = np.deg2rad([28., -25., 25.])
    right = np.array([-np.sin(azim), np.cos(azim), 0.])
    up = np.array([-np.sin(elev) * np.cos(azim), -np.sin(elev) * np.sin(azim), np.cos(elev)])
    depth = np.array([np.cos(elev) * np.cos(azim), np.cos(elev) * np.sin(azim), np.sin(elev)])
    rotation = np.array([[np.cos(roll), -np.sin(roll)], [np.sin(roll), np.cos(roll)]])
    camera = np.vstack([rotation @ np.vstack([right, up]), depth])
    np.testing.assert_allclose(camera @ camera.T, np.eye(3), atol=1e-14)
    return camera


def figure_style() -> dict:
    font = Path("/mnt/c/Windows/Fonts/times.ttf")
    family = "DejaVu Serif"
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        family = font_manager.FontProperties(fname=str(font)).get_name()
    return {"font.family": family, "font.size": 18, "mathtext.fontset": "stix",
            "axes.labelsize": 23, "xtick.labelsize": 18, "ytick.labelsize": 18,
            "axes.linewidth": .9, "savefig.facecolor": "white"}


def make_airfoil_scene(case: str, xyz: np.ndarray, controls: np.ndarray, limits: np.ndarray,
                       dpi: int):
    fig, ax = plt.subplots(figsize=(10, 3.6), dpi=dpi)
    fig.subplots_adjust(left=.095, right=.985, bottom=.22, top=.80)
    baseline, = ax.plot(xyz[:, 0], xyz[:, 1], color=".65", ls="--", lw=1.1)
    surface, = ax.plot(xyz[:, 0], xyz[:, 1], color="black", lw=1.9, zorder=4)
    rows = [ax.plot(controls[:, j, 0, 0], controls[:, j, 0, 1], color=BLUE,
                    lw=1.5, marker="o", ms=5.5, mfc="white", mew=1.25)[0]
            for j in range(2)]
    columns = LineCollection(controls[:, :, 0, :2], colors=BLUE, linewidths=1.15)
    ax.add_collection(columns)
    ax.set(xlim=limits[:, 0], ylim=limits[:, 1], xlabel=r"$x/c$", ylabel=r"$y/c$")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks(np.linspace(0., 1., 6))
    ax.set_yticks([-.1, 0., .1])
    ax.tick_params(which="both", direction="in", length=5)
    handles = [Line2D([], [], color="black", lw=1.9, label=case),
               Line2D([], [], color=BLUE, lw=1.5, marker="o", mfc="white", label="FFD control lattice"),
               Line2D([], [], color=".65", ls="--", lw=1.1, label="Baseline")]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, 1.04),
              ncol=3, frameon=False, fontsize=16, columnspacing=1.25, handlelength=1.8)

    def update(deformed_xyz: np.ndarray, deformed_controls: np.ndarray) -> None:
        surface.set_data(deformed_xyz[:, 0], deformed_xyz[:, 1])
        for j, row in enumerate(rows):
            row.set_data(deformed_controls[:, j, 0, 0], deformed_controls[:, j, 0, 1])
        columns.set_segments(deformed_controls[:, :, 0, :2])

    return fig, update


def make_wing_scene(case: str, faces: np.ndarray, outlines: list[np.ndarray],
                    lattice: ControlLattice, camera: np.ndarray, limits: np.ndarray, dpi: int):
    fig, ax = plt.subplots(figsize=(9, 6.4), dpi=dpi)
    fig.subplots_adjust(left=.015, right=.985, bottom=.02, top=.90)
    edges, panels = control_connectivity(lattice.shape)
    colors = np.vstack([np.tile([.72, .76, .80, 1.], (len(faces), 1)),
                        np.tile([0., .184, .655, .12], (len(panels), 1))])
    surface = PolyCollection([], linewidths=0, edgecolors="none", antialiaseds=False)
    boundary = LineCollection([], colors="black", linewidths=.75, alpha=.85)
    grid = LineCollection([], colors=BLUE, linewidths=.85, alpha=.88)
    for collection in (surface, boundary, grid):
        ax.add_collection(collection)
    nodes = ax.scatter([], [], s=17, c="#252B33", edgecolors="black", linewidths=.4, zorder=5)
    ax.set_xlim(*limits[:, 0])
    ax.set_ylim(*limits[:, 1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    fig.text(.045, .965, "ONERA M6" if case == "M6" else case, fontsize=22, ha="left", va="top")
    triad = fig.add_axes([.80, .785, .18, .20])
    for label, end in zip("XYZ", (np.eye(3) @ camera.T)[:, :2]):
        triad.annotate("", xy=end, xytext=(0, 0), arrowprops=dict(arrowstyle="-", lw=1.2))
        triad.text(*(end * 1.18), label, fontsize=18, ha="center", va="center")
    triad.set(xlim=(-1.25, 1.25), ylim=(-1.25, 1.25), aspect="equal")
    triad.set_axis_off()

    def update(deformed_xyz: np.ndarray, deformed_controls: np.ndarray) -> None:
        wing = deformed_xyz @ camera.T
        control = deformed_controls.reshape(-1, 3) @ camera.T
        polygons = np.concatenate([wing[faces], control[panels]])
        order = np.argsort(polygons[..., 2].mean(1))
        surface.set_verts(polygons[order, :, :2])
        surface.set_facecolors(colors[order])
        boundary.set_segments([wing[curve, :2] for curve in outlines])
        grid.set_segments(control[edges, :2])
        nodes.set_offsets(control[:, :2])

    return fig, update


def run_case(case: str, frames: int, fps: int, dpi: int) -> dict:
    directory = ROOT / case
    xyz, faces, outlines, inputs = load_geometry(case)
    is_wing = faces is not None
    input_files = [*inputs, "ffd.xyz", "generate_ffd.py"]
    hashes = {file: digest(directory / file) for file in input_files}
    lattice = ControlLattice.from_plot3d(directory / "ffd.xyz", orders=(4, 2, 2))
    print(f"{case}: embedding {len(xyz)} geometry points...", flush=True)
    uvw, error = lattice.embed(xyz, tolerance=1e-9)
    modes, names = motion_modes(lattice, is_wing)
    # Each frame uses the original parameters, never re-embeds a deformed surface.
    mode_lattices = [ControlLattice(mode, lattice.orders) for mode in modes]
    surface_modes = np.stack([mode.evaluate(uvw) for mode in mode_lattices])
    phase = 2. * np.pi * np.arange(frames) / frames
    coefficients = np.column_stack([np.sin(phase), np.sin(2. * phase)])
    sample = np.stack(np.meshgrid(np.linspace(0., 1., 33), np.linspace(0., 1., 25),
                                 [0., .5, 1.], indexing="ij"), axis=-1).reshape(-1, 3)
    baseline_jac = lattice.jacobian(sample)
    mode_jac = np.stack([mode.jacobian(sample) for mode in mode_lattices])
    minimum_jacobian = float("inf")
    maximum_displacement = 0.
    camera = camera_matrix() if is_wing else np.eye(3)
    bounds = np.array([[np.inf, np.inf], [-np.inf, -np.inf]])
    for weight in coefficients:
        jac = baseline_jac + np.tensordot(weight, mode_jac, axes=(0, 0))
        determinants = np.linalg.det(jac)
        if not np.isfinite(determinants).all() or np.any(determinants <= 0.):
            raise ValueError(f"{case}: a demonstration frame folds the sampled FFD mapping")
        minimum_jacobian = min(minimum_jacobian, float(determinants.min()))
        displacement = np.tensordot(weight, surface_modes, axes=(0, 0))
        maximum_displacement = max(maximum_displacement, float(np.linalg.norm(displacement, axis=1).max()))
        control = lattice.points + np.tensordot(weight, modes, axes=(0, 0))
        projected = np.vstack([xyz + displacement, control.reshape(-1, 3)]) @ camera.T
        bounds[0] = np.minimum(bounds[0], projected[:, :2].min(0))
        bounds[1] = np.maximum(bounds[1], projected[:, :2].max(0))
    if is_wing:
        pad = .045 * np.ptp(bounds, axis=0).max()
        limits = bounds + np.array([[-pad, -pad], [pad, pad]])
    else:
        limits = np.array([[-.035, min(-.115, bounds[0, 1] - .015)],
                           [1.035, max(.135, bounds[1, 1] + .015)]])

    images = []
    palette = None
    with plt.rc_context(figure_style()):
        if is_wing:
            fig, update = make_wing_scene(case, faces, outlines, lattice, camera, limits, dpi)
        else:
            fig, update = make_airfoil_scene(case, xyz, lattice.points, limits, dpi)
        try:
            for index, weight in enumerate(coefficients):
                moved = xyz + np.tensordot(weight, surface_modes, axes=(0, 0))
                control = lattice.points + np.tensordot(weight, modes, axes=(0, 0))
                update(moved, control)
                fig.canvas.draw()
                rgb = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
                if palette is None:
                    palette = rgb.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
                images.append(rgb.quantize(palette=palette, dither=Image.Dither.NONE))
                if (index + 1) % 24 == 0 or index + 1 == frames:
                    print(f"{case}: rendered {index + 1}/{frames} frames", flush=True)
        finally:
            plt.close(fig)
    output = directory / "ffd_deformation.gif"
    temporary = directory / "ffd_deformation.gif.tmp"
    images[0].save(temporary, format="GIF", save_all=True, append_images=images[1:],
                   duration=1000 // fps, loop=0, disposal=2, optimize=False)
    with Image.open(temporary) as gif:
        actual_frames, size = gif.n_frames, gif.size
        duration = sum(gif.seek(i) or gif.info["duration"] for i in range(gif.n_frames))
        if actual_frames != frames or gif.info.get("loop") != 0:
            raise ValueError("GIF frame count or loop metadata differs from the animation")
    if hashes != {file: digest(directory / file) for file in input_files}:
        raise ValueError("A baseline geometry, lattice, or generation script changed")
    temporary.replace(output)
    report = {
        "case": case, "status": "passed", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "illustrative geometry deformation; no CFD or volume-mesh deformation",
        "generator": "demo/animate_ffd.py", "generator_sha256": digest(Path(__file__)),
        "input_sha256": hashes, "mapping": "SU2 BSPLINE_UNIFORM; FFD.ControlLattice",
        "spline_orders": list(lattice.orders), "control_shape": list(lattice.shape),
        "geometry_points": len(xyz), "maximum_baseline_reconstruction_error": error,
        "parameter_coordinates_held_fixed": True, "frames_start_from_baseline": True,
        "motion_mode_names": names, "control_displacement_modes": modes.tolist(),
        "frame_coefficients": coefficients.tolist(),
        "maximum_surface_displacement_in_normalized_units": maximum_displacement,
        "sampled_jacobians_per_frame": len(sample), "minimum_sampled_jacobian_all_frames": minimum_jacobian,
        "fixed_camera": True, "equal_physical_coordinate_scale": True, "viewport": limits.tolist(),
        "wing_surface_grid": "none; geometry outlines only" if is_wing else "not applicable",
        "gif": {"file": output.name, "sha256": digest(output), "frames": actual_frames,
                "size_pixels": list(size), "duration_milliseconds": duration, "loop": "forever",
                "palette": "one shared palette for all frames"},
    }
    (directory / "animation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"{case}: saved {output.name}; {actual_frames} frames; {output.stat().st_size / 1e6:.2f} MB", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--frames", type=int, default=96)
    parser.add_argument("--fps", type=int, choices=(10, 20, 25), default=20)
    parser.add_argument("--dpi", type=int, default=120)
    args = parser.parse_args()
    if args.frames < 12 or args.dpi < 60:
        parser.error("Use at least 12 frames and 60 DPI")
    for case in dict.fromkeys(args.cases):
        run_case(case, args.frames, args.fps, args.dpi)


if __name__ == "__main__":
    main()
