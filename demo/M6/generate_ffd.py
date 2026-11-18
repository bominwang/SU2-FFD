"""Generate an ONERA M6 fitted FFD directly from the official MACH-Aero CAD.

Run this independent example with Python and MDOLab pyGeo installed.
The IGES file is the input; surface panels are evaluated from its NURBS.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.collections import LineCollection, PolyCollection
import numpy as np
from scipy.interpolate import BSpline
import pygeo
import pyspline

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from FFD import ControlLattice, generate_wing_plot3d

CASE = "ONERA M6"
CAD_FILENAME = "geometry.igs"
# Directory-entry identifiers in the original IGES. All ten are wing patches.
PATCH_IDS = (1, 5, 9, 13, 17, 21, 25, 29, 33, 37)
MAIN_PATCH_IDS = (13, 17)
N_SPAN = 9
N_CHORD = 8


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_cad(path):
    """Read the selected untrimmed IGES type-128 rational B-spline patches.

    Parameter payload occupies columns 1--64. Sampling is restricted to the
    intersection of the declared range and the valid B-spline base interval;
    this avoids extrapolating the padded end knots in the OpenVSP IGES.
    """
    import gzip
    raw = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
    lines = raw.decode("ascii").splitlines()
    directory = [line for line in lines if len(line)>72 and line[72] == "D"]
    parameter_lines = {int(line[73:80]):line[:64] for line in lines
                       if len(line)>72 and line[72] == "P"}
    patches = []
    for first, second in zip(directory[::2], directory[1::2]):
        entity = int(first[73:80])
        if entity not in PATCH_IDS:
            continue
        if int(first[:8]) != 128:
            raise ValueError(f"CAD patch {entity} is not a NURBS surface")
        if int(first[48:56]) != 0:
            raise ValueError("This example expects CAD patches without entity transforms")
        start, count = int(first[8:16]), int(second[24:32])
        payload = "".join(parameter_lines[k] for k in range(start, start+count))
        data = np.array([float(t.replace("D","E")) for t in
                         payload.split(";")[0].split(",") if t.strip()])
        nu, nv = data[1:3].astype(int)+1
        ku, kv = data[3:5].astype(int)+1
        index = 10
        tu = data[index:index+nu+ku]; index += nu+ku
        tv = data[index:index+nv+kv]; index += nv+kv
        weights = data[index:index+nu*nv].reshape(nv,nu).T; index += nu*nv
        coefficients = data[index:index+3*nu*nv].reshape(nv,nu,3).transpose(1,0,2)
        index += 3*nu*nv
        declared = data[index:index+4]
        domain = np.array([max(declared[0],tu[ku-1]), min(declared[1],tu[nu]),
                           max(declared[2],tv[kv-1]), min(declared[3],tv[nv])])
        if np.any(domain[[1,3]] <= domain[[0,2]]) or np.any(weights <= 0):
            raise ValueError(f"Invalid CAD domain or weights on patch {entity}")
        patches.append(dict(entity=entity, tu=tu, tv=tv, ku=int(ku), kv=int(kv),
                            weights=weights, coefficients=coefficients,
                            declared_domain=declared.tolist(), domain=domain))
    if {p["entity"] for p in patches} != set(PATCH_IDS):
        raise ValueError("Expected wing CAD patches are missing")
    return patches


def evaluate_patch(patch, nu, nv):
    """Evaluate the original rational CAD basis without refitting its geometry."""
    u0,u1,v0,v1 = patch["domain"]
    # Cosine spacing resolves the curved leading edge in the chord direction.
    u = np.linspace(u0,u1,nu)
    v = v0+(v1-v0)*.5*(1-np.cos(np.linspace(0,np.pi,nv)))
    shape = patch["weights"].shape
    bu = BSpline(patch["tu"],np.eye(shape[0]),patch["ku"]-1,extrapolate=False)(u)
    bv = BSpline(patch["tv"],np.eye(shape[1]),patch["kv"]-1,extrapolate=False)(v)
    denominator = np.einsum("ui,ij,vj->uv",bu,patch["weights"],bv,optimize=True)
    weighted = patch["coefficients"]*patch["weights"][...,None]
    numerator = np.stack([bu@weighted[...,axis]@bv.T for axis in range(3)],axis=-1)
    points = numerator/denominator[...,None]
    if not np.isfinite(points).all():
        raise ValueError("Nonfinite CAD evaluation")
    return points


def prepare_geometry(patches):
    """Sample CAD for projection, retaining a single isotropic normalization."""
    grids = []
    for patch in patches:
        main = patch["entity"] in MAIN_PATCH_IDS
        nu,nv = (33,97) if main else ((9,97) if patch["coefficients"].shape[1]>10 else (5,5))
        grids.append(evaluate_patch(patch,nu,nv))
    main_grid = grids[PATCH_IDS.index(MAIN_PATCH_IDS[1])]
    root = main_grid[0]
    origin = root[np.argmin(root[:,0])].copy()
    origin[2] = 0.
    chord = float(np.ptp(root[:,0]))
    grids = [(g-origin)/chord for g in grids]
    main_grid = grids[PATCH_IDS.index(MAIN_PATCH_IDS[1])]
    le = np.array([row[np.argmin(row[:,0])] for row in main_grid])
    te = np.array([row[np.argmax(row[:,0])] for row in main_grid])
    # References lie slightly inside the actual main wing surface.
    stations = np.linspace(le[0,1]+1e-5,le[-1,1]-1e-5,3)
    leading = np.column_stack([np.interp(stations,le[:,1],le[:,axis]) for axis in range(3)])
    trailing = np.column_stack([np.interp(stations,te[:,1],te[:,axis]) for axis in range(3)])
    delta = trailing-leading
    leading += 1e-4*delta
    trailing -= 1e-4*delta
    quads,triangles = [],[]
    for grid in grids:
        a,b,c,d = grid[:-1,:-1],grid[1:,:-1],grid[1:,1:],grid[:-1,1:]
        quads.extend(np.stack([a,b,c,d],axis=-2).reshape(-1,4,3))
        triangles.extend(np.stack([a,b,c],axis=-2).reshape(-1,3,3))
        triangles.extend(np.stack([a,c,d],axis=-2).reshape(-1,3,3))
    points = np.unique(np.concatenate([g.reshape(-1,3) for g in grids]),axis=0)
    triangles = np.asarray(triangles)
    area = np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],
                                   triangles[:,2]-triangles[:,0]),axis=1)
    triangles = triangles[area>1e-14]
    return points,triangles,np.asarray(quads),leading,trailing,dict(
        origin_original_units=origin.tolist(), uniform_length_scale=chord,
        coordinates="(original - origin) / root_chord",
        surface_representation="direct rational IGES CAD tessellation",
        sampled_patch_shapes=[list(g.shape[:2]) for g in grids])


def geometry_outlines(patches, geometry_report):
    """Return only wing edge and root/tip profile curves."""
    by_id = {p["entity"]:p for p in patches}
    curves = []
    for entity in (13,17):
        grid = evaluate_patch(by_id[entity],65,161)
        curves.extend([grid[0],grid[:,0],grid[:,-1]])
    for entity in (29,33):
        cap = evaluate_patch(by_id[entity],25,161)
        curves.extend([cap[:,0],cap[:,-1],cap[-1]])
    origin = np.asarray(geometry_report["origin_original_units"])
    scale = geometry_report["uniform_length_scale"]
    return [(curve-origin)/scale for curve in curves]


def plot_ffd(directory, quads, lattice, patches, geometry_report):
    """Orthographic CAD view; the camera rotates all three axes rigidly."""
    font = Path("/mnt/c/Windows/Fonts/times.ttf")
    family = "DejaVu Serif"
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        family = font_manager.FontProperties(fname=str(font)).get_name()
    elev,azim,roll = np.deg2rad([28,-25,25])
    right = np.array([-np.sin(azim),np.cos(azim),0])
    up = np.array([-np.sin(elev)*np.cos(azim),-np.sin(elev)*np.sin(azim),np.cos(elev)])
    depth = np.array([np.cos(elev)*np.cos(azim),np.cos(elev)*np.sin(azim),np.sin(elev)])
    rotation = np.array([[np.cos(roll),-np.sin(roll)],[np.sin(roll),np.cos(roll)]])
    camera = np.vstack([rotation@np.vstack([right,up]),depth])
    if not np.allclose(camera@camera.T,np.eye(3),atol=1e-14):
        raise ValueError("Camera must preserve equal physical scales")
    def project(x):
        return np.asarray(x)@camera.T
    control = lattice.points
    segments,panels = [],[]
    for axis in range(3):
        for index in np.ndindex(control.shape[:3]):
            other = list(index); other[axis] += 1
            if other[axis]<control.shape[axis]:
                segments.append([control[index],control[tuple(other)]])
        for end in (0,control.shape[axis]-1):
            face = np.take(control,end,axis=axis)
            for i in range(face.shape[0]-1):
                for j in range(face.shape[1]-1):
                    panels.append(face[[i,i+1,i+1,i],[j,j,j+1,j+1]])
    flat = control.reshape(-1,3)
    all_points = np.vstack([quads.reshape(-1,3),flat])
    bounds = np.array([project(all_points)[:,:2].min(0),project(all_points)[:,:2].max(0)])
    span = bounds[1]-bounds[0]
    pad = .045*max(span)
    limits = bounds+np.array([[-pad,-pad],[pad,pad]])
    # A 2D equal-aspect orthographic drawing avoids mplot3d's implicit clipping.
    style = {"font.family":family,"font.size":18,"svg.fonttype":"none",
             "savefig.facecolor":"white"}
    with plt.rc_context(style):
        fig,ax = plt.subplots(figsize=(9,6.4))
        fig.subplots_adjust(left=.015,right=.985,bottom=.02,top=.90)
        outlines = geometry_outlines(patches,geometry_report)
        wing = project(quads)
        shell = project(np.array(panels))
        polygons = list(wing[...,:2])+list(shell[...,:2])
        depths = np.r_[wing[...,2].mean(1),shell[...,2].mean(1)]
        facecolors = np.vstack([np.tile([.72,.76,.80,1.],(len(wing),1)),
                                np.tile([0,.184,.655,.12],(len(shell),1))])
        edgecolors = np.vstack([np.tile([0.,0.,0.,0.],(len(wing),1)),
                                np.tile([0,.184,.655,0.],(len(shell),1))])
        order = np.argsort(depths)
        ax.add_collection(PolyCollection([polygons[i] for i in order],
            facecolors=facecolors[order],edgecolors=edgecolors[order],
            linewidths=0,antialiaseds=False))
        ax.add_collection(LineCollection([project(curve)[:,:2] for curve in outlines],
                                         colors="black",linewidths=.75,alpha=.85))
        ax.add_collection(LineCollection(project(np.array(segments))[...,:2],
                                         colors="#002FA7",linewidths=.85,alpha=.88))
        dots = project(flat)
        ax.scatter(dots[:,0],dots[:,1],s=17,c="#252B33",
                   edgecolors="black",linewidths=.4,zorder=5)
        ax.set_xlim(*limits[:,0]); ax.set_ylim(*limits[:,1])
        ax.set_aspect("equal",adjustable="box"); ax.set_axis_off()
        fig.text(.045,.965,CASE,fontsize=22,ha="left",va="top")
        triad = fig.add_axes([.80,.785,.18,.20])
        orientation = project(np.eye(3))[:,:2]
        for label,end in zip("XYZ",orientation):
            triad.annotate("",xy=end,xytext=(0,0),arrowprops=dict(arrowstyle="-",lw=1.2))
            triad.text(*(end*1.18),label,fontsize=18,ha="center",va="center")
        triad.set(xlim=(-1.25,1.25),ylim=(-1.25,1.25),aspect="equal")
        triad.set_axis_off()
        for extension in ("png","svg"):
            fig.savefig(directory/f"ffd.{extension}",dpi=300)
        plt.close(fig)
    return dict(projection="orthographic",physical_axis_scales="equal",
                camera_elevation_degrees=28,camera_azimuth_degrees=-25,camera_roll_degrees=25,
                control_color="#002FA7",enclosure_opacity=.12,
                surface_grid="none; wing outlines and root/tip profiles only",
                surface_outline_curves=len(outlines),
                files=["ffd.png","ffd.svg"],size_inches=[9,6.4],png_dpi=300)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot-only",action="store_true")
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent
    path = directory/CAD_FILENAME
    patches = read_cad(path)
    points,triangles,quads,leading,trailing,geometry_report = prepare_geometry(patches)
    output = directory/"ffd.xyz"
    parameters = dict(n_span=N_SPAN,n_chord=N_CHORD,absolute_margins=[.035,.055,.025],
                      relative_margins=[.01,0.,.05],lift_index=3)
    if not args.plot_only:
        print(f"{CASE}: generating FFD from {len(patches)} CAD patches...",flush=True)
        generate_wing_plot3d(triangles,output,surface_format="point-point",
                            leading_edge=leading,trailing_edge=trailing,
                            **parameters,overwrite=True)
    lattice = ControlLattice.from_plot3d(output,orders=(4,2,2),tag="M6_FFD")
    if args.plot_only:
        report = json.loads((directory/"validation.json").read_text())
    else:
        print(f"Checking {len(points)} CAD sample points...",flush=True)
        uvw,error = lattice.embed(points,tolerance=1e-9)
        samples = np.stack(np.meshgrid(np.linspace(0,1,31),np.linspace(0,1,33),
                                      [0,.5,1],indexing="ij"),axis=-1).reshape(-1,3)
        jac = lattice.jacobian(samples)
        det = np.linalg.det(jac)
        scaled = det/np.prod(np.linalg.norm(jac,axis=-2),axis=-1)
        checks = dict(all_CAD_sample_points_embedded=bool(error<=1e-9),
                      all_parameters_inside=bool(np.all((uvw>=0)&(uvw<=1))),
                      positive_sampled_jacobians=bool(np.all(det>0)),
                      nonsingular_sampled_jacobians=bool(np.min(abs(scaled))>1e-10))
        if not all(checks.values()):
            raise ValueError(checks)
        report = dict(case=CASE,status="passed",
            scope="baseline CAD and sampled FFD embedding",
            generated_at_utc=datetime.now(timezone.utc).isoformat(),
            source=json.loads((directory/"source.json").read_text()),
            input_sha256={CAD_FILENAME:sha256(path)},geometry=geometry_report,
            cad_patches=[dict(entity=p["entity"],domain=p["domain"].tolist(),
                             declared_domain=p["declared_domain"]) for p in patches],
            generation_parameters=parameters,backend="pygeo.geo_utils.createFittedWingFFD",
            versions=dict(pygeo=pygeo.__version__,pyspline=pyspline.__version__),
            python_executable=sys.executable,control_shape=list(lattice.shape),
            spline_orders=list(lattice.orders),CAD_sample_points=len(points),
            projection_triangles=len(triangles),maximum_embedding_error=error,
            jacobian_samples=len(samples),minimum_sampled_jacobian=float(det.min()),
            minimum_sampled_scaled_jacobian=float(abs(scaled).min()),checks=checks)
    report["figure"] = plot_ffd(directory,quads,lattice,patches,geometry_report)
    report["output_sha256"] = {f:sha256(directory/f) for f in ("ffd.xyz","ffd.png","ffd.svg")}
    (directory/"validation.json").write_text(json.dumps(report,indent=2)+"\n")
    print(f"{CASE}: {report['status']}; {lattice.shape}; embedding error "
          f"{report['maximum_embedding_error']:.3e}",flush=True)


if __name__ == "__main__":
    main()
