"""Control-grid interchange and initial embedding using SU2's spline basis.

Geometry calculations use float64, as do the existing SU2 mesh coordinates.
This module initializes the baseline; SU2 handles subsequent deformations.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import least_squares, root
from scipy.spatial import cKDTree


@dataclass
class ControlLattice:
    """Single tensor-product volume with physical coordinates [ni,nj,nk,3].

    ``orders`` are spline degrees plus one, NOT SU2's FFD_DEGREE entries.
    SU2's BSPLINE_UNIFORM uses open uniform parameter knots.
    """

    points: np.ndarray
    orders: tuple[int, int, int]
    tag: str = "FITTED_FFD"

    def __post_init__(self) -> None:
        self.points = np.array(self.points, dtype=np.float64, copy=True)
        self.orders = tuple(self.orders)
        if self.points.ndim != 4 or self.points.shape[-1] != 3:
            raise ValueError("Control points must have shape (ni,nj,nk,3)")
        if not np.isfinite(self.points).all() or min(self.shape) < 2:
            raise ValueError("Expected finite coordinates and >=2 controls per axis")
        if len(self.orders) != 3 or any(
            not isinstance(k, (int, np.integer)) or not 2 <= k <= n
            for k, n in zip(self.orders, self.shape)
        ):
            raise ValueError("Each spline order must be an integer in [2, control count]")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", self.tag):
            raise ValueError("FFD tag must contain only letters, digits, '_' or '-'")
        self._splines = []
        for n, k in zip(self.shape, self.orders):
            knots = np.r_[np.zeros(k), np.arange(1, n-k+1)/(n-k+1), np.ones(k)]
            self._splines.append(BSpline(knots, np.eye(n), k-1))

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.points.shape[:3]

    def evaluate(self, uvw: np.ndarray) -> np.ndarray:
        """Evaluate physical coordinates for a point or an array of points."""
        uvw = np.asarray(uvw, dtype=np.float64)
        b = [s(uvw[..., i]) for i, s in enumerate(self._splines)]
        return np.einsum("...i,...j,...k,ijkc->...c", *b, self.points, optimize=True)

    def jacobian(self, uvw: np.ndarray) -> np.ndarray:
        """Return physical-coordinate by parameter-coordinate Jacobians."""
        uvw = np.asarray(uvw, dtype=np.float64)
        b = [s(uvw[..., i]) for i, s in enumerate(self._splines)]
        columns = []
        for axis, spline in enumerate(self._splines):
            factors = b.copy()
            factors[axis] = spline.derivative()(uvw[..., axis])
            columns.append(np.einsum("...i,...j,...k,ijkc->...c", *factors,
                                     self.points, optimize=True))
        return np.stack(columns, axis=-1)

    def embed(self, xyz: np.ndarray, tolerance: float = 1e-10) -> tuple[np.ndarray, float]:
        """Invert the baseline mapping and reject outside/unreconstructed points.

        Nearest sampled physical points seed the small nonlinear inversions.
        ``tolerance`` is a reconstruction tolerance in mesh coordinate units.
        """
        xyz = np.asarray(xyz, dtype=np.float64)
        if xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz) or not np.isfinite(xyz).all():
            raise ValueError("Expected a nonempty finite (N,3) surface point array")
        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("Embedding tolerance must be positive")
        samples = np.stack(np.meshgrid(*[
            np.linspace(0, 1, min(2*n-1, 21)) for n in self.shape
        ], indexing="ij"), axis=-1).reshape(-1, 3)
        sampled_xyz = self.evaluate(samples)
        # A consistent nonzero sampled Jacobian rejects collapsed/folded seeds.
        jac = self.jacobian(samples)
        det = np.linalg.det(jac)
        relative_det = det / np.maximum(np.prod(np.linalg.norm(jac, axis=-2), axis=-1),
                                       np.finfo(float).tiny)
        if np.any(np.abs(relative_det) < 1e-10) or (np.any(det > 0) and np.any(det < 0)):
            raise ValueError("Control lattice has a singular or folded sampled mapping")
        scale = np.ptp(sampled_xyz, axis=0)
        if np.any(scale <= 0):
            raise ValueError("Control lattice must span all three coordinates")
        tree = cKDTree(sampled_xyz/scale)
        seeds = samples[tree.query(xyz/scale)[1]]
        parameters = np.empty_like(xyz)
        for i, (point, seed) in enumerate(zip(xyz, seeds)):
            def fun(t):
                return (self.evaluate(t)-point)/scale

            def derivative(t):
                return self.jacobian(t)/scale[:, None]

            solution = root(fun, seed, jac=derivative, options={"xtol": 1e-10})
            p = solution.x
            if (not np.isfinite(p).all() or np.any(p < -1e-12) or np.any(p > 1+1e-12)
                    or np.max(np.abs(self.evaluate(p)-point)) > tolerance):
                p = least_squares(fun, np.clip(seed, 1e-8, 1-1e-8), jac=derivative,
                                  bounds=(0, 1), xtol=1e-14, ftol=1e-14,
                                  gtol=1e-14, max_nfev=150).x
            if np.any(p < -1e-12) or np.any(p > 1+1e-12):
                raise ValueError(f"Surface point {i} is outside the control lattice")
            # Only parameter endpoint roundoff is rounded; geometry is never moved.
            p = np.clip(p, 0, 1)
            error = float(np.max(np.abs(self.evaluate(p)-point)))
            if not np.isfinite(error) or error > tolerance:
                raise ValueError(f"Surface point {i} cannot be embedded: residual={error:g}")
            parameters[i] = p
        error = float(np.max(np.abs(self.evaluate(parameters)-xyz)))
        return parameters, error

    @classmethod
    def from_plot3d(cls, path: str | Path, *, orders: tuple[int, int, int],
                    tag: str = "FITTED_FFD") -> ControlLattice:
        """Read the single-volume ASCII PLOT3D format produced by pyGeo."""
        tokens = Path(path).read_text().split()
        if not tokens or int(tokens[0]) != 1:
            raise ValueError("This interface accepts one ASCII PLOT3D volume")
        shape = tuple(int(v) for v in tokens[1:4])
        if len(shape) != 3 or min(shape) < 2:
            raise ValueError("Invalid PLOT3D dimensions")
        values = np.array([float(v.replace("D", "E").replace("d", "e"))
                           for v in tokens[4:]])
        count = int(np.prod(shape))
        if len(values) != 3*count:
            raise ValueError("PLOT3D coordinate count does not match dimensions")
        points = np.stack([v.reshape(shape, order="F") for v in values.reshape(3, count)], axis=-1)
        return cls(points, orders, tag)

    def write_plot3d(self, path: str | Path) -> None:
        """Write an ASCII control lattice, retaining i-fastest ordering."""
        lines = ["1", " ".join(map(str, self.shape))]
        for axis in range(3):
            values = self.points[..., axis].ravel(order="F")
            lines.extend(" ".join(f"{v:.17g}" for v in row)
                         for row in np.array_split(values, max(1, (len(values)+5)//6)))
        Path(path).write_text("\n".join(lines)+"\n")
