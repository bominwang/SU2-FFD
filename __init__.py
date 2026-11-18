"""Fitted FFD lattices for 2D airfoils and 3D wings, with native SU2 export.

Generate control grids with pyGeo, read existing PLOT3D lattices, and embed
SU2 boundary nodes in a uniform B-spline volume. CFD grids remain separate.
"""

from .lattice import ControlLattice
from .pygeo_adapter import generate_airfoil_plot3d, generate_wing_plot3d
from .su2 import export_su2, read_surface

__all__ = [
    "ControlLattice", "generate_airfoil_plot3d", "generate_wing_plot3d",
    "export_su2", "read_surface",
]
