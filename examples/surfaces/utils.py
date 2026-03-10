"""
surfaces/utils.py

Implementation of gold-sulfur surface functionalities.
"""

import matplotlib.pyplot as plt
import numpy as np
from ase.build import fcc111
from ase.visualize import view
from ase.visualize.plot import plot_atoms


def generate_aus_surface(
    n_au_layers: int,
    n_au_atoms: int,
    n_s_atoms: int,
    repeat_x: int = 3,
    repeat_y: int = 3,
    vacuum: float = 20.0,
    height: float = 2.365,
    transform: bool = False,
    plot: bool = False,
    show: bool = False,
):
    if transform:
        au_surface = fcc111("Au", size=(1, 1, n_au_layers))
        au_surface.cell[2, 2] = vacuum

        skew_factor = repeat_y * 2 / 3
        T = np.array(
            [[repeat_x, 0, 0], [-skew_factor, repeat_y, 0], [0, 0, 1]]
        )
        cell = T @ au_surface.cell

        au_surface = au_surface.repeat((repeat_x, repeat_y, 1))
        au_surface.set_cell(cell)
        au_surface.wrap()
    else:
        au_surface = fcc111("Au", size=(repeat_x, repeat_y, n_au_layers))
        au_surface.cell[2, 2] = vacuum
        cell = au_surface.cell

    threshold = au_surface.get_positions()[:, 2].max()

    shift = n_au_layers * height + 0.5

    atoms_s = fcc111("S", a=2.0, size=(1, 1, 1))
    atoms_s.positions[:, 2] += shift
    atoms_s = atoms_s.repeat((n_s_atoms, 1, 1))

    atoms_au = fcc111("Au", a=2.0, size=(1, 1, 1))
    atoms_au.positions[:, 2] += shift
    atoms_au.positions[:, 1] += cell[1, 1] * 0.50
    atoms_au = atoms_au.repeat((n_au_atoms, 1, 1))

    au_surface = au_surface.copy() + atoms_s.copy()
    au_surface = au_surface.copy() + atoms_au.copy()

    rep = f"AuL{n_au_layers}-Au{n_au_atoms}-S{n_s_atoms}_{repeat_x}x{repeat_y}"
    if transform:
        rep += "T"

    if plot:
        fig, ax = plt.subplots()
        plot_atoms(au_surface, ax=ax)
        ax.axis("off")

    if show:
        view(au_surface)

    return au_surface, threshold, rep
