"""
monolayers/utils.py

Implementation of 2D monolayer and multilayer functionalities.
"""

import matplotlib.pyplot as plt
import numpy as np
from ase.io import read
from ase.visualize import view
from ase.visualize.plot import plot_atoms


def load_mos2_monolayer(
    structure_path: str,
    plot: bool = False,
    show: bool = False,
):
    atoms = read(structure_path)

    n_atoms = len(atoms.get_atomic_numbers())
    dimensions = n_atoms * 3

    symbols = atoms.symbols
    rep = structure_path.split("/")[-1].split(".")[0]

    if "5x5x1" in structure_path:
        center = np.array([0.46666, 0.53333, 0.49085])
        radius = 7.0
    elif "6x6x1" in structure_path:
        center = np.array([0.44443, 0.55553, 0.49085])
        radius = 7.0
    elif "7x7x1" in structure_path:
        center = np.array([0.42775, 0.56933, 0.49085])
        radius = 7.0
    elif "8x8x1" in structure_path:
        center = np.array([0.41667, 0.58335, 0.49085])
        radius = 9.0

    if plot:
        fig, ax = plt.subplots()
        plot_atoms(atoms, ax=ax)
        ax.axis("off")

    if show:
        view(atoms)

    return atoms, dimensions, symbols, rep, center, radius
