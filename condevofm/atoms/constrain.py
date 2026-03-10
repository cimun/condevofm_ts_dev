"""
constrain.py

Implementation of the Constrainer, IndexConstrainer, ThresholdConstrainer and
SphereConstrainer classes.
"""

from abc import ABC, abstractmethod
from typing import Tuple

import ase
import numpy as np


class Constrainer(ABC):
    """
    Constrainer class with support for:
    - fixed atoms (always constrained, never move)
    - frozen atoms (not modified during candidate generation, but can relax)
    - free atoms (modifiable and relaxable)
    """

    def get_distances_from_position(
        self, atoms: ase.atoms.Atoms, position: Tuple[float, float, float]
    ):
        atoms_copy = atoms.copy()
        atom_at_position = ase.atoms.Atom(symbol="H", position=position)
        atoms_copy.append(atom_at_position)
        distances = atoms_copy.get_all_distances(mic=True)[-1]
        return distances[:-1]  # exclude defect atom

    def get_distances_from_scaled_position(
        self,
        atoms: ase.atoms.Atoms,
        scaled_position: Tuple[float, float, float],
    ):
        return self.get_distances_from_position(
            atoms=atoms, position=scaled_position @ atoms.cell
        )

    @abstractmethod
    def get_fixed_indices(self):
        pass

    @abstractmethod
    def get_frozen_indices(self):
        pass

    def get_all_indices(self, atoms):
        fixed_indices = list(self.get_fixed_indices(atoms))
        frozen_indices = list(self.get_frozen_indices(atoms))
        fixed_set = set(fixed_indices)
        frozen_set = set(frozen_indices) - fixed_set
        all_constrained = fixed_set | frozen_set
        free_indices = [
            atom.index for atom in atoms if atom.index not in all_constrained
        ]
        fixed_indices = sorted(fixed_set)
        frozen_indices = sorted(frozen_set)
        free_indices = sorted(free_indices)
        return fixed_indices, frozen_indices, free_indices


class IndexConstrainer(Constrainer):
    """
    IndexConstrainer docstring
    """

    def __init__(
        self,
        fix_indices=None,
        freeze_indices=None,
    ):
        self.fix_indices = fix_indices or []
        self.freeze_indices = freeze_indices or []

    def get_indices(self, atoms, indices):
        return [atom.index for atom in atoms if atom.index in indices]

    def get_fixed_indices(self, atoms):
        return self.get_indices(atoms, self.fix_indices)

    def get_frozen_indices(self, atoms):
        return self.get_indices(atoms, self.freeze_indices)

    def __str__(self):
        return f"IndexConstrainer(fix_indices={self.fix_indices}, freeze_indices={self.freeze_indices})"

    __repr__ = __str__


class ThresholdConstrainer(Constrainer):
    """
    ThresholdConstrainer docstring
    """

    def __init__(
        self,
        fix_axis=None,
        fix_threshold=None,
        freeze_axis=None,
        freeze_threshold=None,
    ):
        self.fix_axis = fix_axis
        self.fix_threshold = fix_threshold
        self.freeze_axis = freeze_axis
        self.freeze_threshold = freeze_threshold

    def get_indices(self, atoms, axis, threshold):
        return [
            atom.index for atom in atoms if atom.position[axis] < threshold
        ]

    def get_fixed_indices(self, atoms):
        return self.get_indices(atoms, self.fix_axis, self.fix_threshold)

    def get_frozen_indices(self, atoms):
        return self.get_indices(atoms, self.freeze_axis, self.freeze_threshold)

    def __str__(self):
        return f"ThresholdConstrainer(fix_axis={self.fix_axis}, fix_threshold={self.fix_threshold}, freeze_axis={self.freeze_axis}, freeze_threshold={self.freeze_threshold})"

    __repr__ = __str__


class SphereConstrainer(Constrainer):
    """
    SphereConstrainer docstring
    """

    def __init__(
        self,
        fix_center=None,
        fix_radius=None,
        freeze_center=None,
        freeze_radius=None,
    ):
        self.fix_center = fix_center
        self.fix_radius = fix_radius
        self.freeze_center = freeze_center
        self.freeze_radius = freeze_radius

    def get_indices(self, atoms, center, radius):
        indices = []
        if np.all(center) and radius:
            dist = self.get_distances_from_scaled_position(
                atoms=atoms, scaled_position=center
            )
            indices = np.where(dist > radius)[0].tolist()
        return indices

    def get_fixed_indices(self, atoms):
        return self.get_indices(atoms, self.fix_center, self.fix_radius)

    def get_frozen_indices(self, atoms):
        return self.get_indices(atoms, self.freeze_center, self.freeze_radius)

    def __str__(self):
        return f"SphereConstrainer(fix_center={self.fix_center}, fix_radius={self.fix_radius}, freeze_center={self.freeze_center}, freeze_radius={self.freeze_radius})"

    __repr__ = __str__
