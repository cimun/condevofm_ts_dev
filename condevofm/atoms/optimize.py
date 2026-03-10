"""
optimize.py

Implementation of the Optimizer class.
"""

import os
from typing import Optional

import ase
import torch
from ase.constraints import FixAtoms
from ase.io.jsonio import encode
from ase.optimize.optimize import Optimizer as AseOptimizer

from condevofm.atoms import (
    Constrainer,
    atoms_list_to_solutions,
    calculate_atoms_list,
    init_calc,
    solutions_to_atoms_list,
)


class Optimizer:
    """
    Applies only fixed constraints during optimization.
    Frozen atoms are excluded from sampling but can move in optimization.
    """

    def __init__(
        self,
        founder_atoms,
        constrainer: Constrainer,
        calc: str,
        optimizer: AseOptimizer,
        fmax: float,
        steps: int,
        logfile: str,
        multiproc: bool,
        n_proc: int,
        device: str,
        e_cutoff: float,
        progress_bar: bool = False,
        save_traj: bool = False,
        traj_path: Optional[str] = None,
        save_interval: Optional[int] = 10,
    ):
        self.founder_atoms = founder_atoms
        self.n_atoms = len(founder_atoms)

        # Get indices
        self.fixed_indices, self.frozen_indices, self.free_indices = (
            constrainer.get_all_indices(founder_atoms)
        )

        # Subsets
        self.fixed_atoms = (
            founder_atoms[self.fixed_indices].copy()
            if self.fixed_indices
            else None
        )
        self.frozen_atoms = (
            founder_atoms[self.frozen_indices].copy()
            if self.frozen_indices
            else None
        )
        self.free_atoms = founder_atoms[self.free_indices].copy()

        # Attach indices info to free_atoms
        info = (
            dict(self.free_atoms.info)
            if self.free_atoms.info is not None
            else {}
        )
        info["indices"] = list(self.free_indices)
        self.free_atoms.info = info

        # ES dimension = free only
        self.free_n_atoms = len(self.free_atoms)
        self.free_positions = torch.from_numpy(
            self.free_atoms.get_positions().flatten()
        )
        self.dimensions = self.free_positions.shape[0]

        # Movable in optimization = free + frozen
        self.movable_indices = self.free_indices + self.frozen_indices
        self.movable_atoms = founder_atoms[self.movable_indices].copy()
        self.movable_dim = len(self.movable_indices) * 3

        # Optimization params
        self.calc_str = calc
        self.calc = init_calc(calc, device)
        self.optimizer = optimizer
        self.fmax = fmax
        self.steps = steps
        self.logfile = logfile
        self.multiproc = multiproc
        self.n_proc = n_proc
        self.device = device
        self.e_cutoff = e_cutoff
        self.progress_bar = progress_bar

        self.save_traj = save_traj
        self.traj_path = traj_path
        if self.save_traj and self.traj_path is not None:
            os.makedirs(self.traj_path, exist_ok=True)
        self.save_interval = save_interval

    def refresh_indices(self, solutions):
        atoms_list = solutions_to_atoms_list(
            solutions=solutions,
            founder_atoms=self.founder_atoms,
            fixed_atoms=self.fixed_atoms,
            fixed_indices=self.fixed_indices,
            frozen_atoms=self.frozen_atoms,
            frozen_indices=self.frozen_indices,
            free_atoms=self.free_atoms,
            calc=self.calc,
        )
        if self.frozen_indices or self.fixed_indices:
            for atoms in atoms_list:
                all_constrained = list(
                    set(self.fixed_indices + self.frozen_indices)
                )
                atoms.set_constraint(FixAtoms(indices=all_constrained))
        solutions = atoms_list_to_solutions(
            atoms_list=atoms_list,
            free_indices=self.free_indices,
            dimensions=self.dimensions,
        )
        return solutions

    def optimize(self, solutions, gen=None):
        atoms_list = solutions_to_atoms_list(
            solutions=solutions,
            founder_atoms=self.founder_atoms,
            fixed_atoms=self.fixed_atoms,
            fixed_indices=self.fixed_indices,
            frozen_atoms=self.frozen_atoms,
            frozen_indices=self.frozen_indices,
            free_atoms=self.free_atoms,
            calc=self.calc,
            free_indices=self.free_indices,
        )
        optimized_atoms_list = calculate_atoms_list(
            atoms_list=atoms_list,
            func=self.optimize_atoms,
            desc="Optimization\t",
            multiproc=self.multiproc,
            n_proc=self.n_proc,
            progress_bar=self.progress_bar,
            kwargs={"steps": self.steps, "gen": gen},
        )
        solutions = atoms_list_to_solutions(
            atoms_list=optimized_atoms_list,
            free_indices=self.free_indices,
            dimensions=self.dimensions,
        )
        return solutions, optimized_atoms_list

    def optimize_atoms(
        self,
        atoms,
        steps,
        gen: Optional[int] = None,
        idx: int = 0,
        save_interval: Optional[int] = 10,
    ):
        if self.fixed_indices:
            atoms.set_constraint(FixAtoms(indices=self.fixed_indices))
        atoms.wrap()

        # If saving trajectories
        if save_interval < steps:
            save_interval = steps

        if self.save_traj and self.traj_path is not None and gen is not None:
            os.makedirs(self.traj_path, exist_ok=True)
            traj_filename = os.path.join(self.traj_path, f"gen_{gen}.traj")
            dyn = self.optimizer(
                atoms=atoms,
                logfile=self.logfile,
                trajectory=traj_filename,
                append_trajectory=True,
            )
        else:
            dyn = self.optimizer(atoms=atoms, logfile=self.logfile)

        dyn.run(fmax=self.fmax, steps=steps)
        dyn.atoms.wrap()
        return dyn.atoms

    def encode_params(self):
        params = {
            "founder_atoms": encode(self.founder_atoms),
            "fixed_atoms": encode(self.fixed_atoms),
            "frozen_atoms": encode(self.frozen_atoms),
            "free_atoms": encode(self.free_atoms),
            "fixed_indices": self.fixed_indices,
            "frozen_indices": self.frozen_indices,
            "free_indices": self.free_indices,
            "calc": self.calc_str,
            "multiproc": self.multiproc,
            "n_proc": self.n_proc,
            "progress_bar": self.progress_bar,
            "device": self.device,
            "e_cutoff": self.e_cutoff,
        }
        return params

    def __str__(self):
        return f"Optimizer(founder_atoms={self.founder_atoms}, fixed_indices={self.fixed_indices}, frozen_indices={self.frozen_indices}, free_indices={self.free_indices}, calc={self.calc}, optimizer={self.optimizer}, fmax={self.fmax:.5f}, steps={self.steps}, logfile={self.logfile}, multiproc={self.multiproc}, n_proc={self.n_proc}, device={self.device}, save_traj={self.save_traj}, traj_path={self.traj_path}, save_interval={self.save_interval})"

    __repr__ = __str__
