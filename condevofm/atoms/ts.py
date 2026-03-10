"""
ts.py

Implementation of transition-state utility functions for ase.atoms.Atoms objects.
"""

import os
import tempfile

import numpy as np
import torch
from ase.io.jsonio import decode
from ase.vibrations import Vibrations
from ase.visualize import view

from .calculate import (
    calculate_atoms_list,
    init_calc,
    solutions_to_atoms_list,
)


def check_saddle_point(atoms, eval_tol=0.1):
    """
    Quantifies saddle point quality using both mass-weighted frequencies
    and raw Hessian eigenvalues.

    Safe for multiprocessing - uses temporary directories to avoid cache
    conflicts.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        original_cwd = os.getcwd()
        os.chdir(tmpdir)

        try:
            vib = Vibrations(atoms)
            vib.run()

            freqs = vib.get_frequencies()
            imaginary_freqs = [f for f in freqs if np.iscomplex(f)]

            hessian_raw = vib.get_vibrations().get_hessian_2d()
            raw_evals = np.linalg.eigvalsh(hessian_raw)
            negative_evals = [val for val in raw_evals if val < -eval_tol]

            results = {
                "is_saddle_point": len(negative_evals) == 1,
                "n_imaginary_freqs": len(imaginary_freqs),
                "imaginary_freq_val": imaginary_freqs[0]
                if imaginary_freqs
                else None,
                "all_freqs_cm1": np.round(freqs, 3),
                "n_negative_evals": len(negative_evals),
                "raw_eigenvalues": np.round(raw_evals, 3),
                "lowest_eval": raw_evals[0],
            }

            vib.clean()
            return results

        finally:
            os.chdir(original_cwd)


def get_lowest_eigenvalue(atoms, kwargs=None):
    """
    Helper function to extract lowest eigenvalue from an atoms object.
    Used for multiprocessing in minimize_eigenval.
    """
    return check_saddle_point(atoms)["lowest_eval"]


def minimize_eigenval(
    population: torch.Tensor,
    obj_params: dict,
    filter: bool = True,
    show: bool = False,
    bad_eigenval: float = 1e6,
) -> torch.Tensor:
    """
    Evaluate population based on minimizing the lowest eigenvalue from
    saddle point analysis.
    """
    founder_atoms = decode(obj_params["founder_atoms"])
    fixed_atoms = decode(obj_params["fixed_atoms"])
    free_atoms = decode(obj_params["free_atoms"])

    frozen_atoms_raw = obj_params.get("frozen_atoms", None)
    frozen_atoms = (
        decode(frozen_atoms_raw) if frozen_atoms_raw is not None else None
    )
    frozen_indices = obj_params.get("frozen_indices", [])

    free_indices = obj_params.get("free_indices")
    if free_indices is None:
        if free_atoms.info is None or "indices" not in free_atoms.info:
            raise ValueError(
                "free_indices missing: set obj_params['free_indices'] or free_atoms.info['indices']."
            )
        free_indices = free_atoms.info["indices"]

    calc = init_calc(obj_params["calc"], obj_params["device"])

    atoms_list = obj_params.get("relaxed_atoms_list", None)
    if atoms_list is None:
        atoms_list = solutions_to_atoms_list(
            solutions=population,
            founder_atoms=founder_atoms,
            fixed_atoms=fixed_atoms,
            fixed_indices=obj_params["fixed_indices"],
            free_atoms=free_atoms,
            calc=calc,
            frozen_atoms=frozen_atoms,
            frozen_indices=frozen_indices,
        )

    eigenvalues = calculate_atoms_list(
        atoms_list,
        func=get_lowest_eigenvalue,
        desc="Eigenvalue Evaluation",
        multiproc=obj_params["multiproc"],
        n_proc=obj_params["n_proc"],
        progress_bar=obj_params["progress_bar"],
        kwargs={},
    )

    if filter:
        eval_cutoff = obj_params.get("eval_cutoff", -0.1)
        updated_eigenvalues = []
        for atoms, eigenval in zip(atoms_list, eigenvalues):
            if eigenval < eval_cutoff:
                updated_eigenvalues.append(eigenval)
            else:
                updated_eigenvalues.append(bad_eigenval)
        eigenvalues = updated_eigenvalues

    if show:
        atoms_list_sorted = [
            atoms for _, atoms in sorted(zip(eigenvalues, atoms_list))
        ]
        view(atoms_list_sorted)

    return torch.Tensor(eigenvalues)
