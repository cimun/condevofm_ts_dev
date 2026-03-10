"""
evaluate.py

Implementation of evaluation functions for ase.atoms.Atoms objects.
"""

from typing import Callable

import numpy as np
import torch
from ase.io.jsonio import decode
from ase.visualize import view

from condevofm.atoms import (
    calculate_atoms_list,
    init_calc,
    solutions_to_atoms_list,
)


def get_potential_energy(atoms, kwargs=None):
    return atoms.get_potential_energy()


def evaluate_population_with_calc(
    population: torch.Tensor,
    obj_params: dict,
    evaluate_atoms: Callable,
    evaluate_kwargs: dict = {},
    filter: bool = True,
    show: bool = False,
    penalty: float = -1e6,
) -> torch.Tensor:

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

    atoms_list = obj_params.get("optimized_atoms_list", None)
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

    if evaluate_atoms == "get_potential_energy":
        evaluate_atoms_func = get_potential_energy
    else:
        raise ValueError(
            "Currently only 'get_potential_energy' implemented as evaluation function."
        )

    fitness = calculate_atoms_list(
        atoms_list,
        func=evaluate_atoms_func,
        desc="Evaluation\t",
        multiproc=obj_params["multiproc"],
        n_proc=obj_params["n_proc"],
        progress_bar=obj_params["progress_bar"],
        kwargs=evaluate_kwargs,
    )

    if filter and evaluate_atoms_func == get_potential_energy:
        updated_fitness = []
        for energy in fitness:
            if energy < obj_params["e_cutoff"]:
                updated_fitness.append(np.abs(penalty * energy))
            else:
                updated_fitness.append(energy)
        fitness = updated_fitness

    if show:
        atoms_list_sorted = [
            atoms for _, atoms in sorted(zip(fitness, atoms_list))
        ]
        view(atoms_list_sorted)

    return torch.Tensor(fitness)
