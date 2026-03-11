"""
charlx.py

Implementation of the CHARLX class.
"""

import torch
from condevo.es import CHARLES


class CHARLX(CHARLES):
    """
    CHARLX with support for fixed (always constrained) and frozen
    (sampling-only) atoms.
    """

    def __init__(self, constrainer, optimizer, n_gens, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.constrainer = constrainer
        self.optimizer = optimizer
        self.optimized_atoms_list = None
        self.n_gens = n_gens
        self.curr_gen = 0

    def ask(self):
        self.solutions = super().ask()
        self.solutions = self.optimizer.refresh_indices(self.solutions)
        if not isinstance(self.solutions, torch.Tensor):
            self.solutions = torch.as_tensor(self.solutions)
        self.solutions = self.solutions.to(self.device)

        self.solutions, self.optimized_atoms_list = self.optimizer.optimize(
            self.solutions, gen=self.curr_gen
        )
        if not isinstance(self.solutions, torch.Tensor):
            self.solutions = torch.as_tensor(self.solutions)
        self.solutions = self.solutions.to(self.device)

        self.curr_gen += 1
        return self.solutions
