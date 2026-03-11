"""
condition.py

Implementation of the BaseCondition, OriginCondition and AxisCondition classes.
"""

from abc import ABC, abstractmethod

import numpy as np
from condevo.es.guidance import Condition
from torch import Tensor, float64, ones, tensor


class BaseCondition(Condition, ABC):
    """
    BaseCondition class docstring
    """

    def __init__(self, n_atoms, target=1.0, kwargs=None):
        Condition.__init__(self)
        self.n_atoms = n_atoms
        self.target = target
        self.kwargs = kwargs

    @abstractmethod
    def condition(self, x):
        pass

    def evaluate(self, charles_instance, x, *args, **kwargs):
        is_tensor = isinstance(x, Tensor)
        device = x.device if is_tensor else None
        if is_tensor:
            x = x.detach().cpu().numpy()

        rc = np.zeros(len(x))

        # Get positions (x,y,z) for every atom of every cluster
        x = np.reshape(x, shape=(-1, 3))

        # Apply specific condition to atom positions
        g0 = self.condition(x)
        # Group the atoms together to form the clusters
        g0 = np.reshape(g0, shape=(-1, self.n_atoms))
        # Get only one bool value for every cluster
        g0 = np.all(g0, axis=1)

        # Set target values
        rc[g0[:]] = 1.0
        rc[~g0[:]] = -1.0

        if is_tensor:
            return tensor(rc, device=device, dtype=float64)
        return rc

    def sample(self, charles_instance, num_samples):
        device = getattr(charles_instance, "device", None)
        return ones(num_samples, dtype=float64, device=device) * self.target

    def to_dict(self):
        return {"target": self.target, **Condition.to_dict(self)}


class OriginCondition(BaseCondition):
    """
    OriginCondition class docstring
    """

    def condition(self, x):
        # Calculate distance from origin for every atom of every cluster
        g0 = np.linalg.norm(x, axis=1) < self.kwargs["cond_threshold"]
        return g0

    def __str__(self):
        return f"OriginCondition(n_atoms={self.n_atoms}, target={self.target}, kwargs={self.kwargs})"

    __repr__ = __str__


class AxisCondition(BaseCondition):
    """
    AxisCondition class docstring
    """

    def condition(self, x):
        # Calculate position on axis for every atom of every cluster
        z = np.array(x[:, self.kwargs["cond_axis"]])
        g0 = (z >= self.kwargs["cond_lower_threshold"]) & (
            z < self.kwargs["cond_upper_threshold"]
        )
        return g0

    def __str__(self):
        return f"AxisCondition(n_atoms={self.n_atoms}, target={self.target}, kwargs={self.kwargs})"

    __repr__ = __str__
