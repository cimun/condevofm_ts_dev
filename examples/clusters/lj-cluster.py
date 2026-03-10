"""
clusters/lj_cluster.py

Example evolution script for a Lennard-Jones (LJ) cluster.
"""

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

import pathlib
import random
import shutil

import numpy as np
import torch
from ase import Atoms
from ase.optimize import FIRE
from condevo.es.guidance import KNNNoveltyCondition
from utils import load_lj_cluster

from condevofm.atoms import IndexConstrainer, Optimizer
from condevofm.es import CHARLX
from condevofm.es.guidance import OriginCondition
from condevofm.utils import CorrectedApplyLimitsObjective, run_evo

torch.set_default_dtype(torch.float64)

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# Initialize founder structure
lj_cluster_rand, max_span, dimensions, symbols, rep = load_lj_cluster(
    n_atoms=13,
    wales_path=pathlib.Path("founders"),
    randomize=True,
    random_seed=SEED,
)
founder = Atoms(symbols, np.reshape(lj_cluster_rand, shape=(-1, 3)))

# Initialize constraining parameters
constrainer = IndexConstrainer(
    fix_indices=[],
    freeze_indices=[],
)

# Initialize relaxing parameters
optimizer = Optimizer(
    founder_atoms=founder,
    constrainer=constrainer,
    calc="LJ",
    optimizer=FIRE,
    fmax=0.001,
    steps=1000,
    logfile=None,
    multiproc=True,
    n_proc=16,
    device="cpu",
    e_cutoff=-500,
    progress_bar=False,
)

# Initialize condition around the origin
condition_obj = OriginCondition(
    n_atoms=optimizer.free_n_atoms,
    target=1.0,
    kwargs={"cond_threshold": max_span},
)

# Initialize evolutionary algorithm
es = CHARLX
es_config = dict(
    x0=optimizer.free_positions,
    constrainer=constrainer,
    optimizer=optimizer,
    conditions=(condition_obj, KNNNoveltyCondition()),
    popsize=16,
    n_gens=5,
    sigma_init=1.5,
    selection_pressure=20.0,
    elite_ratio=0.15,
    crossover_ratio=0.125,
    mutation_rate=0.05,
    diff_batch_size=256,
    diff_max_epoch=1000,
    buffer_size=1000,
    is_genetic_algorithm=True,
    adaptive_selection_pressure=True,
    readaptation=False,
    forget_best=False,
)

# Initialize neural network
nn = "MLP"
nn_config = dict(
    num_hidden=96,
    num_layers=8,
    activation="LeakyReLU",
    num_params=optimizer.dimensions,
    num_conditions=len(es_config["conditions"]),
)

# Initialize diffusion model
diff = "GGDDIM"
diff_config = dict(
    num_steps=5000,
    lamba_range=1.0,
    geometry="radial",
    axis=None,
    lower_threshold=0.0,
    upper_threshold=max_span * 1.5,
    diff_origin=[0.0, 0.0, 0.0],
    overlap_penalty=True,
    train_on_penalty=True,
    progress_bar=False,
)

# Initialize objective that will be maximized
obj = CorrectedApplyLimitsObjective(
    foo_module="condevofm.atoms",
    foo="evaluate_population_with_calc",
    foo_kwargs={
        "obj_params": optimizer.encode_params(),
        "evaluate_atoms": "get_potential_energy",
        "evaluate_kwargs": {},
    },
    maximize=True,
    dim=optimizer.dimensions,
)

# Define destination path for output data
dst = f"_evos/{rep}"
dst += f"_P-{es_config['popsize']}"
dst += f"_G-{es_config['n_gens']}"
dst += f"_F-{optimizer.fmax}"
dst += f"_S-{optimizer.steps}"
dst += f"_U-{diff_config['upper_threshold']:.3f}"

# Remove old folder before new evolution
shutil.rmtree(dst, ignore_errors=True)

# Execute CHARLX evolution
evo = run_evo(
    generations=es_config["n_gens"],
    es=es,
    es_config=es_config,
    nn=nn,
    nn_config=nn_config,
    diff=diff,
    diff_config=diff_config,
    objective=obj,
    dst=dst,
    params={"save_diffusion": False},
)
