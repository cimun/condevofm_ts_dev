"""
surfaces/aus_surface.py

Example evolution script for a gold sulfur (AuS) surface.
"""

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

import random
import shutil

import numpy as np
import torch
from ase.optimize import FIRE
from condevo.es.guidance import KNNNoveltyCondition
from utils import generate_aus_surface

from condevofm.atoms import Optimizer, ThresholdConstrainer
from condevofm.es import CHARLX
from condevofm.es.guidance import AxisCondition
from condevofm.utils import CorrectedApplyLimitsObjective, run_evo

torch.set_default_dtype(torch.float64)

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# Initialize founder structure
founder, threshold, rep = generate_aus_surface(
    n_au_layers=4,
    n_au_atoms=4,
    n_s_atoms=4,
    repeat_x=3,
    repeat_y=3,
    transform=False,  # True,
    vacuum=20.0,
)

# Initialize fixing parameters
constrainer = ThresholdConstrainer(
    fix_axis=[2],
    fix_threshold=7.25,
    freeze_axis=[2],
    freeze_threshold=6.25,
)

# Initialize foundation model
MODEL = "OMAT"  # "MATPES"
if MODEL == "OMAT":
    calc = "../../models/mace-omat-pbe.model"
    e_cutoff = -1000
    model_suffix = "_omat"
elif MODEL == "MATPES":
    calc = "../../models/mace-matpes-r2scan.model"
    e_cutoff = -5000
    model_suffix = "_matpes"

# Initialize relaxing parameters
optimizer = Optimizer(
    founder_atoms=founder,
    constrainer=constrainer,
    calc=calc,
    optimizer=FIRE,
    fmax=0.001,
    steps=10,  # 100,
    logfile=None,
    multiproc=False,
    n_proc=None,
    device="cuda",
    e_cutoff=e_cutoff,
    progress_bar=False,
)

# Initialize condition dependent on the chosen axis
axisCondition = AxisCondition(
    n_atoms=optimizer.free_n_atoms,
    target=1.0,
    kwargs={
        "cond_axis": [2],
        "cond_lower_threshold": 7.0,
        "cond_upper_threshold": 10.0,
    },
)

# Initialize evolutionary algorithm
es = CHARLX
es_config = dict(
    x0=optimizer.free_positions,
    constrainer=constrainer,
    optimizer=optimizer,
    conditions=(axisCondition, KNNNoveltyCondition()),
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
    geometry="axis",
    axis=2,
    lower_threshold=7.0,
    upper_threshold=10.0,
    diff_origin=[0.0, 0.0, 8.5],
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
dst += f"_L-{diff_config['lower_threshold']:.3f}"
dst += f"_U-{diff_config['upper_threshold']:.3f}"
dst += f"_O-{diff_config['diff_origin'][2]:.3f}"
dst += model_suffix

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
