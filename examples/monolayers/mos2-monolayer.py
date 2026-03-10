"""
monolayers/mos2_monolayer.py

Example evolution script for a molybdenum disulfide (MoS2) monolayer.
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
from utils import load_mos2_monolayer

from condevofm.atoms import Optimizer, SphereConstrainer
from condevofm.es import CHARLX
from condevofm.es.guidance import AxisCondition
from condevofm.utils import CorrectedApplyLimitsObjective, run_evo

torch.set_default_dtype(torch.float64)

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# Initialize founder structure
founder, dimensions, symbols, rep, center, radius = load_mos2_monolayer(
    structure_path="founders/MoS2_5x5x1_3v-s2.vasp",
)

# Initialize fixing parameters
constrainer = SphereConstrainer(
    fix_center=center,
    fix_radius=radius,
    freeze_center=None,
    freeze_radius=None,
)

# Initialize relaxing parameters
optimizer = Optimizer(
    founder_atoms=founder,
    constrainer=constrainer,
    calc="../../models/mace-mos2-swa.model",
    optimizer=FIRE,
    fmax=0.001,
    steps=10,  # 100,
    logfile=None,
    multiproc=False,
    n_proc=None,
    device="cuda",
    e_cutoff=-510,
    progress_bar=False,
)

# Initialize condition dependent on the chosen axis
axisCondition = AxisCondition(
    n_atoms=optimizer.free_n_atoms,
    target=1.0,
    kwargs={
        "cond_axis": [2],
        "cond_lower_threshold": 10.0,
        "cond_upper_threshold": 14.0,
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
    lower_threshold=10.0,
    upper_threshold=14.0,
    diff_origin=[0.0, 0.0, 12.0],
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
dst += f"_mace"

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
