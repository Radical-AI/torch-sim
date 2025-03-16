"""Example script demonstrating batched MACE model optimization with hot-swapping."""

# /// script
# dependencies = [
#     "mace-torch>=0.3.10",
#     "matbench-discovery>=1.3.1",
# ]
# ///

import os
import time

import torch
from mace.calculators.foundations_models import mace_mp
from matbench_discovery.data import DataFiles, ase_atoms_from_zip

from torch_sim.autobatching import HotSwappingAutoBatcher
from torch_sim.io import atoms_to_state
from torch_sim.models.mace import MaceModel
from torch_sim.optimizers import unit_cell_fire
from torch_sim.runners import generate_force_convergence_fn


# --- Setup and Configuration ---
# Device and data type configuration
device = torch.device("cuda")
dtype = torch.float32
print(f"job will run on {device=}")

# --- Model Initialization ---
PERIODIC = True
print("Loading MACE model...")
mace_checkpoint_url = "https://github.com/ACEsuit/mace-mp/releases/download/mace_omat_0/mace-omat-0-medium.model"
mace = mace_mp(model=mace_checkpoint_url, return_raw_model=True)
mace_model = MaceModel(
    model=mace,
    device=device,
    periodic=True,
    dtype=dtype,
    compute_force=True,
)

# Optimization parameters
fmax = 0.05  # Force convergence threshold
n_steps = 10 if os.getenv("CI") else 200_000_000
max_atoms_in_batch = 50 if os.getenv("CI") else 8_000

# --- Data Loading ---
n_structures_to_relax = 2 if os.getenv("CI") else 10
print(f"Loading {n_structures_to_relax:,} structures...")
ase_atoms_list = ase_atoms_from_zip(
    DataFiles.wbm_initial_atoms.path, limit=n_structures_to_relax
)

# --- Optimization Setup ---
# Statistics tracking

# Initialize first batch
fire_init, fire_update = unit_cell_fire(model=mace_model)
fire_states = fire_init(atoms_to_state(ase_atoms_list, device=device, dtype=dtype))

batcher = HotSwappingAutoBatcher(
    model=mace_model,
    memory_scales_with="n_atoms_x_density",
    max_memory_scaler=None,
)
converge_max_force = generate_force_convergence_fn(force_tol=0.05)

start_time = time.perf_counter()

# --- Main Optimization Loop ---
batcher.load_states(fire_states)
all_completed_states, convergence_tensor, state = [], None, None
while (result := batcher.next_batch(state, convergence_tensor))[0] is not None:
    state, completed_states = result
    print(f"Starting new batch of {state.n_batches} states.")

    all_completed_states.extend(completed_states)
    print("Total number of completed states", len(all_completed_states))

    for _step in range(10):
        state = fire_update(state)
    convergence_tensor = converge_max_force(state, last_energy=None)
all_completed_states.extend(result[1])
print("Total number of completed states", len(all_completed_states))

# --- Final Statistics ---
end_time = time.perf_counter()
total_time = end_time - start_time
print(f"Total time taken: {total_time:.2f} seconds")
