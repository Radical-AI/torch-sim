"""NVE simulation with MACE."""

# /// script
# dependencies = [
#     "mace-torch>=0.3.11",
# ]
# ///

import os
import time

import torch
from ase.build import bulk
from mace.calculators.foundations_models import mace_mp

from torch_sim.neighbors import vesin_nl_ts
from torch_sim.quantities import kinetic_energy
from torch_sim.state import SimState
from torch_sim.unbatched.models.mace import UnbatchedMaceModel
from torch_sim.unbatched.unbatched_integrators import nve
from torch_sim.units import MetalUnits as Units


# Set device and data type
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32

# Option 1: Load the raw model from the downloaded model
mace_checkpoint_url = "https://github.com/ACEsuit/mace-mp/releases/download/mace_mpa_0/mace-mpa-0-medium.model"
loaded_model = mace_mp(
    model=mace_checkpoint_url,
    return_raw_model=True,
    default_dtype=dtype,
    device=device,
)

# Option 2: Load from local file (comment out Option 1 to use this)
# MODEL_PATH = "../../../checkpoints/MACE/mace-mpa-0-medium.model"
# loaded_model = torch.load(MODEL_PATH, map_location=device)

# Number of steps to run
N_steps = 20 if os.getenv("CI") else 2_000

# Create diamond cubic Silicon
si_dc = bulk("Si", "diamond", a=5.43, cubic=True).repeat((2, 2, 2))

# Prepare input tensors
positions = torch.tensor(si_dc.positions, device=device, dtype=dtype)
cell = torch.tensor(si_dc.cell.array, device=device, dtype=dtype)
atomic_numbers = torch.tensor(si_dc.get_atomic_numbers(), device=device, dtype=torch.int)
masses = torch.tensor(si_dc.get_masses(), device=device, dtype=dtype)

# Initialize the unbatched MACE model
model = UnbatchedMaceModel(
    model=loaded_model,
    device=device,
    neighbor_list_fn=vesin_nl_ts,
    compute_forces=True,
    compute_stress=False,
    dtype=dtype,
    enable_cueq=False,
)

state = SimState(
    positions=positions,
    masses=masses,
    cell=cell,
    pbc=True,
    atomic_numbers=atomic_numbers,
)

# Run initial inference
results = model(state)

# Setup NVE MD simulation parameters
kT = 1000 * Units.temperature  # Initial temperature (K)
dt = 0.002 * Units.time  # Timestep (ps)


# Initialize NVE integrator
nve_init, nve_update = nve(
    model=model,
    dt=dt,
    kT=kT,
)
state = nve_init(state=state, seed=1)

# Run MD simulation
print("\nStarting NVE molecular dynamics simulation...")
start_time = time.perf_counter()
for step in range(N_steps):
    total_energy = state.energy + kinetic_energy(
        masses=state.masses, momenta=state.momenta
    )
    if step % 10 == 0:
        print(f"Step {step}: Total energy: {total_energy.item():.4f} eV")
    state = nve_update(state=state, dt=dt)
end_time = time.perf_counter()

# Report simulation results
print("\nSimulation complete!")
print(f"Time taken: {end_time - start_time:.2f} seconds")
print(f"Average time per step: {(end_time - start_time) / 1000:.4f} seconds")
print(f"Final total energy: {total_energy.item()} eV")
