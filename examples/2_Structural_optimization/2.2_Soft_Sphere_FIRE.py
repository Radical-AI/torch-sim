"""Structural optimization with soft sphere potential using FIRE optimizer."""

import torch

from torchsim.models.soft_sphere import SoftSphereModel
from torchsim.unbatched_optimizers import fire


# Set up the device and data type
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64

# Set up the random number generator
generator = torch.Generator(device=device)
generator.manual_seed(42)  # For reproducibility

# Create face-centered cubic (FCC) Cu
# 3.61 Å is a typical lattice constant for Cu
a = 3.61  # Lattice constant
PERIODIC = True  # Flag to use periodic boundary conditions

# Generate base FCC unit cell positions (scaled by lattice constant)
base_positions = torch.tensor(
    [
        [0.0, 0.0, 0.0],  # Corner
        [0.0, 0.5, 0.5],  # Face centers
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0],
    ],
    device=device,
    dtype=dtype,
)

# Create 4x4x4 supercell of FCC Cu manually
positions = []
for i in range(4):
    for j in range(4):
        for k in range(4):
            for base_pos in base_positions:
                # Add unit cell position + offset for supercell
                pos = base_pos + torch.tensor([i, j, k], device=device, dtype=dtype)
                positions.append(pos)

# Stack the positions into a tensor
positions = torch.stack(positions)

# Scale by lattice constant
positions = positions * a

# Create the cell tensor
cell = torch.tensor(
    [[4 * a, 0, 0], [0, 4 * a, 0], [0, 0, 4 * a]], device=device, dtype=dtype
)

# Add random perturbation to the positions to start with non-equilibrium structure
positions = positions + 0.2 * torch.randn(
    positions.shape, generator=generator, device=device, dtype=dtype
)

# Create the atomic numbers tensor
atomic_numbers = torch.full((positions.shape[0],), 29, device=device, dtype=torch.int)

# Cu atomic mass in atomic mass units
masses = torch.full((positions.shape[0],), 63.546, device=device, dtype=dtype)

# Initialize the Soft Sphere model
model = SoftSphereModel(
    sigma=2.5,
    device=device,
    dtype=dtype,
    compute_force=True,
    compute_stress=False,
    use_neighbor_list=True,
)


# Run initial simulation and get results
results = model(positions=positions, cell=cell, atomic_numbers=atomic_numbers)

state = {
    "positions": positions,
    "masses": masses,
    "cell": cell,
    "pbc": PERIODIC,
    "atomic_numbers": atomic_numbers,
}

# Initialize FIRE (Fast Inertial Relaxation Engine) optimizer
fire_init, fire_update = fire(
    model=model,
    dt_start=0.005,  # Initial timestep
    dt_max=0.01,  # Maximum timestep
)

state = fire_init(state=state)

# Run optimization for 2000 steps
for step in range(2_000):
    if step % 100 == 0:
        print(f"{step=}: Total energy: {state.energy.item()} eV")
    state = fire_update(state)

# Print max force after optimization
print(f"Initial energy: {results['energy'].item()} eV")
print(f"Final energy: {state.energy.item()} eV")
print(f"Initial max force: {torch.max(torch.abs(results['forces'])).item()} eV/Å")
print(f"Final max force: {torch.max(torch.abs(state.forces)).item()} eV/Å")
