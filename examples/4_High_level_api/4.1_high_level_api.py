"""Basic Lennard-Jones and MACE relaxation examples with batching, custom convergence
criteria, and logging.
"""

# /// script
# dependencies = [
#     "mace-torch>=0.3.10",
#     "pymatgen>=2025.2.18",
# ]
# ///

import os

import numpy as np
import torch
from ase.build import bulk
from mace.calculators.foundations_models import mace_mp
from pymatgen.core import Structure

from torchsim.integrators import nvt_langevin
from torchsim.models.lennard_jones import LennardJonesModel
from torchsim.models.mace import MaceModel
from torchsim.optimizers import unit_cell_fire
from torchsim.quantities import kinetic_energy
from torchsim.runners import integrate, optimize, state_to_atoms, state_to_structures
from torchsim.trajectory import TorchSimTrajectory, TrajectoryReporter
from torchsim.units import MetalUnits


lj_model = LennardJonesModel(
    sigma=2.0,  # Å, typical for Si-Si interaction
    epsilon=0.1,  # eV, typical for Si-Si interaction
    device=torch.device("cpu"),
    dtype=torch.float64,
)

si_atoms = bulk("Si", "fcc", a=5.43, cubic=True)

final_state = integrate(
    system=si_atoms,
    model=lj_model,
    integrator=nvt_langevin,
    n_steps=40 if os.getenv("CI") else 1000,
    temperature=2000,
    timestep=0.002,
)
final_atoms = state_to_atoms(final_state)


trajectory_file = "lj_trajectory.h5md"
# report potential energy every 10 steps and kinetic energy every 20 steps
prop_calculators = {
    10: {"potential_energy": lambda state: state.energy},
    20: {"kinetic_energy": lambda state: kinetic_energy(state.momenta, state.masses)},
}

reporter = TrajectoryReporter(
    trajectory_file,
    # report state every 10 steps
    state_frequency=10,
    prop_calculators=prop_calculators,
)

final_state = integrate(
    system=si_atoms,
    model=lj_model,
    integrator=nvt_langevin,
    n_steps=40 if os.getenv("CI") else 1000,
    temperature=2000,
    timestep=0.002,
    trajectory_reporter=reporter,
)

# Check energy fluctuations
with TorchSimTrajectory(trajectory_file) as traj:
    kinetic_energies = traj.get_array("kinetic_energy")
    potential_energies = traj.get_array("potential_energy")
    final_energy = potential_energies[-1]

    final_atoms = traj.get_atoms(-1)


### basic mace example

# cuda if available
device = "cuda" if torch.cuda.is_available() else "cpu"


mace = mace_mp(model="small", return_raw_model=True)
mace_model = MaceModel(
    model=mace,
    device=device,
    periodic=True,
    dtype=torch.float64,
    compute_force=True,
)

reporter = TrajectoryReporter(
    trajectory_file,
    # report state every 10 steps
    state_frequency=10,
    prop_calculators=prop_calculators,
)

final_state = integrate(
    system=si_atoms,
    model=mace_model,
    integrator=nvt_langevin,
    n_steps=40 if os.getenv("CI") else 100,
    temperature=2000,
    timestep=0.002,
    trajectory_reporter=reporter,
)
final_atoms = state_to_atoms(final_state)


### basic mace example with batching

fe_atoms = bulk("Fe", "fcc", a=5.26, cubic=True)
fe_atoms_supercell = fe_atoms.repeat([2, 2, 2])
si_atoms_supercell = si_atoms.repeat([2, 2, 2])

final_state = integrate(
    system=[si_atoms, fe_atoms, si_atoms_supercell, fe_atoms_supercell],
    model=mace_model,
    integrator=nvt_langevin,
    n_steps=40 if os.getenv("CI") else 100,
    temperature=2000,
    timestep=0.002,
)
final_atoms = state_to_atoms(final_state)
final_fe_atoms_supercell = final_atoms[3]


### basic mace example with batching and reporting

systems = [si_atoms, fe_atoms, si_atoms_supercell, fe_atoms_supercell]

filenames = [f"batch_traj_{i}.h5md" for i in range(len(systems))]
batch_reporter = TrajectoryReporter(
    filenames,
    state_frequency=100,
    prop_calculators=prop_calculators,
)
final_state = integrate(
    system=systems,
    model=mace_model,
    integrator=nvt_langevin,
    n_steps=40 if os.getenv("CI") else 100,
    temperature=2000,
    timestep=0.002,
    trajectory_reporter=batch_reporter,
)

final_energies_per_atom = []
for filename in filenames:
    with TorchSimTrajectory(filename) as traj:
        final_energy = traj.get_array("potential_energy")[-1]
        final_energies_per_atom.append(final_energy / len(traj.get_atoms(-1)))


systems = [si_atoms, fe_atoms, si_atoms_supercell, fe_atoms_supercell]

final_state = optimize(
    system=systems,
    model=mace_model,
    optimizer=unit_cell_fire,
)


systems = [si_atoms, fe_atoms, si_atoms_supercell, fe_atoms_supercell]

rng = np.random.default_rng()
for system in systems:
    system.positions += rng.random(system.positions.shape) * 0.01

final_state = optimize(
    system=systems,
    model=mace_model,
    optimizer=unit_cell_fire,
    convergence_fn=lambda state, last_energy: torch.all(
        last_energy - state.energy < 1e-6 * MetalUnits.energy
    ),
)


lattice = [[5.43, 0, 0], [0, 5.43, 0], [0, 0, 5.43]]
species = ["Si"] * 8
coords = [
    [0.0, 0.0, 0.0],
    [0.25, 0.25, 0.25],
    [0.0, 0.5, 0.5],
    [0.25, 0.75, 0.75],
    [0.5, 0.0, 0.5],
    [0.75, 0.25, 0.75],
    [0.5, 0.5, 0.0],
    [0.75, 0.75, 0.25],
]
structure = Structure(lattice, species, coords)
final_state = integrate(
    system=structure,
    model=lj_model,
    integrator=nvt_langevin,
    n_steps=40 if os.getenv("CI") else 1000,
    temperature=2000,
    timestep=0.002,
)
final_structure = state_to_structures(final_state)
