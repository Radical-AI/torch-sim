"""Demo of the amorphous-to-crystalline (A2C) algorithm for a-Si, ported to torchsim from
jax-md https://github.com/jax-md/jax-md/blob/main/jax_md/a2c/a2c_workflow.py.
"""

# /// script
# dependencies = [
#     "mace-torch>=0.3.12",
#     "moyopy>=0.4.1",
#     "pymatgen>=2025.2.18",
# ]
# ///

import os
import time
from collections import defaultdict

import torch
from mace.calculators.foundations_models import mace_mp
from moyopy import MoyoDataset, SpaceGroupType
from moyopy.interface import MoyoAdapter
from pymatgen.analysis.structure_analyzer import SpacegroupAnalyzer
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Composition, Element, Structure
from tqdm import tqdm

import torch_sim as ts
from torch_sim.models.mace import MaceUrls
from torch_sim.unbatched.models.mace import UnbatchedMaceModel
from torch_sim.unbatched.unbatched_integrators import (
    NVTNoseHooverState,
    nvt_nose_hoover,
    nvt_nose_hoover_invariant,
)
from torch_sim.units import MetalUnits as Units
from torch_sim.workflows import a2c


"""
# Example of how to use random_packed_structure_multi
from torchsim.workflows.a2c_utils import random_packed_structure_multi

comp = Composition("Fe80B20")
cell = torch.tensor(
    [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]], dtype=dtype, device=device
)
structure_multi = random_packed_structure_multi(
    composition=comp,
    cell=cell,
    auto_diameter=True,
    device=device,
    dtype=dtype,
    max_iter=100,
)
"""

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32

raw_model = mace_mp(model=MaceUrls.mace_mpa_medium, return_raw_model=True)

# Define system and model
comp = Composition("Si64")
cell = torch.tensor(
    [[11.1, 0.0, 0.0], [0.0, 11.1, 0.0], [0.0, 0.0, 11.1]], dtype=dtype, device=device
)
atomic_numbers = [Element(el).Z for el in comp.get_el_amt_dict()] * int(comp.num_atoms)

atomic_numbers = torch.tensor(atomic_numbers, device=device, dtype=torch.int)
atomic_masses = [Element(el).atomic_mass for el in comp.get_el_amt_dict()] * int(
    comp.num_atoms
)
species = [Element.from_Z(Z).symbol for Z in atomic_numbers]

model = UnbatchedMaceModel(
    model=raw_model,
    device=device,
    compute_forces=True,
    compute_stress=False,  # We don't need stress for MD
    dtype=dtype,
    enable_cueq=False,
)
# Workflow starts here
structure = a2c.random_packed_structure(
    composition=comp,
    cell=cell,
    auto_diameter=True,
    device=device,
    dtype=dtype,
    max_iter=100,
)

# Relax structure in batches of 4
max_optim_steps = (
    1 if os.getenv("CI") else 100
)  # Number of optimization steps for unit cell relaxation

# MD parameters
equi_steps = 25 if os.getenv("CI") else 2500  # MD steps for melt equilibration
cool_steps = 25 if os.getenv("CI") else 2500  # MD steps for quenching equilibration
final_steps = (
    25 if os.getenv("CI") else 2500
)  # MD steps for amorphous phase equilibration
T_high = 2000  # Melt temperature
T_low = 300  # Quench to this temperature
dt = 0.002 * Units.time  # time step = 2fs
tau = 40 * dt  # oscillation period in Nose-Hoover thermostat
simulation_steps = equi_steps + cool_steps + final_steps

nvt_nose_hoover_init, nvt_nose_hoover_update = nvt_nose_hoover(
    model=model,
    kT=T_high * Units.temperature,
    dt=dt,
)

state_dict = {
    "positions": structure.positions,
    "masses": torch.tensor(atomic_masses, device=device, dtype=dtype),
    "cell": cell,
    "pbc": True,
    "atomic_numbers": atomic_numbers,
}
state = nvt_nose_hoover_init(state_dict)

logger = {
    "T": torch.zeros((simulation_steps, 1), device=device, dtype=dtype),
    "H": torch.zeros((simulation_steps, 1), device=device, dtype=dtype),
}


def step_fn(
    step: int, state: NVTNoseHooverState, logger: dict
) -> tuple[NVTNoseHooverState, dict]:
    """Step function for NVT-MD with Nose-Hoover thermostat."""
    current_temp = a2c.get_target_temperature(step, equi_steps, cool_steps, T_high, T_low)
    logger["T"][step] = (
        ts.quantities.calc_kT(masses=state.masses, momenta=state.momenta)
        / Units.temperature
    )
    logger["H"][step] = nvt_nose_hoover_invariant(
        state, kT=current_temp * Units.temperature
    ).item()
    state = nvt_nose_hoover_update(state, kT=current_temp * Units.temperature)
    return state, logger


# Run NVT-MD with the melt-quench-equilibrate temperature profile
for step in range(simulation_steps):
    state, logger = step_fn(step, state, logger)
    temp, invariant = logger["T"][step].item(), logger["H"][step].item()
    print(f"Step {step}: Temperature: {temp:.4f} K: H: {invariant:.4f} eV")

print(
    f"Amorphous structure is ready: positions\n = "
    f"{state.positions}\ncell\n = {state.cell}\nspecies = {species}"
)

# Convert positions to fractional coordinates
fractional_positions = ts.transforms.get_fractional_coordinates(
    positions=state.positions, cell=state.cell
)

# Get subcells to crystallize
subcells = a2c.get_subcells_to_crystallize(
    fractional_positions=fractional_positions,
    species=species,
    d_frac=0.2 if os.getenv("CI") else 0.1,
    n_min=2,
    n_max=8,
)
print(f"Created {len(subcells)} subcells from a-Si")

# To save time in this example, we (i) keep only the "cubic" subcells where a==b==c, and
# (ii) keep if number of atoms in the subcell is 2, 4 or 8. This reduces the number of
# subcells to relax from approx. 80k to around 160.
subcells = [
    subcell
    for subcell in subcells
    if torch.all((subcell[2] - subcell[1]) == (subcell[2] - subcell[1])[0])
    and subcell[0].shape[0] in (2, 4, 8)
]
print(f"Subcells kept for this example: {len(subcells)}")

candidate_structures = a2c.subcells_to_structures(
    candidates=subcells,
    fractional_positions=fractional_positions,
    cell=state.cell,
    species=species,
)

pymatgen_struct_list = [
    Structure(
        lattice=struct[1].detach().cpu().numpy(),
        species=struct[2],
        coords=struct[0].detach().cpu().numpy(),
        coords_are_cartesian=False,
    )
    for struct in candidate_structures
]
# Make sure to compute stress
model._compute_stress = True  # noqa: SLF001

start_time = time.perf_counter()
pymatgen_relaxed_struct_list = []
for struct in tqdm(pymatgen_struct_list):
    state = ts.io.structures_to_state(struct, device=device, dtype=dtype)
    final_state, logger, final_energy, final_pressure = (
        a2c.get_unit_cell_relaxed_structure(
            state=state,
            model=model,
            max_iter=max_optim_steps,
        )
    )

    # Add batch dimension to cell
    final_state.cell = final_state.cell.reshape(1, 3, 3)
    final_struct_list = ts.io.state_to_structures(final_state)

    # NOTE: Possible OOM, so we don't store the logger
    # relaxed_structures.append((pymatgen_struct, logger, final_energy, final_pressure))
    pymatgen_relaxed_struct_list.extend(
        [
            (final_struct, final_energy, final_pressure)
            for final_struct in final_struct_list
        ]
    )

lowest_e_struct = sorted(
    pymatgen_relaxed_struct_list, key=lambda x: x[-2] / x[0].num_sites
)[0]
spg = SpacegroupAnalyzer(lowest_e_struct[0])
print("Space group of predicted crystallization product:", spg.get_space_group_symbol())

spg_counter = defaultdict(int)
for struct in pymatgen_relaxed_struct_list:
    sym_data = MoyoDataset(MoyoAdapter.from_py_obj(struct[0]))
    sp = (sym_data.number, SpaceGroupType(sym_data.number).arithmetic_symbol)
spg_counter[sp] += 1

print("All space groups encountered:", dict(spg_counter))
si_diamond = Structure(
    lattice=[
        [0.0, 2.732954, 2.732954],
        [2.732954, 0.0, 2.732954],
        [2.732954, 2.732954, 0.0],
    ],
    species=["Si", "Si"],
    coords=[[0.5, 0.5, 0.5], [0.75, 0.75, 0.75]],
    coords_are_cartesian=False,
)
struct_match = StructureMatcher().fit(lowest_e_struct[0], si_diamond)
print("Prediction matches diamond-cubic Si?", struct_match)

end_time = time.perf_counter()
print(f"Total time taken to run relaxation: {end_time - start_time:.2f} seconds")
