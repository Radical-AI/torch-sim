from dataclasses import asdict
from pathlib import Path
import typing
from typing import Any

import pytest
import torch
from ase import Atoms
from ase.build import bulk, molecule
from ase.spacegroup import crystal
from phonopy.structure.atoms import PhonopyAtoms
from pymatgen.core import Structure

from torch_sim.io import atoms_to_state, state_to_atoms
from torch_sim.models.lennard_jones import LennardJonesModel
from torch_sim.state import SimState, concatenate_states
from torch_sim.trajectory import TrajectoryReporter
from torch_sim.unbatched.models.lennard_jones import UnbatchedLennardJonesModel
from torch_sim.unbatched.unbatched_integrators import nve


if typing.TYPE_CHECKING:
    from ase.calculators.calculator import Calculator

    from torch_sim.models.interface import ModelInterface


@pytest.fixture
def device() -> torch.device:
    return torch.device("cuda")


@pytest.fixture
def dtype() -> torch.dtype:
    return torch.float64


@pytest.fixture
def ar_atoms() -> Atoms:
    """Create a face-centered cubic (FCC) Argon structure."""
    return bulk("Ar", "fcc", a=5.26, cubic=True)

@pytest.fixture
def fe_atoms() -> Atoms:
    """Create crystalline iron using ASE."""
    return bulk("Fe", "fcc", a=5.26, cubic=True)

@pytest.fixture
def si_atoms() -> Atoms:
    """Create crystalline silicon using ASE."""
    return bulk("Si", "diamond", a=5.43, cubic=True)

@pytest.fixture
def benzene_atoms() -> Atoms:
    """Create benzene using ASE."""
    return molecule("C6H6")


@pytest.fixture
def si_structure() -> Structure:
    """Create crystalline silicon using pymatgen."""
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
    return Structure(lattice, species, coords)


@pytest.fixture
def si_phonopy_atoms() -> Any:
    """Create crystalline silicon using PhonopyAtoms."""
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
    return PhonopyAtoms(
        cell=lattice,
        scaled_positions=coords,
        symbols=species,
        pbc=True,
    )


@pytest.fixture
def si_sim_state(si_atoms: Any, device: torch.device, dtype: torch.dtype) -> Any:
    """Create a basic state from si_structure."""
    return atoms_to_state(si_atoms, device, dtype)


@pytest.fixture
def cu_sim_state(device: torch.device, dtype: torch.dtype) -> SimState:
    """Create crystalline copper using ASE."""
    atoms = bulk("Cu", "fcc", a=3.58, cubic=True)
    return atoms_to_state(atoms, device, dtype)

@pytest.fixture
def mg_sim_state(device: torch.device, dtype: torch.dtype) -> SimState:
    """Create crystalline magnesium using ASE."""
    atoms = bulk("Mg", "hcp", a=3.17, c=5.14)
    return atoms_to_state(atoms, device, dtype)


@pytest.fixture
def sb_sim_state(device: torch.device, dtype: torch.dtype) -> SimState:
    """Create crystalline antimony using ASE."""
    atoms = bulk("Sb", "rhombohedral", a=4.58, alpha=60)
    return atoms_to_state(atoms, device, dtype)


@pytest.fixture
def ti_sim_state(device: torch.device, dtype: torch.dtype) -> SimState:
    """Create crystalline titanium using ASE."""
    atoms = bulk("Ti", "hcp", a=2.94, c=4.64)
    return atoms_to_state(atoms, device, dtype)

@pytest.fixture
def tio2_sim_state(device: torch.device, dtype: torch.dtype) -> SimState:
    """Create crystalline TiO2 using ASE."""
    a, c = 4.60, 2.96
    symbols = ["Ti", "O", "O"]
    basis = [
        (0.5, 0.5, 0),  # Ti
        (0.695679, 0.695679, 0.5),  # O
    ]
    atoms = crystal(
        symbols,
        basis=basis,
        spacegroup=136,  # P4_2/mnm
        cellpar=[a, a, c, 90, 90, 90],
    )
    return atoms_to_state(atoms, device, dtype)


@pytest.fixture
def ga_sim_state(device: torch.device, dtype: torch.dtype) -> SimState:
    """Create crystalline Ga using ASE."""
    a, b, c = 4.43, 7.60, 4.56
    symbols = ["Ga"]
    basis = [
        (0, 0.344304, 0.415401),  # Ga
    ]
    atoms = crystal(
        symbols,
        basis=basis,
        spacegroup=64,  # Cmce
        cellpar=[a, b, c, 90, 90, 90],
    )
    return atoms_to_state(atoms, device, dtype)


@pytest.fixture
def niti_sim_state(device: torch.device, dtype: torch.dtype) -> SimState:
    """Create crystalline NiTi using ASE."""
    a, b, c = 2.89, 3.97, 4.83
    alpha, beta, gamma = 90.00, 105.23, 90.00
    symbols = ["Ni", "Ti"]
    basis = [
        (0.369548, 0.25, 0.217074),  # Ni
        (0.076622, 0.25, 0.671102),  # Ti
    ]
    atoms = crystal(
        symbols,
        basis=basis,
        spacegroup=11,
        cellpar=[a, b, c, alpha, beta, gamma],
    )
    return atoms_to_state(atoms, device, dtype)


@pytest.fixture
def sio2_sim_state(device: torch.device, dtype: torch.dtype) -> SimState:
    """Create an alpha-quartz SiO2 system for testing."""
    atoms = crystal(
        symbols=["O", "Si"],
        basis=[[0.413, 0.2711, 0.2172], [0.4673, 0, 0.3333]],
        spacegroup=152,
        cellpar=[4.9019, 4.9019, 5.3988, 90, 90, 120],
    )
    return atoms_to_state(atoms, device, dtype)

@pytest.fixture
def benzene_sim_state(
    benzene_atoms: Any, device: torch.device, dtype: torch.dtype
) -> Any:
    """Create a basic state from benzene_atoms."""
    return atoms_to_state(benzene_atoms, device, dtype)


@pytest.fixture
def fe_supercell_sim_state(
    fe_atoms: Atoms, device: torch.device, dtype: torch.dtype
) -> Any:
    """Create a face-centered cubic (FCC) iron structure with 4x4x4 supercell."""
    return atoms_to_state(fe_atoms.repeat([4, 4, 4]), device, dtype)


@pytest.fixture
def ar_supercell_sim_state(
    ar_atoms: Atoms, device: torch.device, dtype: torch.dtype
) -> SimState:
    """Create a face-centered cubic (FCC) Argon structure with 2x2x2 supercell."""
    return atoms_to_state(ar_atoms.repeat([2, 2, 2]), device, dtype)


@pytest.fixture
def ar_double_sim_state(ar_supercell_sim_state: SimState) -> SimState:
    """Create a batched state from ar_fcc_sim_state."""
    return concatenate_states(
        [ar_supercell_sim_state, ar_supercell_sim_state],
        device=ar_supercell_sim_state.device,
    )


@pytest.fixture
def si_double_sim_state(si_atoms: Atoms, device: torch.device, dtype: torch.dtype) -> Any:
    """Create a basic state from si_structure."""
    return atoms_to_state([si_atoms, si_atoms], device, dtype)


@pytest.fixture
def mixed_double_sim_state(
    ar_supercell_sim_state: SimState, si_sim_state: SimState
) -> SimState:
    """Create a batched state from ar_fcc_sim_state."""
    return concatenate_states(
        [ar_supercell_sim_state, si_sim_state],
        device=ar_supercell_sim_state.device,
    )


@pytest.fixture
def unbatched_lj_model(
    device: torch.device, dtype: torch.dtype
) -> UnbatchedLennardJonesModel:
    """Create a Lennard-Jones model with reasonable parameters for Ar."""
    return UnbatchedLennardJonesModel(
        use_neighbor_list=True,
        sigma=3.405,
        epsilon=0.0104,
        device=device,
        dtype=dtype,
        compute_forces=True,
        compute_stress=True,
        cutoff=2.5 * 3.405,
    )


@pytest.fixture
def lj_model(device: torch.device, dtype: torch.dtype) -> LennardJonesModel:
    """Create a Lennard-Jones model with reasonable parameters for Ar."""
    return LennardJonesModel(
        use_neighbor_list=True,
        sigma=3.405,
        epsilon=0.0104,
        device=device,
        dtype=dtype,
        compute_forces=True,
        compute_stress=True,
        cutoff=2.5 * 3.405,
    )


@pytest.fixture
def torchsim_trajectory(
    si_sim_state: SimState,
    lj_model: Any,
    tmp_path: Path,
    device: torch.device,
    dtype: torch.dtype,
):
    """Test NVE integration conserves energy."""
    # Initialize integrator
    kT = torch.tensor(300.0, device=device, dtype=dtype)  # Temperature in K
    dt = torch.tensor(0.001, device=device, dtype=dtype)  # Small timestep for stability

    state, update_fn = nve(
        **asdict(si_sim_state),
        model=lj_model,
        dt=dt,
        kT=kT,
    )

    reporter = TrajectoryReporter(tmp_path / "test.hdf5", state_frequency=1)

    # Run several steps
    for step in range(10):
        state = update_fn(state, dt)
        reporter.report(state, step)

    yield reporter.trajectory

    reporter.close()
