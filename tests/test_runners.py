from typing import Any

import numpy as np
import torch
from ase import Atoms
from pymatgen.core import Structure

from torch_sim.autobatching import ChunkingAutoBatcher, HotSwappingAutoBatcher
from torch_sim.integrators import nve, nvt_langevin
from torch_sim.models.lennard_jones import LennardJonesModel
from torch_sim.optimizers import unit_cell_fire
from torch_sim.quantities import kinetic_energy
from torch_sim.runners import (
    atoms_to_state,
    generate_force_convergence_fn,
    initialize_state,
    integrate,
    optimize,
    state_to_atoms,
    state_to_structures,
    structures_to_state,
)
from torch_sim.state import BaseState, split_state
from torch_sim.trajectory import TorchSimTrajectory, TrajectoryReporter
from torch_sim.units import UnitSystem


def test_integrate_nve(
    ar_base_state: BaseState, lj_calculator: Any, tmp_path: Any
) -> None:
    """Test NVE integration with LJ potential."""
    trajectory_file = tmp_path / "nve.h5md"
    reporter = TrajectoryReporter(
        filenames=trajectory_file,
        state_frequency=1,
        prop_calculators={
            1: {"ke": lambda state: kinetic_energy(state.momenta, state.masses)}
        },
    )

    final_state = integrate(
        system=ar_base_state,
        model=lj_calculator,
        integrator=nve,
        n_steps=10,
        temperature=100.0,  # K
        timestep=0.001,  # ps
        unit_system=UnitSystem.metal,
        trajectory_reporter=reporter,
    )

    assert isinstance(final_state, BaseState)
    assert trajectory_file.exists()

    # Check energy conservation
    with TorchSimTrajectory(trajectory_file) as traj:
        energies = traj.get_array("ke")
        std_energy = np.std(energies)
        assert std_energy / np.mean(energies) < 0.1  # 10% tolerance


def test_integrate_single_nvt(
    ar_base_state: BaseState, lj_calculator: Any, tmp_path: Any
) -> None:
    """Test NVT integration with LJ potential."""
    trajectory_file = tmp_path / "nvt.h5md"
    reporter = TrajectoryReporter(
        filenames=trajectory_file,
        state_frequency=1,
        prop_calculators={
            1: {"ke": lambda state: kinetic_energy(state.momenta, state.masses)}
        },
    )

    final_state = integrate(
        system=ar_base_state,
        model=lj_calculator,
        integrator=nvt_langevin,
        n_steps=10,
        temperature=100.0,  # K
        timestep=0.001,  # ps
        unit_system=UnitSystem.metal,
        trajectory_reporter=reporter,
        gamma=0.1,  # ps^-1
    )

    assert isinstance(final_state, BaseState)
    assert trajectory_file.exists()

    # Check energy fluctuations
    with TorchSimTrajectory(trajectory_file) as traj:
        energies = traj.get_array("ke")
        std_energy = np.std(energies)
        assert std_energy / np.mean(energies) < 0.2  # 20% tolerance for NVT


def test_integrate_double_nvt(
    ar_double_base_state: BaseState, lj_calculator: Any
) -> None:
    """Test NVT integration with LJ potential."""
    final_state = integrate(
        system=ar_double_base_state,
        model=lj_calculator,
        integrator=nvt_langevin,
        n_steps=10,
        temperature=100.0,  # K
        timestep=0.001,  # ps
    )

    assert isinstance(final_state, BaseState)
    assert final_state.n_atoms == 64
    assert not torch.isnan(final_state.energy).any()


def test_integrate_double_nvt_with_reporter(
    ar_double_base_state: BaseState, lj_calculator: Any, tmp_path: Any
) -> None:
    """Test NVT integration with LJ potential."""
    trajectory_files = [tmp_path / "nvt_0.h5md", tmp_path / "nvt_1.h5md"]
    reporter = TrajectoryReporter(
        filenames=trajectory_files,
        state_frequency=1,
        prop_calculators={
            1: {"ke": lambda state: kinetic_energy(state.momenta, state.masses)}
        },
    )

    final_state = integrate(
        system=ar_double_base_state,
        model=lj_calculator,
        integrator=nvt_langevin,
        n_steps=10,
        temperature=100.0,  # K
        timestep=0.001,  # ps
        unit_system=UnitSystem.metal,
        trajectory_reporter=reporter,
        gamma=0.1,  # ps^-1
    )

    assert isinstance(final_state, BaseState)
    assert final_state.n_atoms == 64
    assert all(traj_file.exists() for traj_file in trajectory_files)

    # Check energy fluctuations
    for traj_file in trajectory_files:
        with TorchSimTrajectory(traj_file) as traj:
            energies = traj.get_array("ke")
        std_energy = np.std(energies)
        assert (std_energy / np.mean(energies) < 0.2).all()  # 20% tolerance for NVT
    assert not torch.isnan(final_state.energy).any()


def test_integrate_many_nvt(
    ar_base_state: BaseState,
    fe_fcc_state: BaseState,
    lj_calculator: Any,
    tmp_path: Any,
) -> None:
    """Test NVT integration with LJ potential."""
    triple_state = initialize_state(
        [ar_base_state, ar_base_state, fe_fcc_state],
        lj_calculator.device,
        lj_calculator.dtype,
    )
    trajectory_files = [
        tmp_path / f"nvt_{batch}.h5md" for batch in range(triple_state.n_batches)
    ]
    reporter = TrajectoryReporter(
        filenames=trajectory_files,
        state_frequency=1,
        prop_calculators={
            1: {"ke": lambda state: kinetic_energy(state.momenta, state.masses)}
        },
    )

    final_state = integrate(
        system=triple_state,
        model=lj_calculator,
        integrator=nve,
        n_steps=10,
        temperature=300.0,  # K
        timestep=0.001,  # ps
        trajectory_reporter=reporter,
    )

    assert isinstance(final_state, BaseState)
    assert all(traj_file.exists() for traj_file in trajectory_files)
    assert not torch.isnan(final_state.energy).any()
    assert not torch.isnan(final_state.positions).any()
    assert not torch.isnan(final_state.momenta).any()

    assert torch.allclose(final_state.energy[0], final_state.energy[1], atol=1e-2)
    assert not torch.allclose(final_state.energy[0], final_state.energy[2], atol=1e-2)


def test_integrate_with_autobatcher(
    ar_base_state: BaseState,
    fe_fcc_state: BaseState,
    lj_calculator: Any,
) -> None:
    """Test integration with autobatcher."""
    states = [ar_base_state, fe_fcc_state, ar_base_state]
    triple_state = initialize_state(
        states,
        lj_calculator.device,
        lj_calculator.dtype,
    )
    autobatcher = ChunkingAutoBatcher(
        model=lj_calculator,
        memory_scales_with="n_atoms",
        max_memory_scaler=260,
    )
    final_state = integrate(
        system=triple_state,
        model=lj_calculator,
        integrator=nve,
        n_steps=10,
        temperature=300.0,
        timestep=0.001,
        autobatcher=autobatcher,
    )

    assert isinstance(final_state, BaseState)
    split_final_state = split_state(final_state)
    for init_state, final_state in zip(states, split_final_state, strict=False):
        assert torch.all(final_state.atomic_numbers == init_state.atomic_numbers)
        assert torch.any(final_state.positions != init_state.positions)


def test_integrate_with_autobatcher_and_reporting(
    ar_base_state: BaseState,
    fe_fcc_state: BaseState,
    lj_calculator: Any,
    tmp_path: Any,
) -> None:
    """Test integration with autobatcher."""
    states = [ar_base_state, fe_fcc_state, ar_base_state]
    triple_state = initialize_state(
        states,
        lj_calculator.device,
        lj_calculator.dtype,
    )
    autobatcher = ChunkingAutoBatcher(
        model=lj_calculator,
        memory_scales_with="n_atoms",
        max_memory_scaler=260,
    )
    final_state = integrate(
        system=triple_state,
        model=lj_calculator,
        integrator=nve,
        n_steps=10,
        temperature=300.0,
        timestep=0.001,
        autobatcher=autobatcher,
    )
    trajectory_files = [
        tmp_path / f"nvt_{batch}.h5md" for batch in range(triple_state.n_batches)
    ]
    reporter = TrajectoryReporter(
        filenames=trajectory_files,
        state_frequency=1,
        prop_calculators={1: {"pe": lambda state: state.energy}},
    )
    final_state = integrate(
        system=triple_state,
        model=lj_calculator,
        integrator=nve,
        n_steps=10,
        temperature=300.0,
        timestep=0.001,
        trajectory_reporter=reporter,
        autobatcher=autobatcher,
    )

    assert all(traj_file.exists() for traj_file in trajectory_files)

    assert isinstance(final_state, BaseState)
    split_final_state = split_state(final_state)
    for init_state, final_state in zip(states, split_final_state, strict=False):
        assert torch.all(final_state.atomic_numbers == init_state.atomic_numbers)
        assert torch.any(final_state.positions != init_state.positions)

    for init_state, traj_file in zip(states, trajectory_files, strict=False):
        with TorchSimTrajectory(traj_file) as traj:
            final_state = traj.get_state(-1)
            energies = traj.get_array("pe")
            energy_steps = traj.get_steps("pe")
            assert len(energies) == 10
            assert len(energy_steps) == 10

        assert torch.all(final_state.atomic_numbers == init_state.atomic_numbers)
        assert torch.any(final_state.positions != init_state.positions)


def test_optimize_fire(
    ar_base_state: BaseState, lj_calculator: Any, tmp_path: Any
) -> None:
    """Test FIRE optimization with LJ potential."""
    trajectory_files = [tmp_path / "opt.h5md"]
    reporter = TrajectoryReporter(
        filenames=[tmp_path / "opt.h5md"],
        prop_calculators={1: {"energy": lambda state: state.energy}},
    )
    ar_base_state.positions += torch.randn_like(ar_base_state.positions) * 0.1

    original_state = ar_base_state.clone()

    final_state = optimize(
        system=ar_base_state,
        model=lj_calculator,
        optimizer=unit_cell_fire,
        convergence_fn=generate_force_convergence_fn(force_tol=1e-1),
        unit_system=UnitSystem.metal,
        trajectory_reporter=reporter,
    )

    with TorchSimTrajectory(trajectory_files[0]) as traj:
        energies = traj.get_array("energy")

    # Check force convergence
    assert torch.all(final_state.forces < 3e-1)
    assert energies.shape[0] > 10
    assert energies[0] > energies[-1]
    assert not torch.allclose(original_state.positions, final_state.positions)


def test_default_converged_fn(
    ar_base_state: BaseState, lj_calculator: Any, tmp_path: Any
) -> None:
    """Test default converged function."""
    ar_base_state.positions += torch.randn_like(ar_base_state.positions) * 0.1

    trajectory_files = [tmp_path / "opt.h5md"]
    reporter = TrajectoryReporter(
        filenames=trajectory_files,
        prop_calculators={1: {"energy": lambda state: state.energy}},
    )

    original_state = ar_base_state.clone()

    final_state = optimize(
        system=ar_base_state,
        model=lj_calculator,
        optimizer=unit_cell_fire,
        trajectory_reporter=reporter,
    )

    with TorchSimTrajectory(trajectory_files[0]) as traj:
        energies = traj.get_array("energy")

    assert energies[-3] > energies[-1]
    assert not torch.allclose(original_state.positions, final_state.positions)


def test_batched_optimize_fire(
    ar_double_base_state: BaseState,
    lj_calculator: Any,
    tmp_path: Any,
) -> None:
    """Test batched FIRE optimization with LJ potential."""
    trajectory_files = [
        tmp_path / f"nvt_{idx}.h5md" for idx in range(ar_double_base_state.n_batches)
    ]
    reporter = TrajectoryReporter(
        filenames=trajectory_files,
        state_frequency=1,
        prop_calculators={
            1: {"ke": lambda state: kinetic_energy(state.momenta, state.masses)}
        },
    )

    final_state = optimize(
        system=ar_double_base_state,
        model=lj_calculator,
        optimizer=unit_cell_fire,
        convergence_fn=generate_force_convergence_fn(force_tol=1e-1),
        unit_system=UnitSystem.metal,
        trajectory_reporter=reporter,
    )

    assert torch.all(final_state.forces < 1e-4)


def test_single_structure_to_state(si_structure: Structure, device: torch.device) -> None:
    """Test conversion from pymatgen Structure to state tensors."""
    state = structures_to_state(si_structure, device, torch.float64)

    # Check basic properties
    assert isinstance(state, BaseState)
    assert all(
        t.device.type == device.type for t in [state.positions, state.masses, state.cell]
    )
    assert all(
        t.dtype == torch.float64 for t in [state.positions, state.masses, state.cell]
    )
    assert state.atomic_numbers.dtype == torch.int

    # Check shapes and values
    assert state.positions.shape == (8, 3)
    assert torch.allclose(state.masses, torch.full_like(state.masses, 28.0855))  # Si
    assert torch.all(state.atomic_numbers == 14)  # Si atomic number
    assert torch.allclose(
        state.cell,
        torch.diag(torch.full((3,), 5.43, device=device, dtype=torch.float64)),
    )


def test_optimize_with_autobatcher(
    ar_base_state: BaseState,
    fe_fcc_state: BaseState,
    lj_calculator: Any,
) -> None:
    """Test optimize with autobatcher."""
    states = [ar_base_state, fe_fcc_state, ar_base_state]
    triple_state = initialize_state(
        states,
        lj_calculator.device,
        lj_calculator.dtype,
    )
    autobatcher = HotSwappingAutoBatcher(
        model=lj_calculator,
        memory_scales_with="n_atoms",
        max_memory_scaler=260,
    )
    final_state = optimize(
        system=triple_state,
        model=lj_calculator,
        optimizer=unit_cell_fire,
        convergence_fn=generate_force_convergence_fn(force_tol=1e-1),
        autobatcher=autobatcher,
    )

    assert isinstance(final_state, BaseState)
    split_final_state = split_state(final_state)
    for init_state, final_state in zip(states, split_final_state, strict=False):
        assert torch.all(final_state.atomic_numbers == init_state.atomic_numbers)
        assert torch.any(final_state.positions != init_state.positions)


def test_optimize_with_autobatcher_and_reporting(
    ar_base_state: BaseState,
    fe_fcc_state: BaseState,
    lj_calculator: Any,
    tmp_path: Any,
) -> None:
    """Test optimize with autobatcher and reporting."""
    states = [ar_base_state, fe_fcc_state, ar_base_state]
    triple_state = initialize_state(
        states,
        lj_calculator.device,
        lj_calculator.dtype,
    )
    triple_state.positions += torch.randn_like(triple_state.positions) * 0.1

    autobatcher = HotSwappingAutoBatcher(
        model=lj_calculator,
        memory_scales_with="n_atoms",
        max_memory_scaler=260,
    )

    trajectory_files = [
        tmp_path / f"opt_{batch}.h5md" for batch in range(triple_state.n_batches)
    ]
    reporter = TrajectoryReporter(
        filenames=trajectory_files,
        state_frequency=1,
        prop_calculators={1: {"pe": lambda state: state.energy}},
    )

    final_state = optimize(
        system=triple_state,
        model=lj_calculator,
        optimizer=unit_cell_fire,
        convergence_fn=generate_force_convergence_fn(force_tol=1e-1),
        trajectory_reporter=reporter,
        autobatcher=autobatcher,
    )

    assert all(traj_file.exists() for traj_file in trajectory_files)

    assert isinstance(final_state, BaseState)
    split_final_state = split_state(final_state)
    for init_state, final_state in zip(states, split_final_state, strict=False):
        assert torch.all(final_state.atomic_numbers == init_state.atomic_numbers)
        assert torch.any(final_state.positions != init_state.positions)
        assert torch.all(final_state.forces < 1e-1)

    for init_state, traj_file in zip(states, trajectory_files, strict=False):
        with TorchSimTrajectory(traj_file) as traj:
            traj_state = traj.get_state(-1)
            energies = traj.get_array("pe")
            energy_steps = traj.get_steps("pe")
            assert len(energies) > 0
            assert len(energy_steps) > 0
            # Check that energy decreases during optimization
            assert energies[0] > energies[-1]

        assert torch.all(traj_state.atomic_numbers == init_state.atomic_numbers)
        assert torch.any(traj_state.positions != init_state.positions)


def test_multiple_structures_to_state(
    si_structure: Structure, device: torch.device
) -> None:
    """Test conversion from list of pymatgen Structure to state tensors."""
    state = structures_to_state([si_structure, si_structure], device, torch.float64)

    # Check basic properties
    assert isinstance(state, BaseState)
    assert state.positions.shape == (16, 3)
    assert state.masses.shape == (16,)
    assert state.cell.shape == (2, 3, 3)
    assert state.pbc
    assert state.atomic_numbers.shape == (16,)
    assert state.batch.shape == (16,)
    assert torch.all(
        state.batch == torch.repeat_interleave(torch.tensor([0, 1], device=device), 8)
    )


def test_single_atoms_to_state(si_atoms: Atoms, device: torch.device) -> None:
    """Test conversion from ASE Atoms to state tensors."""
    state = atoms_to_state(si_atoms, device, torch.float64)

    # Check basic properties
    assert isinstance(state, BaseState)
    assert state.positions.shape == (8, 3)
    assert state.masses.shape == (8,)
    assert state.cell.shape == (1, 3, 3)
    assert state.pbc
    assert state.atomic_numbers.shape == (8,)
    assert state.batch.shape == (8,)
    assert torch.all(state.batch == 0)


def test_multiple_atoms_to_state(si_atoms: Atoms, device: torch.device) -> None:
    """Test conversion from ASE Atoms to state tensors."""
    state = atoms_to_state([si_atoms, si_atoms], device, torch.float64)

    # Check basic properties
    assert isinstance(state, BaseState)
    assert state.positions.shape == (16, 3)
    assert state.masses.shape == (16,)
    assert state.cell.shape == (2, 3, 3)
    assert state.pbc
    assert state.atomic_numbers.shape == (16,)
    assert state.batch.shape == (16,)
    assert torch.all(
        state.batch == torch.repeat_interleave(torch.tensor([0, 1], device=device), 8),
    )


def test_state_to_structure(ar_base_state: BaseState) -> None:
    """Test conversion from state tensors to list of pymatgen Structure."""
    structures = state_to_structures(ar_base_state)
    assert len(structures) == 1
    assert isinstance(structures[0], Structure)
    assert len(structures[0]) == 32


def test_state_to_multiple_structures(ar_double_base_state: BaseState) -> None:
    """Test conversion from state tensors to list of pymatgen Structure."""
    structures = state_to_structures(ar_double_base_state)
    assert len(structures) == 2
    assert isinstance(structures[0], Structure)
    assert isinstance(structures[1], Structure)
    assert len(structures[0]) == 32
    assert len(structures[1]) == 32


def test_state_to_atoms(ar_base_state: BaseState) -> None:
    """Test conversion from state tensors to list of ASE Atoms."""
    atoms = state_to_atoms(ar_base_state)
    assert len(atoms) == 1
    assert isinstance(atoms[0], Atoms)
    assert len(atoms[0]) == 32


def test_state_to_multiple_atoms(ar_double_base_state: BaseState) -> None:
    """Test conversion from state tensors to list of ASE Atoms."""
    atoms = state_to_atoms(ar_double_base_state)
    assert len(atoms) == 2
    assert isinstance(atoms[0], Atoms)
    assert isinstance(atoms[1], Atoms)
    assert len(atoms[0]) == 32
    assert len(atoms[1]) == 32


def test_initialize_state_from_structure(
    si_structure: Structure, device: torch.device
) -> None:
    """Test conversion from pymatgen Structure to state tensors."""
    state = initialize_state([si_structure], device, torch.float64)
    assert isinstance(state, BaseState)
    assert state.positions.shape == si_structure.cart_coords.shape
    assert state.cell.shape[1:] == si_structure.lattice.matrix.shape


def test_initialize_state_from_state(
    ar_base_state: BaseState, device: torch.device
) -> None:
    """Test conversion from BaseState to BaseState."""
    state = initialize_state(ar_base_state, device, torch.float64)
    assert isinstance(state, BaseState)
    assert state.positions.shape == ar_base_state.positions.shape
    assert state.masses.shape == ar_base_state.masses.shape
    assert state.cell.shape == ar_base_state.cell.shape


def test_initialize_state_from_atoms(si_atoms: Atoms, device: torch.device) -> None:
    """Test conversion from ASE Atoms to BaseState."""
    state = initialize_state([si_atoms], device, torch.float64)
    assert isinstance(state, BaseState)
    assert state.positions.shape == si_atoms.positions.shape
    assert state.masses.shape == si_atoms.get_masses().shape
    assert state.cell.shape[1:] == si_atoms.cell.array.T.shape


def test_to_atoms(ar_base_state: BaseState) -> None:
    """Test conversion from BaseState to list of ASE Atoms."""
    atoms = state_to_atoms(ar_base_state)
    assert isinstance(atoms[0], Atoms)


def test_to_structures(ar_base_state: BaseState) -> None:
    """Test conversion from BaseState to list of Pymatgen Structure."""
    structures = state_to_structures(ar_base_state)
    assert isinstance(structures[0], Structure)


def test_integrate_with_default_autobatcher(
    ar_base_state: BaseState,
    fe_fcc_state: BaseState,
    lj_calculator: LennardJonesModel,
    monkeypatch: Any,
) -> None:
    """Test integration with autobatcher."""

    def mock_estimate(*args, **kwargs) -> float:  # noqa: ARG001
        return 10000.0

    monkeypatch.setattr("torchsim.autobatching.estimate_max_memory_scaler", mock_estimate)

    states = [ar_base_state, fe_fcc_state, ar_base_state]
    triple_state = initialize_state(
        states,
        lj_calculator.device,
        lj_calculator.dtype,
    )

    final_state = integrate(
        system=triple_state,
        model=lj_calculator,
        integrator=nve,
        n_steps=10,
        temperature=300.0,
        timestep=0.001,
        autobatcher=True,
    )

    assert isinstance(final_state, BaseState)
    split_final_state = split_state(final_state)

    for init_state, final_state in zip(states, split_final_state, strict=False):
        assert torch.all(final_state.atomic_numbers == init_state.atomic_numbers)
        assert torch.any(final_state.positions != init_state.positions)


def test_optimize_with_default_autobatcher(
    ar_base_state: BaseState,
    fe_fcc_state: BaseState,
    lj_calculator: LennardJonesModel,
    monkeypatch: Any,
) -> None:
    """Test optimize with autobatcher."""

    def mock_estimate(*args, **kwargs) -> float:  # noqa: ARG001
        return 10000.0

    monkeypatch.setattr("torchsim.autobatching.estimate_max_memory_scaler", mock_estimate)

    states = [ar_base_state, fe_fcc_state, ar_base_state]
    triple_state = initialize_state(
        states,
        lj_calculator.device,
        lj_calculator.dtype,
    )

    final_state = optimize(
        system=triple_state,
        model=lj_calculator,
        optimizer=unit_cell_fire,
        convergence_fn=generate_force_convergence_fn(force_tol=1e-1),
        autobatcher=True,
    )

    assert isinstance(final_state, BaseState)
    split_final_state = split_state(final_state)
    for init_state, final_state in zip(states, split_final_state, strict=False):
        assert torch.all(final_state.atomic_numbers == init_state.atomic_numbers)
        assert torch.any(final_state.positions != init_state.positions)
