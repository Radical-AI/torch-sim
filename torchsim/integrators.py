"""Batched MD integrators."""

from collections.abc import Callable
from dataclasses import dataclass

import torch

from torchsim.quantities import temperature
from torchsim.state import BaseState
from torchsim.transforms import pbc_wrap_batched


@dataclass
class MDState(BaseState):
    """State information for MD.

    This class represents the complete state of a molecular system being integrated
    with Langevin dynamics in the NVT (constant particle number, volume, temperature)
    ensemble. The Langevin thermostat adds stochastic noise and friction to maintain
    constant temperature.

    Attributes:
        positions: Particle positions with shape [n_particles, n_dimensions]
        momenta: Particle momenta with shape [n_particles, n_dimensions]
        energy: Energy of the system
        forces: Forces on particles with shape [n_particles, n_dimensions]
        masses: Particle masses with shape [n_particles]
        cell: Simulation cell matrix with shape [n_dimensions, n_dimensions]
        pbc: Whether to use periodic boundary conditions

    Properties:
        velocities: Particle velocities computed as momenta/masses
            Has shape [n_particles, n_dimensions]
    """

    momenta: torch.Tensor
    energy: torch.Tensor
    forces: torch.Tensor

    @property
    def velocities(self) -> torch.Tensor:
        """Calculate velocities from momenta and masses."""
        return self.momenta / self.masses.unsqueeze(-1)


def batched_initialize_momenta(
    positions: torch.Tensor,  # shape: (n_batches, n_atoms_per_batch, 3)
    masses: torch.Tensor,  # shape: (n_batches, n_atoms_per_batch)
    kT: torch.Tensor,  # shape: (n_batches,)
    seeds: torch.Tensor,  # shape: (n_batches,)
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Initialize momenta for batched molecular dynamics.

    Args:
        positions: Atomic positions
        masses: Atomic masses
        kT: Temperature in energy units for each batch
        seeds: Random seeds for each batch
        device: Torch device
        dtype: Torch dtype

    Returns:
        momenta: Random momenta with shape (n_batches, n_atoms_per_batch, 3)
    """
    n_atoms_per_batch = positions.shape[1]

    # Create a generator for each batch using the provided seeds
    generators = [torch.Generator(device=device).manual_seed(int(seed)) for seed in seeds]

    # Generate random momenta for all batches at once
    momenta = torch.stack(
        [
            torch.randn((n_atoms_per_batch, 3), device=device, dtype=dtype, generator=gen)
            for gen in generators
        ]
    )

    # Scale by sqrt(mass * kT)
    mass_factors = torch.sqrt(masses).unsqueeze(-1)  # shape: (n_batches, n_atoms, 1)
    kT_factors = torch.sqrt(kT).view(-1, 1, 1)  # shape: (n_batches, 1, 1)
    momenta *= mass_factors * kT_factors

    # Remove center of mass motion for batches with more than one atom
    # Calculate mean momentum for each batch
    mean_momentum = torch.mean(momenta, dim=1, keepdim=True)  # shape: (n_batches, 1, 3)

    # Create a mask for batches with more than one atom
    multi_atom_mask = torch.tensor(n_atoms_per_batch > 1, device=device, dtype=torch.bool)

    # Subtract mean momentum where needed (broadcasting handles the rest)
    return torch.where(
        multi_atom_mask.view(-1, 1, 1),  # shape: (n_batches, 1, 1)
        momenta - mean_momentum,
        momenta,
    )


def calculate_momenta(
    positions: torch.Tensor,
    masses: torch.Tensor,
    kT: torch.Tensor,
    seed: int | None = None,
) -> torch.Tensor:
    """Calculate momenta from positions and masses."""
    device = positions.device
    dtype = positions.dtype

    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)

    # Generate random momenta from normal distribution
    momenta = torch.randn(
        positions.shape, device=device, dtype=dtype, generator=generator
    ) * torch.sqrt(masses * kT).unsqueeze(-1)

    # Center the momentum if more than one particle
    if positions.shape[0] > 1:
        momenta = momenta - torch.mean(momenta, dim=0, keepdim=True)

    return momenta


def momentum_step(state: MDState, dt: torch.Tensor) -> MDState:
    """Update particle momenta using current forces.

    This function performs the momentum update step of velocity Verlet integration
    by applying forces over the timestep dt.

    Args:
        state: Current system state containing forces and momenta
        dt: Integration timestep

    Returns:
        Updated state with new momenta after force application

    Notes:
        - Implements p(t+dt) = p(t) + F(t)*dt
        - Used as half-steps in velocity Verlet algorithm
        - Preserves time-reversibility when used symmetrically
    """
    new_momenta = state.momenta + state.forces * dt
    state.momenta = new_momenta
    return state


def position_step(state: MDState, dt: torch.Tensor) -> MDState:
    """Update particle positions using current velocities.

    This function performs the position update step of velocity Verlet integration
    by propagating particles according to their velocities over timestep dt.

    Args:
        state: Current system state containing positions and velocities
        dt: Integration timestep

    Returns:
        Updated state with new positions after propagation

    Notes:
        - Implements r(t+dt) = r(t) + v(t)*dt
        - Handles periodic boundary conditions if enabled
        - Used as half-steps in velocity Verlet algorithm
    """
    new_positions = state.positions + state.velocities * dt

    if state.pbc:
        # Split positions and cells by batch
        new_positions = pbc_wrap_batched(new_positions, state.cell, state.batch)

    state.positions = new_positions
    return state


def stochastic_step(
    state: MDState,
    dt: torch.Tensor,
    kT: torch.Tensor,
    gamma: torch.Tensor,
) -> MDState:
    """Apply stochastic noise and friction for Langevin dynamics.

    This function implements the stochastic part of Langevin dynamics by applying
    random noise and friction forces to particle momenta. The noise amplitude is
    chosen to maintain the target temperature kT.

    Args:
        state: Current system state containing positions, momenta, etc.
        dt: Integration timestep
        kT: Target temperature in energy units
        gamma: Friction coefficient controlling noise strength

    Returns:
        Updated state with new momenta after stochastic step

    Notes:
        - Uses Ornstein-Uhlenbeck process for correct thermal sampling
        - Noise amplitude scales with sqrt(mass) for equipartition
        - Preserves detailed balance through fluctuation-dissipation relation
    """
    c1 = torch.exp(-gamma * dt)
    c2 = torch.sqrt(kT * (1 - c1**2))

    # Generate random noise from normal distribution
    noise = torch.randn_like(state.momenta, device=state.device, dtype=state.dtype)
    new_momenta = c1 * state.momenta + c2 * torch.sqrt(state.masses).unsqueeze(-1) * noise
    state.momenta = new_momenta
    return state


def nve(
    state: BaseState,
    model: torch.nn.Module,
    dt: torch.Tensor,
    kT: torch.Tensor,
    seed: int | None = None,
) -> tuple[MDState, Callable[[MDState, torch.Tensor], MDState]]:
    """Initialize and return an NVE (microcanonical) integrator.

    This function sets up integration in the NVE ensemble, where particle number (N),
    volume (V), and total energy (E) are conserved. It returns both an initial state
    and an update function for time evolution.

    Args:
        state: Initial system state containing positions, masses, cell, etc.
        model: Neural network model that computes energies and forces
        dt: Integration timestep
        kT: Temperature for initializing thermal velocities
        seed: Random seed for reproducibility

    Returns:
        tuple:
            - MDState: Initial system state with thermal velocities
            - callable: Update function that evolves system by one timestep

    Notes:
        - Uses velocity Verlet algorithm for time-reversible integration
        - Conserves total energy in the absence of numerical errors
        - Initial velocities sampled from Maxwell-Boltzmann distribution
        - Model must return dict with 'energy' and 'forces' keys
    """

    def nve_update(state: MDState, dt: torch.Tensor = dt, **_) -> MDState:
        """Perform one complete NVE (microcanonical) integration step.

        This function implements the velocity Verlet algorithm for NVE dynamics,
        which provides energy-conserving time evolution. The integration sequence is:
        1. Half momentum update
        2. Full position update
        3. Force update
        4. Half momentum update

        Args:
            state: Current system state containing positions, momenta, forces
            dt: Integration timestep

        Returns:
            Updated state after one complete NVE step

        Notes:
            - Uses velocity Verlet algorithm for time reversible integration
            - Conserves energy in the absence of numerical errors
            - Handles periodic boundary conditions if enabled in state
            - Symplectic integrator preserving phase space volume
        """
        state = momentum_step(state, dt / 2)
        state = position_step(state, dt)

        model_output = model(
            positions=state.positions,
            cell=state.cell,
            batch=state.batch,
            atomic_numbers=state.atomic_numbers,
        )
        state.energy = model_output["energy"]
        state.forces = model_output["forces"]

        return momentum_step(state, dt / 2)

    model_output = model(
        positions=state.positions,
        cell=state.cell,
        batch=state.batch,
        atomic_numbers=state.atomic_numbers,
    )

    momenta = (
        state.momenta
        if getattr(state, "momenta", None) is not None
        else calculate_momenta(state.positions, state.masses, kT, seed)
    )

    # TODO: could consider adding a to_base_state method to BaseState, would
    # clean this up slightly
    initial_state = MDState(
        positions=state.positions,
        masses=state.masses,
        cell=state.cell,
        pbc=state.pbc,
        batch=state.batch,
        atomic_numbers=state.atomic_numbers,
        momenta=momenta,
        energy=model_output["energy"],
        forces=model_output["forces"],
    )

    return initial_state, nve_update


def nvt_langevin(
    state: BaseState,
    model: torch.nn.Module,
    dt: torch.Tensor,
    kT: torch.Tensor,
    gamma: torch.Tensor | None = None,
    seed: int | None = None,
) -> tuple[MDState, Callable[[MDState, torch.Tensor], MDState]]:
    """Initialize and return an NVT (canonical) integrator using Langevin dynamics.

    This function sets up integration in the NVT ensemble, where particle number (N),
    volume (V), and temperature (T) are conserved. It returns both an initial state
    and an update function for time evolution.

    Args:
        state: Initial system state containing positions, masses, cell, etc.
        model: Neural network model that computes energies and forces
        dt: Integration timestep
        kT: Target temperature in energy units
        gamma: Friction coefficient for Langevin thermostat
        seed: Random seed for reproducibility

    Returns:
        tuple:
            - MDState: Initial system state with thermal velocities
            - Callable[[MDState, torch.Tensor], MDState]: Update function for nvt langevin

    Notes:
        - Uses BAOAB splitting scheme for Langevin dynamics
        - Preserves detailed balance for correct NVT sampling
        - Handles periodic boundary conditions if enabled in state
    """
    gamma = gamma or 1 / (100 * dt)

    def langevin_update(
        state: MDState,
        dt: torch.Tensor = dt,
        kT: torch.Tensor = kT,
        gamma: torch.Tensor = gamma,
    ) -> MDState:
        """Perform one complete Langevin dynamics integration step.

        This function implements the BAOAB splitting scheme for Langevin dynamics,
        which provides accurate sampling of the canonical ensemble. The integration
        sequence is:
        1. Half momentum update (B)
        2. Half position update (A)
        3. Full stochastic update (O)
        4. Half position update (A)
        5. Half momentum update (B)

        Args:
            state: Current system state containing positions, momenta, forces
            dt: Integration timestep
            kT: Target temperature (energy units)
            gamma: Friction coefficient for Langevin thermostat

        Returns:
            Updated state after one complete Langevin step
        """
        state = momentum_step(state, dt / 2)
        state = position_step(state, dt / 2)
        state = stochastic_step(state, dt, kT, gamma)
        state = position_step(state, dt / 2)

        model_output = model(
            positions=state.positions,
            cell=state.cell,
            batch=state.batch,
            atomic_numbers=state.atomic_numbers,
        )
        state.energy = model_output["energy"]
        state.forces = model_output["forces"]

        return momentum_step(state, dt / 2)

    model_output = model(
        positions=state.positions,
        cell=state.cell,
        batch=state.batch,
        atomic_numbers=state.atomic_numbers,
    )

    momenta = (
        state.momenta
        if getattr(state, "momenta", None) is not None
        else calculate_momenta(state.positions, state.masses, kT, seed)
    )

    initial_state = MDState(
        positions=state.positions,
        masses=state.masses,
        cell=state.cell,
        pbc=state.pbc,
        batch=state.batch,
        atomic_numbers=state.atomic_numbers,
        momenta=momenta,
        energy=model_output["energy"],
        forces=model_output["forces"],
    )

    return initial_state, langevin_update


def validate_replica_exchange_ensemble(state: MDState) -> None:
    """Validate that the ensemble is appropriate for replica exchange."""
    assert state.n_batches > 1, "Ensemble must contain multiple replicas"

    # replicas must have same atomic numbers
    assert (state.atomic_numbers == state.atomic_numbers[0]).all(), (
        "All replicas must have same atomic numbers"
    )


def replica_exchange_step(
    state: MDState,
    seed: int | None = None,
) -> MDState:
    """Perform one step of replica exchange.

    Args:
        state: The state to perform the replica exchange on
        seed: The seed for the random number generator

    """
    # TODO: need to make temperature batchable

    validate_replica_exchange_ensemble(state)

    generator = torch.Generator(device=state.device)
    if seed is not None:
        generator.manual_seed(seed)

    # select a random pair of replicas
    i = torch.randint(0, state.n_batches - 1, generator=generator)
    j = i + 1

    # swap i and j 50% of the time
    if torch.rand(1, generator=generator) < 0.5:
        i, j = j, i

    # TODO: kinetic or potential energy?
    delta_e = state.energy[i] - state.energy[j]

    temperatures = temperature(state.momenta, state.masses, batch=state.batch)
    delta_beta = 1 / temperatures[i] - 1 / temperatures[j]

    # metropolis criterion
    p_acceptance = torch.exp(delta_beta * delta_e)

    i_mask = state.batch == i
    j_mask = state.batch == j

    # metropolis criterion
    if torch.rand(1, generator=generator) < p_acceptance:
        # swap positions
        state.positions[i_mask], state.positions[j_mask] = (
            state.positions[j_mask],
            state.positions[i_mask],
        )

        # rescale momenta
        state.momenta[i_mask] *= torch.sqrt(temperatures[j] / temperatures[i])
        state.momenta[j_mask] *= torch.sqrt(temperatures[i] / temperatures[j])

    return state
