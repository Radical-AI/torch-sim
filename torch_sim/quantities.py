"""Functions for computing physical quantities."""

import torch

from torch_sim.state import SimState
from torch_sim.units import MetalUnits


# @torch.jit.script
def count_dof(tensor: torch.Tensor) -> int:
    """Count the degrees of freedom in the system.

    Args:
        tensor: Tensor to count the degrees of freedom in

    Returns:
        Number of degrees of freedom
    """
    return tensor.numel()


# @torch.jit.script
def calc_kT(  # noqa: N802
    momenta: torch.Tensor,
    masses: torch.Tensor,
    velocities: torch.Tensor | None = None,
    batch: torch.Tensor | None = None,
) -> torch.Tensor:
    """Calculate temperature in energy units from momenta/velocities and masses.

    Args:
        momenta (torch.Tensor): Particle momenta, shape (n_particles, n_dim)
        masses (torch.Tensor): Particle masses, shape (n_particles,)
        velocities (torch.Tensor | None): Particle velocities, shape (n_particles, n_dim)
        batch (torch.Tensor | None): Optional tensor indicating batch membership of
        each particle

    Returns:
        torch.Tensor: Scalar temperature value
    """
    if momenta is not None and velocities is not None:
        raise ValueError("Must pass either momenta or velocities, not both")

    if momenta is None and velocities is None:
        raise ValueError("Must pass either momenta or velocities")

    if momenta is None:
        # If velocity provided, calculate mv^2
        squared_term = (velocities**2) * masses.unsqueeze(-1)
    else:
        # If momentum provided, calculate v^2 = p^2/m^2
        squared_term = (momenta**2) / masses.unsqueeze(-1)

    if batch is None:
        # Count total degrees of freedom
        dof = count_dof(squared_term)
        return torch.sum(squared_term) / dof
    # Sum squared terms for each batch
    flattened_squared = torch.sum(squared_term, dim=-1)

    # Count degrees of freedom per batch
    batch_sizes = torch.bincount(batch)
    dof_per_batch = batch_sizes * squared_term.shape[-1]  # multiply by n_dimensions

    # Calculate temperature per batch
    batch_sums = torch.segment_reduce(
        flattened_squared, reduce="sum", lengths=batch_sizes
    )
    return batch_sums / dof_per_batch


def calc_temperature(
    momenta: torch.Tensor,
    masses: torch.Tensor,
    velocities: torch.Tensor | None = None,
    batch: torch.Tensor | None = None,
    units: object = MetalUnits.temperature,
) -> torch.Tensor:
    """Calculate temperature from momenta/velocities and masses.

    Args:
        momenta (torch.Tensor): Particle momenta, shape (n_particles, n_dim)
        masses (torch.Tensor): Particle masses, shape (n_particles,)
        velocities (torch.Tensor | None): Particle velocities, shape (n_particles, n_dim)
        batch (torch.Tensor | None): Optional tensor indicating batch membership of
        each particle
        units (object): Units to return the temperature in

    Returns:
        torch.Tensor: Temperature value in specified units
    """
    return calc_kT(momenta, masses, velocities, batch) / units


# @torch.jit.script
def calc_kinetic_energy(
    momenta: torch.Tensor,
    masses: torch.Tensor,
    velocities: torch.Tensor | None = None,
    batch: torch.Tensor | None = None,
) -> torch.Tensor:
    """Computes the kinetic energy of a system.

    Args:
        momenta (torch.Tensor): Particle momenta, shape (n_particles, n_dim)
        masses (torch.Tensor): Particle masses, shape (n_particles,)
        velocities (torch.Tensor | None): Particle velocities, shape (n_particles, n_dim)
        batch (torch.Tensor | None): Optional tensor indicating batch membership of
        each particle

    Returns:
        If batch is None: Scalar tensor containing the total kinetic energy
        If batch is provided: Tensor of kinetic energies per batch
    """
    if momenta is not None and velocities is not None:
        raise ValueError("Must pass either momenta or velocities, not both")
    if momenta is None and velocities is None:
        raise ValueError("Must pass either momenta or velocities")

    if momenta is None:
        # Using velocities
        squared_term = (velocities**2) * masses.unsqueeze(-1)
    else:
        # Using momentum
        squared_term = (momenta**2) / masses.unsqueeze(-1)

    if batch is None:
        return 0.5 * torch.sum(squared_term)
    flattened_squared = torch.sum(squared_term, dim=-1)
    return 0.5 * torch.segment_reduce(
        flattened_squared, reduce="sum", lengths=torch.bincount(batch)
    )


def calc_heat_flux(
    momenta: torch.Tensor | None,
    masses: torch.Tensor,
    velocities: torch.Tensor | None,
    energies: torch.Tensor,
    stress: torch.Tensor,
    batch: torch.Tensor | None = None,
    *,  # Force keyword arguments for booleans
    is_centroid_stress: bool = False,
    is_virial_only: bool = False,
) -> torch.Tensor:
    r"""Calculate the heat flux vector.

    Computes the microscopic heat flux, :math:`\mathbf{J}`
    defined as:

    .. math::
        \mathbf{J} = \mathbf{J}^c + \mathbf{J}^v

    where the convective part :math:`\mathbf{J}^c` and virial part
    :math:`\mathbf{J}^v` are:

    .. math::
        \mathbf{J}^c &= \sum_i \epsilon_i \mathbf{v}_i \\
        \mathbf{J}^v &= \sum_i \sum_j \mathbf{S}_{ij} \cdot \mathbf{v}_j

    where :math:`\epsilon_i` is the per-atom energy (p.e. + k.e.),
    :math:`\mathbf{v}_i` is velocity, and :math:`\mathbf{S}_{ij}` is the
    per-atom stress tensor.

    Args:
        momenta: Particle momenta, shape (n_particles, n_dim)
        masses: Particle masses, shape (n_particles,)
        velocities: Particle velocities, shape (n_particles, n_dim)
        energies: Per-atom energies (p.e. + k.e.), shape (n_particles,)
        stress: Per-atom stress tensor components:
            - If is_centroid_stress=False: shape (n_particles, 6) for
              :math:`[\sigma_{xx}, \sigma_{yy}, \sigma_{zz},
              \sigma_{xy}, \sigma_{xz}, \sigma_{yz}]`
            - If is_centroid_stress=True: shape (n_particles, 9) for
              :math:`[\mathbf{r}_{ix}f_{ix}, \mathbf{r}_{iy}f_{iy},
              \mathbf{r}_{iz}f_{iz}, \mathbf{r}_{ix}f_{iy},
              \mathbf{r}_{ix}f_{iz}, \mathbf{r}_{iy}f_{iz},
              \mathbf{r}_{iy}f_{ix}, \mathbf{r}_{iz}f_{ix},
              \mathbf{r}_{iz}f_{iy}]`
        batch: Optional tensor indicating batch membership
        is_centroid_stress: Whether stress uses centroid formulation
        is_virial_only: If True, returns only virial part :math:`\mathbf{J}^v`

    Returns:
        Heat flux vector of shape (3,) or (n_batches, 3)
    """
    if momenta is not None and velocities is not None:
        raise ValueError("Must pass either momenta or velocities, not both")
    if momenta is None and velocities is None:
        raise ValueError("Must pass either momenta or velocities")

    # Deduce velocities
    if velocities is None:
        velocities = momenta / masses.unsqueeze(-1)

    convective_flux = energies.unsqueeze(-1) * velocities

    # Calculate virial flux
    if is_centroid_stress:
        # Centroid formulation: r_i[x,y,z] . f_i[x,y,z]
        virial_x = -(
            stress[:, 0] * velocities[:, 0]  # r_ix.f_ix.v_x
            + stress[:, 3] * velocities[:, 1]  # r_ix.f_iy.v_y
            + stress[:, 4] * velocities[:, 2]  # r_ix.f_iz.v_z
        )
        virial_y = -(
            stress[:, 6] * velocities[:, 0]  # r_iy.f_ix.v_x
            + stress[:, 1] * velocities[:, 1]  # r_iy.f_iy.v_y
            + stress[:, 5] * velocities[:, 2]  # r_iy.f_iz.v_z
        )
        virial_z = -(
            stress[:, 7] * velocities[:, 0]  # r_iz.f_ix.v_x
            + stress[:, 8] * velocities[:, 1]  # r_iz.f_iy.v_y
            + stress[:, 2] * velocities[:, 2]  # r_iz.f_iz.v_z
        )
    else:
        # Standard stress tensor components
        virial_x = -(
            stress[:, 0] * velocities[:, 0]  # s_xx.v_x
            + stress[:, 3] * velocities[:, 1]  # s_xy.v_y
            + stress[:, 4] * velocities[:, 2]  # s_xz.v_z
        )
        virial_y = -(
            stress[:, 3] * velocities[:, 0]  # s_xy.v_x
            + stress[:, 1] * velocities[:, 1]  # s_yy.v_y
            + stress[:, 5] * velocities[:, 2]  # s_yz.v_z
        )
        virial_z = -(
            stress[:, 4] * velocities[:, 0]  # s_xz.v_x
            + stress[:, 5] * velocities[:, 1]  # s_yz.v_y
            + stress[:, 2] * velocities[:, 2]  # s_zz.v_z
        )

    virial_flux = torch.stack([virial_x, virial_y, virial_z], dim=-1)

    if batch is None:
        # All atoms
        virial_sum = torch.sum(virial_flux, dim=0)
        if is_virial_only:
            return virial_sum
        conv_sum = torch.sum(convective_flux, dim=0)
        return conv_sum + virial_sum

    # All atoms in each batch
    n_batches = int(torch.max(batch).item() + 1)
    virial_sum = torch.zeros(
        (n_batches, 3), device=velocities.device, dtype=velocities.dtype
    )
    virial_sum.scatter_add_(0, batch.unsqueeze(-1).expand(-1, 3), virial_flux)

    if is_virial_only:
        return virial_sum

    conv_sum = torch.zeros(
        (n_batches, 3), device=velocities.device, dtype=velocities.dtype
    )
    conv_sum.scatter_add_(0, batch.unsqueeze(-1).expand(-1, 3), convective_flux)
    return conv_sum + virial_sum


def batchwise_max_force(state: SimState) -> torch.Tensor:
    """Compute the maximum force per batch.

    Args:
        state (SimState): State to compute the maximum force per batch for.

    Returns:
        torch.Tensor: Maximum forces per batch
    """
    batch_wise_max_force = torch.zeros(
        state.n_batches, device=state.device, dtype=state.dtype
    )
    max_forces = state.forces.norm(dim=1)
    return batch_wise_max_force.scatter_reduce(
        dim=0, index=state.batch, src=max_forces, reduce="amax"
    )
