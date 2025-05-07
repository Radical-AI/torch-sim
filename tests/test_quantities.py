"""Tests for quantities module functions."""

import pytest
import torch
from numpy.testing import assert_allclose

from torch_sim.quantities import calc_heat_flux


class TestHeatFlux:
    """Test suite for heat flux calculations."""

    @pytest.fixture
    def mock_simple_system(
        self,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """Simple system with known values."""
        return {
            "velocities": torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 2.0, 0.0],
                    [0.0, 0.0, 3.0],
                ],
                device=device,
            ),
            "energies": torch.tensor([1.0, 2.0, 3.0], device=device),
            "stress": torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 3.0, 0.0, 0.0, 0.0],
                ],
                device=device,
            ),
            "masses": torch.ones(3, device=device),
        }

    def test_unbatched_total_flux(
        self, mock_simple_system: dict[str, torch.Tensor]
    ) -> None:
        """Test total heat flux calculation for unbatched case."""
        flux = calc_heat_flux(
            momenta=None,
            masses=mock_simple_system["masses"],
            velocities=mock_simple_system["velocities"],
            energies=mock_simple_system["energies"],
            stress=mock_simple_system["stress"],
            is_virial_only=False,
        )

        # Heat flux parts should cancel out
        expected = torch.zeros(3, device=flux.device)
        assert_allclose(flux.cpu().numpy(), expected.cpu().numpy())

    def test_unbatched_virial_only(
        self, mock_simple_system: dict[str, torch.Tensor]
    ) -> None:
        """Test virial-only heat flux calculation for unbatched case."""
        virial = calc_heat_flux(
            momenta=None,
            masses=mock_simple_system["masses"],
            velocities=mock_simple_system["velocities"],
            energies=mock_simple_system["energies"],
            stress=mock_simple_system["stress"],
            is_virial_only=True,
        )

        expected = -torch.tensor([1.0, 4.0, 9.0], device=virial.device)
        assert_allclose(virial.cpu().numpy(), expected.cpu().numpy())

    def test_batched_calculation(self, device: torch.device) -> None:
        """Test heat flux calculation with batched data."""
        velocities = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 3.0],
            ],
            device=device,
        )
        energies = torch.tensor([1.0, 2.0, 3.0], device=device)
        stress = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 3.0, 0.0, 0.0, 0.0],
            ],
            device=device,
        )
        batch = torch.tensor([0, 0, 1], device=device)

        flux = calc_heat_flux(
            momenta=None,
            masses=torch.ones(3, device=device),
            velocities=velocities,
            energies=energies,
            stress=stress,
            batch=batch,
        )

        # Each batch should cancel heat flux parts
        expected = torch.zeros((2, 3), device=device)
        assert_allclose(flux.cpu().numpy(), expected.cpu().numpy())

    def test_centroid_stress(self, device: torch.device) -> None:
        """Test heat flux with centroid stress formulation."""
        velocities = torch.tensor([[1.0, 1.0, 1.0]], device=device)
        energies = torch.tensor([1.0], device=device)

        # Symmetric cross-terms
        stress = torch.tensor(
            [[1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]], device=device
        )

        flux = calc_heat_flux(
            momenta=None,
            masses=torch.ones(1, device=device),
            velocities=velocities,
            energies=energies,
            stress=stress,
            is_centroid_stress=True,
        )

        # Heatflux should be [-1,-1,-1]
        expected = torch.full((3,), -1.0, device=device)
        assert_allclose(flux.cpu().numpy(), expected.cpu().numpy())

    def test_momenta_input(self, device: torch.device) -> None:
        """Test heat flux calculation using momenta instead."""
        momenta = torch.tensor([[1.0, 0.0, 0.0]], device=device)
        masses = torch.tensor([2.0], device=device)
        energies = torch.tensor([1.0], device=device)
        stress = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]], device=device)

        flux = calc_heat_flux(
            momenta=momenta,
            masses=masses,
            velocities=None,
            energies=energies,
            stress=stress,
        )

        # Heat flux terms should cancel out
        expected = torch.zeros(3, device=device)
        assert_allclose(flux.cpu().numpy(), expected.cpu().numpy())
