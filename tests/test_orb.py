import pytest
import torch
from pymatgen.core import Composition

from torch_sim.optimizers import fire
from torch_sim.workflows.a2c import random_packed_structure


try:
    from orb_models.forcefield import pretrained

    from torch_sim.models.orb import OrbModel
except ImportError:
    pytest.skip("ORB not installed", allow_module_level=True)


def get_orb_model(device: torch.device) -> OrbModel:
    orb_ff = pretrained.orb_v3_conservative_inf_omat(
        device=device,
        precision="float32-high",
    )
    return OrbModel(model=orb_ff, device=device)


def test_orb_relaxation_on_random_structure(
    device: torch.device, dtype: torch.dtype
) -> None:
    """Test relaxation of a random structure using the ORB model and torchsim FIRE optimizer."""
    # Generate a random structure (e.g., 4 Cu atoms in a 5x5x5 cell)
    comp = Composition("Cu4")
    cell = torch.eye(3, device=device, dtype=dtype) * 5.0
    state = random_packed_structure(
        composition=comp,
        cell=cell,
        seed=123,
        diameter=2.5,
        max_iter=5,  # quick overlap minimization
        device=device,
        dtype=dtype,
    )
    model = get_orb_model(device=device)
    # Initial energy/forces
    initial_results = model(state)
    initial_energy = initial_results["energy"].item()
    initial_forces = initial_results["forces"]
    initial_max_force = torch.max(torch.norm(initial_forces, dim=1)).item()
    # Relax using torchsim FIRE optimizer
    fire_init, fire_update = fire(model=model, dt_max=0.1, dt_start=0.05)
    fire_state = fire_init(state)
    for _ in range(10):
        fire_state = fire_update(fire_state)
    # Final energy/forces
    final_results = model(fire_state)
    final_forces = final_results["forces"]
    final_max_force = torch.max(torch.norm(final_forces, dim=1)).item()
    # Assert force decreases
    assert final_max_force < initial_max_force


if __name__ == "__main__":
    test_orb_relaxation_on_random_structure(
        device=torch.device("cpu"), dtype=torch.float32
    )
