import pytest
import torch

from tests.conftest import make_model_calculator_consistency_test
from torch_sim.models.interface import validate_model_outputs


try:
    from fairchem.core import OCPCalculator
    from fairchem.core.models.model_registry import model_name_to_local_file

    from torch_sim.models.fairchem import FairChemModel

except ImportError:
    pytest.skip("FairChem not installed", allow_module_level=True)


@pytest.fixture(scope="session")
def model_path(tmp_path_factory: pytest.TempPathFactory) -> str:
    tmp_path = tmp_path_factory.mktemp("fairchem_checkpoints")
    return model_name_to_local_file(
        "EquiformerV2-31M-S2EF-OC20-All+MD", local_cache=str(tmp_path)
    )


@pytest.fixture
def fairchem_model(model_path: str, device: torch.device) -> FairChemModel:
    cpu = device.type == "cpu"
    return FairChemModel(
        model=model_path,
        cpu=cpu,
        seed=0,
    )


@pytest.fixture
def ocp_calculator(model_path: str) -> OCPCalculator:
    return OCPCalculator(checkpoint_path=model_path, cpu=False, seed=0)


test_fairchem_ocp_consistency = make_model_calculator_consistency_test(
    test_name="fairchem_ocp",
    model_fixture_name="fairchem_model",
    calculator_fixture_name="ocp_calculator",
    sim_state_names=[
        "cu_sim_state",
        "ti_sim_state",
        "si_sim_state",
        "sio2_sim_state",
        # "benzene_sim_state",  # TODO: Turn on when #111 fixed
    ],
    rtol=1e-2,
    atol=1e-2,
)


# fairchem batching is broken on CPU, do not replicate this skipping
# logic in other models tests
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Batching does not work properly on CPU for FAIRchem",
)
def test_validate_model_outputs(
    fairchem_model: FairChemModel, device: torch.device
) -> None:
    validate_model_outputs(fairchem_model, device, torch.float32)
