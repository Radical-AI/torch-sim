"""Elastic tensor calculation with MACE."""

# /// script
# dependencies = [
#     "ase>=3.24",
#     "mace-torch>=0.3.11",
#     "spglib>=2.6",
# ]
# ///

import torch
from ase import units
from ase.atoms import Atoms
from ase.build import bulk
from ase.spacegroup import get_spacegroup
from mace.calculators.foundations_models import mace_mp

from torch_sim.elastic import (
    BravaisType,
    ElasticState,
    get_elastic_tensor,
    get_elementary_deformations,
    get_full_elastic_tensor,
)
from torch_sim.models.mace import MaceModel
from torch_sim.optimizers import frechet_cell_fire
from torch_sim.runners import generate_force_convergence_fn, optimize
from torch_sim.state import initialize_state


def get_bravais_type(  # noqa: C901
    atoms: Atoms, symprec: float = 1e-5
) -> tuple[str, BravaisType, str, int]:
    """Determine Bravais lattice type from ASE Atoms object.

    Args:
        atoms: ASE Atoms object
        symprec: Symmetry precision for spacegroup determination

    Returns:
        Tuple containing:
        - str: Lattice type (primitive, body-centered, etc.)
        - BravaisType: Bravais lattice enum
        - str: Space group symbol
        - int: Space group number

    Notes:
        Lattice types are determined based on space group number:
        - Cubic: 195-230
        - Hexagonal: 168-194
        - Trigonal: 143-167
        - Tetragonal: 75-142
        - Orthorhombic: 16-74
        - Monoclinic: 3-15
        - Triclinic: 1-2

        Within each system, specific centering is determined by
        analyzing cell parameters and angles.
    """
    # Get spacegroup information
    spacegroup = get_spacegroup(atoms, symprec=symprec)
    sg_nr = spacegroup.no
    sg_symbol = spacegroup.symbol

    # Determine lattice type and Bravais lattice
    if 195 <= sg_nr <= 230:  # Cubic
        if sg_nr in [
            195,
            198,
            200,
            201,
            205,
            207,
            208,
            212,
            213,
            215,
            218,
            221,
            222,
            223,
            224,
        ]:
            latt_type = "primitive"
        elif sg_nr in [196, 202, 203, 209, 210, 216, 219, 225, 226, 227, 228]:
            latt_type = "face-centered"
        else:
            latt_type = "body-centered"
        bravais = BravaisType.CUBIC

    elif 168 <= sg_nr <= 194:  # Hexagonal
        latt_type = "primitive"
        bravais = BravaisType.HEXAGONAL

    elif 143 <= sg_nr <= 167:  # Trigonal
        # R-centered
        latt_type = "rhombohedral" if sg_nr <= 148 else "primitive"
        bravais = BravaisType.TRIGONAL

    elif 75 <= sg_nr <= 142:  # Tetragonal
        if sg_nr in [
            75,
            76,
            77,
            78,
            81,
            83,
            84,
            85,
            86,
            89,
            90,
            91,
            92,
            93,
            94,
            95,
            96,
            99,
            100,
            101,
            102,
            103,
            104,
            105,
            106,
            111,
            112,
            113,
            114,
            115,
            116,
            117,
            118,
            123,
            124,
            125,
            126,
            127,
            128,
            129,
            130,
            131,
            132,
            133,
            134,
            135,
            136,
            137,
            138,
        ]:
            latt_type = "primitive"
        else:
            latt_type = "body-centered"
        bravais = BravaisType.TETRAGONAL

    elif 16 <= sg_nr <= 74:  # Orthorhombic
        if sg_nr in [
            16,
            17,
            18,
            19,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            47,
            48,
            49,
            50,
            51,
            52,
            53,
            54,
            55,
            56,
            57,
            58,
            59,
            60,
            61,
            62,
        ]:
            latt_type = "primitive"
        elif sg_nr in [20, 21, 35, 36, 37, 38, 39, 40, 41, 63, 64, 65, 66, 67, 68]:
            latt_type = "face-centered"
        elif sg_nr in [22, 23, 42, 43, 69, 70]:
            latt_type = "body-centered"
        else:
            latt_type = "base-centered"
        bravais = BravaisType.ORTHORHOMBIC

    elif 3 <= sg_nr <= 15:  # Monoclinic
        if sg_nr in [3, 4, 6, 7, 10, 11, 13, 14]:
            latt_type = "primitive"
        else:
            latt_type = "base-centered"
        bravais = BravaisType.MONOCLINIC

    else:  # Triclinic
        latt_type = "primitive"
        bravais = BravaisType.TRICLINIC

    return latt_type, bravais, sg_symbol, sg_nr


device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float64
mace_checkpoint_url = "https://github.com/ACEsuit/mace-mp/releases/download/mace_mpa_0/mace-mpa-0-medium.model"
mace_mp_kwargs = dict(
    model=mace_checkpoint_url,
    enable_cueq=False,
    device=device,
    default_dtype="float64",
)
raw_model = mace_mp(**mace_mp_kwargs, return_raw_model=True)
calculator = mace_mp(**mace_mp_kwargs, return_raw_model=False)
model = MaceModel(
    raw_model, device=device, dtype=dtype, compute_forces=True, compute_stress=True
)

# Copper
N = 2
atoms = bulk("Cu", "fcc", a=3.58, cubic=True)
atoms = atoms.repeat((N, N, N))

state = initialize_state(atoms, device=device, dtype=dtype)
state = optimize(
    state,
    model=model,
    convergence_fn=generate_force_convergence_fn(0.05),
    optimizer=frechet_cell_fire,
)

state = ElasticState(
    position=torch.tensor(atoms.get_positions(), device=device, dtype=dtype),
    cell=torch.tensor(atoms.get_cell().array, device=device, dtype=dtype),
)

latt_type, bravais_type, sg_symbol, sg_nr = get_bravais_type(atoms)
deformations = get_elementary_deformations(
    state, n_deform=6, max_strain=2.0, bravais_type=bravais_type
)

atoms.calc = calculator
ref_pressure = -torch.mean(torch.tensor(atoms.get_stress()[:3], device=device), dim=0)

stresses = torch.zeros((len(deformations), 6), device=device, dtype=torch.float64)
for idx, deformation in enumerate(deformations):
    atoms.cell = deformation.cell.cpu().numpy()
    atoms.positions = deformation.position.cpu().numpy()
    stresses[idx] = torch.tensor(atoms.get_stress(), device=device)

C_ij, B_ij = get_elastic_tensor(state, deformations, stresses, ref_pressure, bravais_type)

full_c_ij = get_full_elastic_tensor(C_ij, bravais_type)

print(full_c_ij / units.GPa)
