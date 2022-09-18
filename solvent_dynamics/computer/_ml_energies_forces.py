"""
STATUS: NOT TESTED

"""

import torch

from torch_geometric.data.data import Data
from typing import NamedTuple, Optional


class EnergiesForces(NamedTuple):
    energies: torch.Tensor
    forces: torch.Tensor


def _ml_energies_forces(
        model: torch.nn.Module,
        res_model: Optional[torch.nn.Module],
        structure: Data,
        u_energy_evs: float,
        rms_force_evs: float
    ) -> EnergiesForces:
    y = model(structure)
    if res_model:
        y = res_model(y)
    e = y * rms_force_evs + u_energy_evs
    f = _ml_forces(y, structure.pos) * rms_force_evs

    return EnergiesForces(e.clone(), f.clone())

def _ml_forces(energies: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    pos.requires_grad = True
    nstates = energies.size(dim=0)
    forces = []
    for i in range(nstates):
        f = torch.autograd.grad(-energies[i], pos, create_graph=True, retain_graph=True)[0]
        forces.append(f)
    forces = torch.stack(forces, dim=0)

    return forces
