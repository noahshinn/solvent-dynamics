"""
STATUS: DEV

Zhu-Nakamura Theory for generalized surface hopping.
J. Chem. Phys. 102, 7448 (1995); https://doi.org/10.1063/1.469057

Formulas given in supporting information below:
http://www.rsc.org/suppdata/c8/cp/c8cp02651c/c8cp02651c1.pdf

"""

import torch

from typing import NamedTuple


class SurfaceHoppingMetrics(NamedTuple):
    """
    a: state-density matrix
    h: energy matrix
    d: non-adiabatic matrix
    velo: velocities
    hop_type: "NO HOP" | "HOP" | "FRUSTRATED"
    state: current electronic energy state

    """
    a: torch.Tensor
    h: torch.Tensor
    d: torch.Tensor
    velo: torch.Tensor
    hop_type: str 
    state: int

def _surface_hopping(
        state: int,
        nstates: int,
        mass: torch.Tensor,
        coord: torch.Tensor,
        coord_prev: torch.Tensor,
        coord_prev_prev: torch.Tensor,
        velo: torch.Tensor,
        energies: torch.Tensor,
        energies_prev: torch.Tensor,
        energies_prev_prev: torch.Tensor,
        forces: torch.Tensor,
        forces_prev: torch.Tensor,
        forces_prev_prev: torch.Tensor,
        ke: torch.Tensor,
        ic_e_thresh: float,
        isc_e_thresh: float
    ) -> SurfaceHoppingMetrics:
    """
    ...

    Args:

    Returns:

    """
    

    return SurfaceHoppingMetrics(a, h, d, v, hop_type, new_state)














