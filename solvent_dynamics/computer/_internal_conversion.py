"""
STATUS: DEV

Internal conversion probability by the Zhu-Nakamura Theory.

Formula found in the source below:
http://www.rsc.org/suppdata/c8/cp/c8cp02651c/c8cp02651c1.pdf

"""

import math
import torch

from typing import NamedTuple


def _delta_e(
        energies: torch.Tensor,
        energies_prev: torch.Tensor,
        energies_prev_prev: torch.Tensor,
        state_cur: int,
        state_other: int
    ) -> torch.Tensor:
    d_E = [
        energies[state_other] - energies[state_cur],
        energies_prev[state_other] - energies_prev[state_cur],
        energies_prev_prev[state_other] - energies_prev_prev[state_cur]
    ]
    return torch.cat(d_E, dim=0).abs()


class P_NACS(NamedTuple):
    p: torch.Tensor
    nacs = torch.Tensor


def _internal_conversion(
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
        ic_e_thresh: float
    ) -> P_NACS:
    """
    Computes a tensor of internal conversion hopping probabilities between
    the current electronic state and every other state.

    Args:
        state (int): Electronic energy state of which this molecular system is
            populating.
        nstates (int): The number of electronic states.
        mass (torch.Tensor): Atomic mass tensor of size (N) where N is the
            number of atoms.
        coords (torch.Tensor): Coordinate positions of size (N, 3) where N is
            the number of atoms and 3 corresponds to x, y, and z positions in
            Angstroms.
            *current coordinates
        coords_prev (torch.Tensor): Coordinate positions of size (N, 3) where
            N is the number of atoms and 3 corresponds to x, y, and z positions
            in Angstroms.
            *prev coordinates
        coords_prev_prev (torch.Tensor): Coordinate positions of size (N, 3)
            where N is the number of atoms and 3 corresponds to x, y, and z
            positions in Angstroms.
            *prev prev coordinates
        velo (torch.Tensor): Atomic velocities with respect to the x, y, and z
            axis given by a tensor of size (N, 3) where N is the number of
            atoms.
        energies (torch.Tensor): A potential energy tensor of size K where K
            is the number of electronic states.
            *current energies
        energies_prev (torch.Tensor): A potential energy tensor of size K where
            K is the number of electronic states.
            *prev step energies
        energies_prev_prev (torch.Tensor): A potential energy tensor of size K
            where K is the number of electronic states.
            *prev prev step energies
        forces (torch.Tensor): An atomic force tensor with respect to the x, y,
            and z axis given by a tensor of size (K, N, 3), where K is the
            number of electronic states and N is the number of atoms.
            *current forces
        forces_prev (torch.Tensor): An atomic force tensor with respect to the
            x, y, and z axis given by a tensor of size (K, N, 3), where K is
            the number of electronic states and N is the number of atoms.
            *prev step forces
        forces_prev_prev (torch.Tensor): An atomic force tensor with respect to
            the x, y, and z axis given by a tensor of size (K, N, 3), where K
            is the number of electronic states and N is the number of atoms.
            *prev prev step forces
        ke (torch.Tensor): A scalar value representing the total kinetic
            energy of the molecular system.
        ic_e_thresh (torch.Tensor): energy gap threshold to compute Zhu-
            Nakamura surface hopping between the same spin states
         
    Returns:
        ic (torch.Tensor): A tensor of probabilities of size (K - 1) where K
            is the number of electronic states.

    """

    e = energies_prev[state] + ke

    ic_c = []
    for i in range(nstates):
        low_state = min(i, state)
        high_state = max(i, state)
        delta_e = _delta_e(energies, energies_prev, energies_prev_prev, state, i)
        #

        bt = -1 / (coord - coord_prev_prev)
        tf1_1 = forces[low_state] * (coord_prev - coord_prev_prev)
        tf2_1 = forces_prev_prev[high_state] * (coord_prev - coord)
        f_ia_1 = bt * (tf1_1 - tf2_1)

        tf1_2 = forces[high_state] * (coord_prev - coord_prev_prev)
        tf2_2 = forces_prev_prev[low_state] * coord_prev - coord
        f_ia_2 = bt * (tf1_2 - tf2_2)

        f_a = torch.sum((f_ia_2 - f_ia_1) ** 2 / mass) ** 0.5 # type: ignore
        f_b = torch.sum(f_ia_1 * f_ia_2 / mass).abs() ** 0.5 # type: ignore
        d_e = ...
        a_2 = (f_a * f_b) / (2 * d_e ** 3) # type: ignore
        n = math.pi / (4 * a_2 ** 0.5)

        b_2 = ...
        s = ...
        m = (2 / (b_2 + torch.abs(b_2 ** 2 + s)) ** 0.5 # type: ignore

        p = torch.exp(-n * m)
        ic_c.append(p)
    
    ic = torch.cat(ic_c, dim=0)

    return ic





















