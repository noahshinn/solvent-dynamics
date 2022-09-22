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
    return torch.stack(d_E, dim=0).abs()


class P_NACS(NamedTuple):
    p: torch.Tensor
    nacs: torch.Tensor


def _internal_conversion(
        cur_state: int,
        other_state: int,
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
        cur_state (int): Electronic energy state of which this molecular system is
            populating.
        other_state (int): Electronic energy state of the other electronic
            state in question
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
        p, nacs (torch.Tensor, torch.Tensor): The hopping probability between
            the two given electronic states and the non-adiabatic matrix.

    """

    delta_e = _delta_e(energies, energies_prev, energies_prev_prev, cur_state, other_state)
    e = energies_prev[cur_state] + ke
    avg_e = (energies_prev[other_state] + energies_prev[cur_state]) / 2

    # check hop condition

    low_state = min(other_state, cur_state)
    high_state = max(other_state, cur_state)

    bt = -1 / (coord - coord_prev_prev)
    tf1_1 = forces[low_state] * (coord_prev - coord_prev_prev)
    tf2_1 = forces_prev_prev[high_state] * (coord_prev - coord)
    f_ia_1 = bt * (tf1_1 - tf2_1)

    tf1_2 = forces[high_state] * (coord_prev - coord_prev_prev)
    tf2_2 = forces_prev_prev[low_state] * coord_prev - coord
    f_ia_2 = bt * (tf1_2 - tf2_2)

    f_a = torch.sum((f_ia_2 - f_ia_1) ** 2 / mass) ** 0.5
    f_b = torch.sum(f_ia_1 * f_ia_2 / mass).abs() ** 0.5
    a_2 = (f_a * f_b) / (2 * delta_e ** 3)
    n = math.pi / (4 * a_2 ** 0.5)

    b_2 = (e - avg_e) * f_a / (f_b * delta_e)
    s = torch.sum(f_ia_1 * f_ia_2).sign()
    m = 2 / ((b_2 + torch.abs(b_2 ** 2 + s)) ** 0.5)

    p = torch.exp(-n * m)

    pnacs = (f_ia_2 - f_ia_1) / mass.pow(2)
    nacs = pnacs / torch.sum(pnacs ** 2).pow(0.5)
    
    return P_NACS(p, nacs)


if __name__ == '__main__':
    p, nacs = _internal_conversion(
        cur_state=0,
        other_state=1,
        mass=torch.rand(51),
        coord=torch.rand(51, 3),
        coord_prev=torch.rand(51, 3),
        coord_prev_prev=torch.rand(51, 3),
        velo=torch.rand(51, 3),
        energies=torch.rand(3),
        energies_prev=torch.rand(3),
        energies_prev_prev=torch.rand(3),
        forces=torch.rand(3, 51, 3),
        forces_prev=torch.rand(3, 51, 3),
        forces_prev_prev=torch.rand(3, 51, 3),
        ke=torch.rand(1),
        ic_e_thresh=0.3 
    ) 


















