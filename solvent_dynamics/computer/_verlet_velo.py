"""
STATUS: NOT TESTED

Next velocity propagation
https://en.wikipedia.org/wiki/Verlet_integration

# HERE:
    - replace loop with vmap

"""

import torch


def _verlet_velo(
        state: int,
        coords: torch.Tensor,
        mass: torch.Tensor,
        velo: torch.Tensor,
        forces: torch.Tensor,
        forces_prev: torch.Tensor,
        delta_t: float,
    ) -> torch.Tensor:
    """
    Computes the next atomic coordinate positions using Vertlet Integration.

    Args:
        state (int): Electronic energy state of which this molecular system is
            populating.
        coords (torch.Tensor): Coordinate positions of size (N, 3) where N is
            the number of atoms and 3 corresponds to x, y, and z positions in
            Angstroms.
        mass (torch.Tensor): Atomic mass tensor of size (N) where N is the
            number of atoms.
        velo (torch.Tensor): Atomic velocities with respect to the x, y, and z
            axis given by a tensor of size (N, 3) where N is the number of
            atoms.
        forces (torch.Tensor): Current atomic forces with respect to the x, y,
            and z axis given by a tensor of size (K, N, 3), where K is the
            number of electronic states and N is the number of atoms.
        forces_prev (torch.Tensor): Previous atomic forces with respect to the
            x, y, and z axis given by a tensor of size (K, N, 3), where K is
            the number of electronic states and N is the number of atoms.
        delta_t (float): The change in time from the previous snapshot to
            this snapshot in atomic units of time, au.

    Returns:
        next_velo (torch.Tensor): Atomic velocities with respect to the x, y, and z
            axis given by a tensor of size (N, 3) where N is the number of
            atoms.

    """
    natoms = coords.size(dim=0)
    next_velo = []
    for i in range(natoms):
        delta_velo = 0.5 * (forces_prev[state][i] + forces[state][i]) / mass[i] * delta_t
        next_velo.extend([velo - delta_velo])

    next_velo = torch.stack(next_velo, dim=0)

    return next_velo
