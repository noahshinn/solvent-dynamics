"""
STATUS: DEV

>> namd = NAMD(*args, **kwargs)
>> namd.run()

# HERE:
    - prop_duration units
    - delta_t units
    - figure out when and how to log

"""

import torch

from trajectory import TrajectoryPropagator

from typing import Optional


_TITLE = 'solvated-cyp'
_INITCOND = 0
_NINITCOND = 500
_METHOD = 'wigner'
_FORMAT = 'xyz'
_GL_SEED = 1
_TEMP = 300


class NAMD():
    def __init__(
            self,
            model: torch.nn.Module,
            res_model: Optional[torch.nn.Module],
            ntraj: int,
            prop_duration: float,
            delta_t: float
        ) -> None:
        """
        Manages all trajectory propagations.

        Args:
            model (torch.nn.Module): A trained and loaded neural network
                model.
            res_model (torch.nn.Module): A trained and loaded residual
                block placed on top of the outputs of the standard model.
            ntraj (int): The number of trajectories to propagate.
            prop_duration (float): The max duration of a trajectory.
            delta_t (float): The duration between steps in a trajectory.
            
        """
        self._model = model
        self._res_model = res_model
        self._ntraj = ntraj
        self._prop_duration = prop_duration
        self._delta_t = delta_t
        self._nsteps = int(prop_duration / delta_t)

        self._model = ...
        self._res_model = ...

    def run(self) -> None:
        for i in range(self._ntraj):
            traj = ...
            for step in range(self._nsteps):
                traj.propagate() # type: ignore
                if not traj.status(): # type: ignore
                    break
