import torch


_TITLE = 'solvated-cyp'
_INITCOND = 0
_NINITCOND = 500
_METHOD = 'wigner'
_FORMAT = 'xyz'
_GL_SEED = 1
_TEMP = 300

# MODULE
class DynamicsComputer():
    def __init__(self) -> None:
        """
        A suite of computations.

        """
        NotImplemented()

    def update_coords(self) -> None:
        NotImplemented()
    def update_velo(self) -> None:
        NotImplemented()
    def compute_kinetic_energy(self) -> None:
        NotImplemented()
    def compute_potential_energy(self) -> None:
        NotImplemented()
    def compute_forces(self) -> None:
        NotImplemented()
    def compute_surface_hopping(self) -> None:
        NotImplemented()

# MODULE
class DynamicsLogger():
    def __init__(self) -> None:
        """
        A suite of logging utils.

        """
        NotImplemented()


class TrajectoryPropagator():
    def __init__(self) -> None:
        # possible args
        # - initial conditions

        NotImplemented()

    def run(self) -> None:
        """
        
        LOOP: for i in steps
            self._propagate()
            self._thermodynamics  # check with Dan, Waruni
            self._surface_hop
            self._check_status
            self._log
        
        print early termination or completion

        """

        NotImplemented()

    def _propagate(self) -> None:
        """
        Propagates a trajectory through a window containing previous-previous,
        previous, current data.

        self._shift_window_nuclear()    
        self._update_kinetics()
        self._update_potential_e()
        self._update_forces()
        self._reset()
        
        """

        NotImplemented()

    def _surface_hop(self) -> None:
        """
        self._shift_window_electronic()
        self.

        """

    def _shift_window_nuclear(self) -> None:
        """
        previous-previous = previous
        previous = current

        """

        NotImplemented()

    def _shift_window_electronic(self) -> None:
        """
        previous-previous = previous
        previous = current

        """

        NotImplemented()


class NAMD():
    def __init__(self) -> None:
        # possible args
        # - global config
        # - initial conditions
        # - loaded ml model  -- or multiple models?

        NotImplemented()

    def run(self) -> None:
        """
        log headers

        LOOP: for i in trajectories
            try:
                traj = init_traj(i)
                traj.run()

            except:
                trajectory terminated
        """

        NotImplemented()
    
    def _propagate(self) -> None:
        # handle initial kinetics
            # poss. excess initial kinetic energy
            # poss. scale kinetic energy

        NotImplemented()


def namd() -> None:
    # load global config

    # load inital conditions

    # load model

    # create molecular dynamics instance `md`

    # md.run()

    NotImplemented()
    

if __name__ == '__main__':
    namd() 
