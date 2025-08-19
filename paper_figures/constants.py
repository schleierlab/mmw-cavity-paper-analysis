import numpy as np
from numpy.typing import ArrayLike, NDArray
from suprtools.rf.couplers import CylindricalProbe
from uncertainties import ufloat

probe = CylindricalProbe(radius=0.287e-3/2, tip_length=2.4e-3)
probe_offset = 1.4e-3


# TODO make this base SI (meters) instead of mm
def stage_position_to_radius(stage_pos: ArrayLike) -> NDArray:
    '''
    Stage position (in mm) to distance from optical axis (in mm).
    '''
    return ((47.96 + 48.34) / 4) + 1.02 - (np.asarray(stage_pos) + 4.8)


# rough estimate for insertion loss in cooldowns Caecilia and Cassia
caecilia_cassia_insertion_loss_db = ufloat(35, 5)

# uncertainty in residue values at base T
# due to distortion due to vibrations
distortion_uncertainty = ufloat(1, 0.3)


flaminia_q_range = (20, 34)
