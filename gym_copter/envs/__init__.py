'''
Copyright (C) 2019 Simon D. Levy

MIT License
'''

from gym_copter.envs.lander1d import Lander1D  # noqa: F401
from gym_copter.envs.lander2d import Lander2D  # noqa: F401
from gym_copter.envs.lander3d import Lander3D  # noqa: F401
from gym_copter.envs.hover1d import Hover1D  # noqa: F401
from gym_copter.envs.hover2d import Hover2D  # noqa: F401
from gym_copter.envs.hover3d import Hover3D  # noqa: F401

from gym_copter.envs.archive.hover3dV1 import Hover3DV1
from gym_copter.envs.archive.hover3dV2 import Hover3DV2
from gym_copter.envs.archive.hover3dV3 import Hover3DV3
from gym_copter.envs.archive.hover3dV4 import Hover3DV4
from gym_copter.envs.archive.hover3dV5 import Hover3DV5
from gym_copter.envs.archive.hover3dV6 import Hover3DV6
from gym_copter.envs.archive.hover3dV7 import Hover3DV7
from gym_copter.envs.archive.hover3dV8 import Hover3DV8
from gym_copter.envs.archive.hover3dV9 import Hover3DV9
from gym_copter.envs.archive.hover3dV10 import Hover3DV10

from gym_copter.envs.archive.hover3dV12 import Hover3DV12
from gym_copter.envs.archive.hover3dV13 import Hover3DV13

# bug fix 23/02/23

from gym_copter.envs.archive.hover3dV14 import Hover3DV14
from gym_copter.envs.archive.hover3dV15 import Hover3DV15
from gym_copter.envs.archive.hover3dV16 import Hover3DV16
from gym_copter.envs.archive.hover3dV17 import Hover3DV17
from gym_copter.envs.archive.hover3dV18 import Hover3DV18
from gym_copter.envs.archive.hover3dV19 import Hover3DV19
from gym_copter.envs.archive.hover3dV20 import Hover3DV20

# Fault in training
from gym_copter.envs.archive.hover3dV21 import Hover3DV21
from gym_copter.envs.archive.hover3dV22 import Hover3DV22
from gym_copter.envs.archive.hover3dV23 import Hover3DV23
from gym_copter.envs.archive.hover3dV24 import Hover3DV24
from gym_copter.envs.archive.hover3dV25 import Hover3DV25

# Wrong bodyToInertial Function
from gym_copter.envs.archive.hover3dV26 import Hover3DV26
from gym_copter.envs.archive.hover3dV27 import Hover3DV27
from gym_copter.envs.archive.hover3dV28 import Hover3DV28
from gym_copter.envs.archive.hover3dV29 import Hover3DV29
from gym_copter.envs.archive.hover3dV30 import Hover3DV30

# Final Experimentation - All have wind and no out of bounds termination
from gym_copter.envs.models.modelA.modelA_V1 import ModelA_V1
from gym_copter.envs.models.modelA.modelA_V2 import ModelA_V2  # THIS ONE DOES HAVE TERMINATION ON BOUNDS

from gym_copter.envs.models.modelB.modelB_V1 import ModelB_V1

from gym_copter.envs.models.modelB.modelB_V2 import ModelB_V2  # Passive FTC
from gym_copter.envs.models.modelB.modelB_V3 import ModelB_V3  # Passive FTC

print("Initialised ENVS")