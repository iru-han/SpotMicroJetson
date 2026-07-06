"""
Shared state/API for the MJCF-based SpotMicroAI backends (MuJoCo, Genesis).

Both backends load the same urdf/spot_micro.xml, share the same leg/joint
naming and sign conventions, and expose the same Robot API consumed by
pybullet_automatic_gait.py (getPos, getIMU, getAngle, getHeightParam,
resetBody, feetPosition, bodyRotation, bodyPosition, step). This module
holds everything that's identical between them; each backend subclasses
CustomRobotBase and only implements the physics-engine-specific parts.
"""

import os
import math

import numpy as np

from Kinematics.kinematics import Kinematic

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "urdf", "spot_micro.xml")

LEGS = ['front_left', 'front_right', 'rear_left', 'rear_right']
PARTS = ['shoulder', 'leg', 'foot']
DIRS = [[-1, 1, 1], [1, 1, 1], [-1, 1, 1], [1, 1, 1]]

# Matches PyBullet's initial yaw (p.getQuaternionFromEuler([0, 0, 90.0])) so
# "forward" in the gait/keyboard convention points the same way across engines.
PYBULLET_INIT_QUAT_WXYZ = (0.5253219888177297, 0.0, 0.0, 0.8509035245341184)

FALL_ANGLE = math.pi / 3


def quatToEuler(w, x, y, z):
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    pitch = math.asin(max(-1.0, min(1.0, sinp)))

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def hasFallen(roll, pitch):
    return abs(roll) > FALL_ANGLE or abs(pitch) > FALL_ANGLE


class CustomRobotBase:
    """Mix in via `CustomRobotBase._initCommon(self, resetFunc)`; provides the
    gait-facing API that doesn't touch the physics engine directly."""

    def _initCommon(self, resetFunc):
        self.resetFunc = resetFunc

        self.W = 75 + 5 + 40
        self.rot = (0, 0, 0)
        self.pos = (0, 0, 0)
        self.angles = [[0.0, 0.0, 0.0] for _ in range(4)]
        self.Lp = np.array([[120, -100, self.W / 2, 1], [120, -100, -self.W / 2, 1],
                             [-50, -100, self.W / 2, 1], [-50, -100, -self.W / 2, 1]])

        self.kin = Kinematic()

        # No native slider widgets outside PyBullet's GUI; just remember
        # each default so TrottingGait's addUserDebugParameter/
        # readUserDebugParameter calls still work.
        self.height_param = 20.0
        self._debug_params = []

    def getHeightParam(self):
        return self.height_param

    def addUserDebugParameter(self, name, low, high, default):
        self._debug_params.append(default)
        return len(self._debug_params) - 1

    def readUserDebugParameter(self, handle):
        return self._debug_params[handle]

    def setUserDebugParameter(self, handle, value):
        # Lets callers retune a gait constant (e.g. step height) for this
        # engine only, without touching the shared TrottingGait defaults.
        self._debug_params[handle] = value

    def getAngle(self):
        return self.angles

    def feetPosition(self, Lp):
        self.Lp = Lp

    def bodyRotation(self, rot):
        self.rot = rot

    def bodyPosition(self, pos):
        self.pos = pos

    def _targetAngles(self):
        self.angles = self.kin.calcIK(self.Lp, self.rot, self.pos)
        return self.angles
