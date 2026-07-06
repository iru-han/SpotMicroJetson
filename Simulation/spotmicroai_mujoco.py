"""
MuJoCo backend for SpotMicroAI.

Mirrors the public API of spotmicroai.Robot (getPos, getIMU, getAngle,
getHeightParam, resetBody, feetPosition, bodyRotation, bodyPosition, step)
so pybullet_automatic_gait.py can drive either engine unmodified.
"""

import os
import sys
import time
import math

import numpy as np
import mujoco
import mujoco.viewer

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from Kinematics.kinematics import Kinematic

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "urdf", "spot_micro.xml")

LEGS = ['front_left', 'front_right', 'rear_left', 'rear_right']
PARTS = ['shoulder', 'leg', 'foot']
DIRS = [[-1, 1, 1], [1, 1, 1], [-1, 1, 1], [1, 1, 1]]

MAX_SUBSTEPS_PER_CALL = 50


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


class Robot:

    def __init__(self, useFixedBase=False, useStairs=True, resetFunc=None):
        self.resetFunc = resetFunc

        self.W = 75 + 5 + 40
        self.rot = (0, 0, 0)
        self.pos = (0, 0, 0)
        self.angles = [[0.0, 0.0, 0.0] for _ in range(4)]
        self.Lp = np.array([[120, -100, self.W / 2, 1], [120, -100, -self.W / 2, 1],
                             [-50, -100, self.W / 2, 1], [-50, -100, -self.W / 2, 1]])

        self.kin = Kinematic()

        self.model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        self.data = mujoco.MjData(self.model)

        self.init_qpos = self.data.qpos.copy()
        self.init_qpos[2] = 0.3
        # Match PyBullet's initial yaw (p.getQuaternionFromEuler([0, 0, 90.0]))
        # so "forward" in the gait/keyboard convention points the same way in
        # both engines.
        self.init_qpos[3:7] = [0.5253219888177297, 0.0, 0.0, 0.8509035245341184]
        self.data.qpos[:] = self.init_qpos
        mujoco.mj_forward(self.model, self.data)

        self.actuator_ids = {}
        for leg in LEGS:
            for part in PARTS:
                key = f"{leg}_{part}"
                self.actuator_ids[key] = self.model.actuator(key + "_ctrl").id

        # No native slider widget in the MuJoCo passive viewer;
        # keep parity with the PyBullet "height" debug parameter as a constant.
        self.height_param = 20.0
        self._debug_params = []

        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        # Group 1 holds the collision-proxy geoms (shown in orange by the MJCF's
        # collision_material) which otherwise render on top of the visual meshes.
        self.viewer.opt.geomgroup[1] = 0
        self.wall_start = time.time()

    def getHeightParam(self):
        return self.height_param

    def addUserDebugParameter(self, name, low, high, default):
        # No slider widgets in the passive viewer; just remember the default.
        self._debug_params.append(default)
        return len(self._debug_params) - 1

    def readUserDebugParameter(self, handle):
        return self._debug_params[handle]

    def setUserDebugParameter(self, handle, value):
        # Lets callers retune a gait constant (e.g. step height) for this
        # engine only, without touching the shared TrottingGait defaults.
        self._debug_params[handle] = value

    def getPos(self):
        return tuple(self.data.qpos[0:3])

    def getIMU(self):
        w, x, y, z = self.data.qpos[3:7]
        roll, pitch, yaw = quatToEuler(w, x, y, z)
        linearVel = tuple(self.data.qvel[0:3])
        angularVel = tuple(self.data.qvel[3:6])
        return roll, pitch, yaw, linearVel, angularVel

    def getAngle(self):
        return self.angles

    def resetBody(self):
        self.data.qpos[:] = self.init_qpos
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)
        self.wall_start = time.time() - self.data.time
        if self.resetFunc:
            self.resetFunc()

    def feetPosition(self, Lp):
        self.Lp = Lp

    def bodyRotation(self, rot):
        self.rot = rot

    def bodyPosition(self, pos):
        self.pos = pos

    def step(self):
        self.angles = self.kin.calcIK(self.Lp, self.rot, self.pos)

        for lx, leg in enumerate(LEGS):
            for px, part in enumerate(PARTS):
                key = f"{leg}_{part}"
                target = self.angles[lx][px] * DIRS[lx][px]
                self.data.ctrl[self.actuator_ids[key]] = target

        target_time = time.time() - self.wall_start
        steps = 0
        while self.data.time < target_time and steps < MAX_SUBSTEPS_PER_CALL:
            mujoco.mj_step(self.model, self.data)
            steps += 1

        if not self.viewer.is_running():
            raise SystemExit("MuJoCo viewer closed")
        self.viewer.sync()

        roll, pitch, _ = quatToEuler(*self.data.qpos[3:7])
        if abs(roll) > math.pi / 3 or abs(pitch) > math.pi / 3:
            self.resetBody()
            return False
        return True
