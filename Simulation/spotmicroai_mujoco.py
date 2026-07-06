"""
MuJoCo backend for SpotMicroAI.

Mirrors the public API of spotmicroai.Robot (getPos, getIMU, getAngle,
getHeightParam, resetBody, feetPosition, bodyRotation, bodyPosition, step)
so pybullet_automatic_gait.py can drive either engine unmodified. Shared
state/API with the Genesis backend lives in spotmicro_common.py.
"""

import os
import sys
import time

import mujoco
import mujoco.viewer

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from spotmicro_common import (
    CustomRobotBase, MODEL_PATH, LEGS, PARTS, DIRS,
    PYBULLET_INIT_QUAT_WXYZ, quatToEuler, hasFallen,
)

MAX_SUBSTEPS_PER_CALL = 50


class Robot(CustomRobotBase):

    def __init__(self, useFixedBase=False, useStairs=True, resetFunc=None):
        self._initCommon(resetFunc)

        self.model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        self.data = mujoco.MjData(self.model)

        self.init_qpos = self.data.qpos.copy()
        self.init_qpos[2] = 0.3
        self.init_qpos[3:7] = PYBULLET_INIT_QUAT_WXYZ
        self.data.qpos[:] = self.init_qpos
        mujoco.mj_forward(self.model, self.data)

        self.actuator_ids = {}
        for leg in LEGS:
            for part in PARTS:
                key = f"{leg}_{part}"
                self.actuator_ids[key] = self.model.actuator(key + "_ctrl").id

        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        # Group 1 holds the collision-proxy geoms (shown in orange by the MJCF's
        # collision_material) which otherwise render on top of the visual meshes.
        self.viewer.opt.geomgroup[1] = 0
        self.wall_start = time.time()

    def getPos(self):
        return tuple(self.data.qpos[0:3])

    def getIMU(self):
        w, x, y, z = self.data.qpos[3:7]
        roll, pitch, yaw = quatToEuler(w, x, y, z)
        linearVel = tuple(self.data.qvel[0:3])
        angularVel = tuple(self.data.qvel[3:6])
        return roll, pitch, yaw, linearVel, angularVel

    def resetBody(self):
        self.data.qpos[:] = self.init_qpos
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)
        self.wall_start = time.time() - self.data.time
        if self.resetFunc:
            self.resetFunc()

    def step(self):
        angles = self._targetAngles()

        for lx, leg in enumerate(LEGS):
            for px, part in enumerate(PARTS):
                key = f"{leg}_{part}"
                self.data.ctrl[self.actuator_ids[key]] = angles[lx][px] * DIRS[lx][px]

        target_time = time.time() - self.wall_start
        steps = 0
        while self.data.time < target_time and steps < MAX_SUBSTEPS_PER_CALL:
            mujoco.mj_step(self.model, self.data)
            steps += 1

        if not self.viewer.is_running():
            raise SystemExit("MuJoCo viewer closed")
        self.viewer.sync()

        roll, pitch, _ = quatToEuler(*self.data.qpos[3:7])
        if hasFallen(roll, pitch):
            self.resetBody()
            return False
        return True
