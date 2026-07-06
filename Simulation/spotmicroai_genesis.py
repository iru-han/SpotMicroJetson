"""
Genesis backend for SpotMicroAI.

Mirrors the public API of spotmicroai.Robot / spotmicroai_mujoco.Robot
(getPos, getIMU, getAngle, getHeightParam, resetBody, feetPosition,
bodyRotation, bodyPosition, step) so pybullet_automatic_gait.py can drive
this engine unmodified. Runs on CPU (no NVIDIA GPU required) and reuses the
same MJCF used by the MuJoCo backend, including its tuned PD gains. Shared
state/API with the MuJoCo backend lives in spotmicro_common.py.
"""

import os
import sys

import numpy as np
import genesis as gs

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from spotmicro_common import (
    CustomRobotBase, MODEL_PATH, LEGS, PARTS, DIRS,
    PYBULLET_INIT_QUAT_WXYZ, quatToEuler, hasFallen,
)

_GENESIS_INITIALIZED = False


class Robot(CustomRobotBase):

    def __init__(self, useFixedBase=False, useStairs=True, resetFunc=None):
        self._initCommon(resetFunc)

        global _GENESIS_INITIALIZED
        if not _GENESIS_INITIALIZED:
            gs.init(backend=gs.cpu, logging_level="warning")
            _GENESIS_INITIALIZED = True

        self.scene = gs.Scene(show_viewer=True, sim_options=gs.options.SimOptions(dt=0.01))
        self.scene.add_entity(gs.morphs.Plane())
        self.entity = self.scene.add_entity(gs.morphs.MJCF(file=MODEL_PATH))
        self.scene.build()

        self.dof_idx = []
        for leg in LEGS:
            for part in PARTS:
                joint = self.entity.get_joint(f"{leg}_{part}")
                self.dof_idx.append(joint.dofs_idx_local[0])

        n = len(self.dof_idx)
        self.entity.set_dofs_kp(np.full(n, 160.0), self.dof_idx)
        self.entity.set_dofs_kv(np.full(n, 25.0), self.dof_idx)
        self.entity.set_dofs_force_range(np.full(n, -35.0), np.full(n, 35.0), self.dof_idx)

        self.init_qpos = self.entity.get_qpos().clone()
        self.init_qpos[2] = 0.3
        self.init_qpos[3:7] = gs.tensor(PYBULLET_INIT_QUAT_WXYZ)
        self.entity.set_qpos(self.init_qpos)

    def getPos(self):
        qpos = self.entity.get_qpos()
        return float(qpos[0]), float(qpos[1]), float(qpos[2])

    def getIMU(self):
        qpos = self.entity.get_qpos()
        roll, pitch, yaw = quatToEuler(float(qpos[3]), float(qpos[4]), float(qpos[5]), float(qpos[6]))
        linearVel = tuple(float(v) for v in self.entity.get_vel())
        angularVel = tuple(float(v) for v in self.entity.get_ang())
        return roll, pitch, yaw, linearVel, angularVel

    def resetBody(self):
        self.entity.set_qpos(self.init_qpos)
        self.entity.zero_all_dofs_velocity()
        if self.resetFunc:
            self.resetFunc()

    def step(self):
        angles = self._targetAngles()

        target = np.zeros(len(self.dof_idx))
        for lx, leg in enumerate(LEGS):
            for px, part in enumerate(PARTS):
                target[lx * 3 + px] = angles[lx][px] * DIRS[lx][px]
        self.entity.control_dofs_position(target, self.dof_idx)

        self.scene.step()

        qpos = self.entity.get_qpos()
        roll, pitch, _ = quatToEuler(float(qpos[3]), float(qpos[4]), float(qpos[5]), float(qpos[6]))
        if hasFallen(roll, pitch):
            self.resetBody()
            return False
        return True
