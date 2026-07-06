"""
Gymnasium environment for SpotMicroAI velocity-tracking locomotion in MuJoCo.

Single flat policy: the agent outputs a small delta around a known-good
standing pose (fed to the MJCF's tuned position actuators) and is rewarded
for tracking a commanded forward velocity while behaving like a real robot
should (low torque, low jerk, upright, joints away from their limits).

Reward/observation/action design follows the battle-tested conventions from
Unitree's legged_gym (github.com/unitreerobotics/unitree_rl_gym, based on
the ETH Zurich legged_gym/rsl_rl lineage) — default-pose-relative action and
observations, exp(-error^2/sigma) velocity tracking, dof_acc/dof_pos_limits
penalties, only_positive_rewards clipping — adapted from Isaac Gym to MuJoCo.
The base "track a single forward velocity" objective and its motivating cost
categories (torque, foot slip, orientation, smoothness) come from the
hierarchical-gait paper (Kim et al. 2021, arXiv:2112.04741); the hierarchical
CPG controller in that paper is a possible follow-up once this works.
"""

import os
import sys
import math

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from spotmicro_common import MODEL_PATH, LEGS, PARTS, PYBULLET_INIT_QUAT_WXYZ, quatToEuler, hasFallen

CONTROL_DT = 0.02  # 50 Hz policy rate
MAX_EPISODE_SECONDS = 15.0
MIN_HEIGHT = 0.10  # below this the robot has collapsed
TARGET_HEIGHT = 0.19  # where this stance naturally settles (see spotmicro_common)

# action = DEFAULT_JOINT_ANGLES + ACTION_SCALE * policy_output, like legged_gym's
# `target angle = actionScale * action + defaultAngle`. Keeps the policy's
# starting point (action=0) at a pose we already know stands up, instead of
# forcing it to discover a stable stance from an arbitrary "mid-range" pose.
ACTION_SCALE = 0.25

# Standing joint angles for the same Lp/rot/pos stance used throughout the
# project (spotmicroai*.py's default Lp, body offset (50, 60, 0)); computed
# once via Kinematic().calcIK. Order matches LEGS x PARTS.
DEFAULT_JOINT_ANGLES = np.array([
    0.1749, -0.8390, 1.6780,   # front_left: shoulder, leg, foot
    -0.1749, -0.8390, 1.6780,  # front_right
    0.1749, -1.0372, 1.6328,   # rear_left
    -0.1749, -1.0372, 1.6328,  # rear_right
])

SOFT_DOF_POS_LIMIT = 0.9  # penalize the outer 10% of each joint's range
ONLY_POSITIVE_REWARDS = True  # clip the summed reward at 0 (legged_gym trick:
# stops the agent from ever preferring "end the episode early" over "keep trying")

# Reward term weights — starting points, not tuned to either reference's units.
W_TRACKING = 1.0
TRACKING_SIGMA = 0.25
W_LIN_VEL_Z = 1.0
W_ANG_VEL_XY = 0.05
W_ANG_VEL_Z = 0.05
W_ORIENTATION = 0.3
W_TORQUE = 1e-4
W_DOF_VEL = 1e-3
W_DOF_ACC = 1e-6
W_ACTION_RATE = 0.02
W_DOF_POS_LIMITS = 1.0
W_BASE_HEIGHT = 1.0
W_ALIVE = 0.5


class SpotMicroMuJoCoEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": int(1 / CONTROL_DT)}

    def __init__(self, render_mode=None, command_vx_range=(0.1, 1.0)):
        super().__init__()
        self.render_mode = render_mode
        self.command_vx_range = command_vx_range

        self.model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        self.data = mujoco.MjData(self.model)
        self.substeps = max(1, round(CONTROL_DT / self.model.opt.timestep))
        self.max_steps = int(MAX_EPISODE_SECONDS / CONTROL_DT)

        self.joint_names = [f"{leg}_{part}" for leg in LEGS for part in PARTS]
        self.actuator_ids = np.array([self.model.actuator(f"{j}_ctrl").id for j in self.joint_names])
        self.joint_qpos_adr = np.array([self.model.jnt_qposadr[self.model.joint(j).id] for j in self.joint_names])
        self.joint_dof_adr = np.array([self.model.jnt_dofadr[self.model.joint(j).id] for j in self.joint_names])
        self.joint_low = np.array([self.model.joint(j).range[0] for j in self.joint_names])
        self.joint_high = np.array([self.model.joint(j).range[1] for j in self.joint_names])

        joint_mid = (self.joint_low + self.joint_high) / 2.0
        joint_span = (self.joint_high - self.joint_low) * SOFT_DOF_POS_LIMIT
        self.soft_low = joint_mid - joint_span / 2.0
        self.soft_high = joint_mid + joint_span / 2.0

        self.init_qpos = self.data.qpos.copy()
        self.init_qpos[2] = 0.3
        self.init_qpos[3:7] = PYBULLET_INIT_QUAT_WXYZ
        self.init_qpos[self.joint_qpos_adr] = DEFAULT_JOINT_ANGLES

        n = len(self.joint_names)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(n,), dtype=np.float32)
        obs_dim = 3 + 3 + 3 + n + n + n + 1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self.viewer = None
        self.prev_action = np.zeros(n, dtype=np.float32)
        self.last_joint_vel = np.zeros(n, dtype=np.float32)
        self.command_vx = 0.5
        self.step_count = 0

        self.np_random_ = np.random.default_rng()

    def _actionToTarget(self, action):
        action = np.clip(action, -1.0, 1.0)
        target = DEFAULT_JOINT_ANGLES + ACTION_SCALE * action
        return np.clip(target, self.joint_low, self.joint_high)

    def _bodyFrameVectors(self):
        w, x, y, z = self.data.qpos[3:7]
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, np.array([w, x, y, z]))
        R = R.reshape(3, 3)
        linvel_world = self.data.qvel[0:3]
        angvel_world = self.data.qvel[3:6]
        linvel_body = R.T @ linvel_world
        angvel_body = R.T @ angvel_world
        gravity_body = R.T @ np.array([0.0, 0.0, -1.0])
        return linvel_body, angvel_body, gravity_body

    def _getObs(self):
        linvel_body, angvel_body, gravity_body = self._bodyFrameVectors()
        qpos_rel = self.data.qpos[self.joint_qpos_adr] - DEFAULT_JOINT_ANGLES
        qvel = self.data.qvel[self.joint_dof_adr]
        return np.concatenate([
            gravity_body, linvel_body, angvel_body,
            qpos_rel, qvel, self.prev_action, [self.command_vx],
        ]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random_ = np.random.default_rng(seed)

        self.data.qpos[:] = self.init_qpos
        self.data.qvel[:] = 0
        # Small randomization so the policy doesn't overfit one exact pose.
        self.data.qpos[self.joint_qpos_adr] += self.np_random_.uniform(-0.05, 0.05, size=len(self.joint_names))
        mujoco.mj_forward(self.model, self.data)

        self.prev_action[:] = 0.0
        self.last_joint_vel[:] = 0.0
        self.command_vx = float(self.np_random_.uniform(*self.command_vx_range))
        self.step_count = 0

        return self._getObs(), {}

    def step(self, action):
        target = self._actionToTarget(action)
        self.data.ctrl[self.actuator_ids] = target

        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)

        linvel_body, angvel_body, _ = self._bodyFrameVectors()
        roll, pitch, _ = quatToEuler(*self.data.qpos[3:7])
        torque = self.data.actuator_force[self.actuator_ids]
        joint_pos = self.data.qpos[self.joint_qpos_adr]
        joint_vel = self.data.qvel[self.joint_dof_adr]
        height = self.data.qpos[2]

        vel_err = linvel_body[0] - self.command_vx
        r_tracking = W_TRACKING * math.exp(-(vel_err ** 2) / TRACKING_SIGMA)
        r_lin_vel_z = -W_LIN_VEL_Z * float(linvel_body[2] ** 2)
        r_ang_vel_xy = -W_ANG_VEL_XY * float(np.sum(angvel_body[:2] ** 2))
        r_ang_vel_z = -W_ANG_VEL_Z * float(angvel_body[2] ** 2)
        r_orientation = -W_ORIENTATION * float(roll ** 2 + pitch ** 2)
        r_torque = -W_TORQUE * float(np.sum(torque ** 2))
        r_dof_vel = -W_DOF_VEL * float(np.sum(joint_vel ** 2))
        r_dof_acc = -W_DOF_ACC * float(np.sum(((joint_vel - self.last_joint_vel) / CONTROL_DT) ** 2))
        r_action_rate = -W_ACTION_RATE * float(np.sum((action - self.prev_action) ** 2))
        limit_violation = np.clip(self.soft_low - joint_pos, 0, None) + np.clip(joint_pos - self.soft_high, 0, None)
        r_dof_pos_limits = -W_DOF_POS_LIMITS * float(np.sum(limit_violation))
        r_base_height = -W_BASE_HEIGHT * float((height - TARGET_HEIGHT) ** 2)

        reward = (W_ALIVE + r_tracking + r_lin_vel_z + r_ang_vel_xy + r_ang_vel_z
                  + r_orientation + r_torque + r_dof_vel + r_dof_acc
                  + r_action_rate + r_dof_pos_limits + r_base_height)
        if ONLY_POSITIVE_REWARDS:
            reward = max(reward, 0.0)

        self.prev_action = np.asarray(action, dtype=np.float32).copy()
        self.last_joint_vel = joint_vel.copy()
        self.step_count += 1

        fallen = hasFallen(roll, pitch) or height < MIN_HEIGHT
        terminated = bool(fallen)
        truncated = self.step_count >= self.max_steps

        info = {
            "reward_tracking": r_tracking, "reward_lin_vel_z": r_lin_vel_z,
            "reward_ang_vel_xy": r_ang_vel_xy, "reward_orientation": r_orientation,
            "reward_torque": r_torque, "reward_dof_vel": r_dof_vel,
            "reward_dof_acc": r_dof_acc, "reward_action_rate": r_action_rate,
            "reward_dof_pos_limits": r_dof_pos_limits, "reward_base_height": r_base_height,
            "vel_error": abs(vel_err),
        }
        return self._getObs(), reward, terminated, truncated, info

    def render(self):
        if self.render_mode != "human":
            return
        if self.viewer is None:
            import mujoco.viewer
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        if self.viewer.is_running():
            self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
