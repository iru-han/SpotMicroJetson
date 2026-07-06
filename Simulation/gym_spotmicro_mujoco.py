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

Revision history (kept short — see git log for the full story):
- v1: standing still already scored well (alive bonus + passive posture/height
  terms), so the policy just stood there. Sharpened tracking, cut the free
  alive bonus.
- v2: fixed standing-still, but nothing punished a planted foot sliding along
  the ground, so it dragged itself forward instead of stepping. Added an
  explicit foot-slip penalty plus a same-instant "diagonal pair in contact"
  bonus.
- v3: the diagonal-pair bonus only checks the current instant, so the policy
  found it could satisfy it by permanently parking one leg in the air and
  shuffling on the other three (no rule required every leg to eventually
  bear weight). Replaced it with legged_gym's feet_air_time reward (pays out
  per foot only at touchdown, scaled by how long that foot was airborne —
  same mechanism Unitree's rl_gym uses) plus a hard penalty if any single
  foot stays airborne far longer than a normal swing phase.
- v4: forward velocity was measured as +local-x, but the MJCF's own front_link
  sits at local x=-0.145 (rear_link at +0.135) — the reward was tracking
  motion toward the robot's rear the whole time. Also added an imitation
  reward toward the reference joint trajectory the existing hand-tuned
  TrottingGait (kinematicMotion.py) would produce for the commanded speed,
  so the learned gait is pulled toward a shape we already know looks right,
  on top of (not instead of) the free-form shaping above.
"""

import os
import sys
import math

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from spotmicro_common import MODEL_PATH, LEGS, PARTS, DIRS, PYBULLET_INIT_QUAT_WXYZ, quatToEuler, hasFallen
from Kinematics.kinematics import Kinematic
from kinematicMotion import TrottingGait

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
# First training run (2M steps) converged to "stand still and don't fall" —
# episode length went from ~30 to ~700 steps while vel_error stayed flat or
# got worse. Standing still was already earning enough from W_ALIVE + the
# passive orientation/height/torque terms that walking's extra risk (falling)
# and cost (torque/dof_acc) wasn't worth it. Sharpened tracking and cut the
# free reward for just surviving so actual walking is required to score well.
W_TRACKING = 3.0
TRACKING_SIGMA = 0.15
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
W_ALIVE = 0.15

# Nothing above penalizes a foot sliding along the ground, so dragging a
# planted foot to shove the body forward was a cheaper way to earn
# r_tracking than a real lift-and-place trot (Kim et al.'s foot-slip cost c7).
W_FOOT_SLIP = 0.5

# feet_air_time: legged_gym's mechanism (github.com/unitreerobotics/unitree_rl_gym)
# for shaping proper swing-then-land steps. Paid out per foot only at the
# instant it touches down, scaled by (how long it was airborne - target) —
# too-quick tapping scores negative, a full natural swing scores positive.
W_FEET_AIR_TIME = 1.0
TARGET_AIR_TIME = 0.25  # seconds, roughly a quarter of a natural stride

# feet_air_time only pays at touchdown, so a foot that's parked in the air
# forever just forfeits that reward — it isn't actively punished. That let a
# policy permanently lift one leg and shuffle on the other three. This adds
# a real, escalating penalty once a foot has been airborne much longer than
# a normal swing, making "never land" costly instead of merely unrewarded.
W_STUCK_LEG = 2.0
MAX_AIR_TIME = 0.6  # seconds

# Imitation reward: reuses the existing, already-tuned TrottingGait
# (kinematicMotion.py) as a reference — at each step we ask "what joint
# angles would the hand-crafted trot use right now for this commanded
# speed?" and reward the policy for landing close to that, on top of (not
# instead of) the free-form shaping above. Weighted heavily per the request
# to lean on this as the dominant shaping signal.
W_IMITATION = 5.0
IMITATION_SIGMA = 0.6
# mm of step length per (m/s) of commanded speed, negative because this
# gait convention steps backward-signed for forward motion (see the W/S key
# mapping in Common/multiprocess_kb.py: forward = negative IDstepLength).
STEP_LENGTH_PER_MPS = -120.0
STEP_LENGTH_LIMIT = 120.0
GAIT_BODY_ROT = (0, 0, 0)
GAIT_BODY_POS = (50, 60, 0)  # same stance offset used throughout the project


class _StaticGaitParams:
    """Feeds TrottingGait fixed defaults — no GUI sliders here, just the
    values it was constructed with (spur width, step timing, etc.)."""

    def __init__(self):
        self.vals = []

    def addUserDebugParameter(self, name, low, high, default):
        self.vals.append(default)
        return len(self.vals) - 1

    def readUserDebugParameter(self, handle):
        return self.vals[handle]


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

        self.ground_geom_id = self.model.geom("ground").id
        self.foot_geom_ids = [self.model.geom(f"{leg}_toe_link_collision").id for leg in LEGS]
        self.foot_body_ids = [self.model.body(f"{leg}_toe_link").id for leg in LEGS]

        self.init_qpos = self.data.qpos.copy()
        self.init_qpos[2] = 0.3
        self.init_qpos[3:7] = PYBULLET_INIT_QUAT_WXYZ
        self.init_qpos[self.joint_qpos_adr] = DEFAULT_JOINT_ANGLES

        n = len(self.joint_names)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(n,), dtype=np.float32)
        obs_dim = 3 + 3 + 3 + n + n + n + 1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self.kin = Kinematic()
        self.trotting = TrottingGait(_StaticGaitParams())
        self.gait_time = 0.0

        self.viewer = None
        self.prev_action = np.zeros(n, dtype=np.float32)
        self.last_joint_vel = np.zeros(n, dtype=np.float32)
        self.command_vx = 0.5
        self.step_count = 0
        self.feet_air_time = np.zeros(4, dtype=np.float32)

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

    def _footContacts(self):
        in_contact = np.zeros(4, dtype=bool)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            other = None
            if c.geom1 == self.ground_geom_id:
                other = c.geom2
            elif c.geom2 == self.ground_geom_id:
                other = c.geom1
            if other in self.foot_geom_ids:
                in_contact[self.foot_geom_ids.index(other)] = True
        return in_contact

    def _footSlip(self, in_contact):
        slip = 0.0
        vel = np.zeros(6)
        for leg_idx, touching in enumerate(in_contact):
            if touching:
                mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY,
                                          self.foot_body_ids[leg_idx], vel, 0)
                slip += float(np.sum(vel[3:5] ** 2))  # world-frame linear vx, vy
        return slip

    def _referenceJointAngles(self):
        step_length = np.clip(STEP_LENGTH_PER_MPS * self.command_vx, -STEP_LENGTH_LIMIT, STEP_LENGTH_LIMIT)
        kb_offset = {"IDstepLength": float(step_length), "IDstepWidth": 0.0,
                     "IDstepAlpha": 0.0, "StartStepping": True}
        Lp = self.trotting.positions(self.gait_time, kb_offset)
        angles = self.kin.calcIK(Lp, GAIT_BODY_ROT, GAIT_BODY_POS)
        return np.array([angles[lx][px] * DIRS[lx][px] for lx in range(4) for px in range(3)])

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
        self.feet_air_time[:] = 0.0
        self.command_vx = float(self.np_random_.uniform(*self.command_vx_range))
        self.step_count = 0
        # Starts negative like the hand-tuned demo (pybullet_automatic_gait.py's
        # `d - 3`): keeps the gait clock away from t=0 exactly, which the
        # underlying TrottingGait divides by zero on, and gives a brief
        # "settle" window before stepping starts.
        self.gait_time = -3.0

        return self._getObs(), {}

    def step(self, action):
        target = self._actionToTarget(action)
        self.data.ctrl[self.actuator_ids] = target

        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)

        self.gait_time += CONTROL_DT

        linvel_body, angvel_body, _ = self._bodyFrameVectors()
        roll, pitch, _ = quatToEuler(*self.data.qpos[3:7])
        torque = self.data.actuator_force[self.actuator_ids]
        joint_pos = self.data.qpos[self.joint_qpos_adr]
        joint_vel = self.data.qvel[self.joint_dof_adr]
        height = self.data.qpos[2]

        # The MJCF's front_link sits at local x=-0.145 and rear_link at
        # x=+0.135 (see urdf/spot_micro.xml) — the robot's front points
        # toward local -X, so "forward" is the negated local-x velocity.
        forward_vel = -linvel_body[0]
        vel_err = forward_vel - self.command_vx
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

        in_contact = self._footContacts()
        r_foot_slip = -W_FOOT_SLIP * self._footSlip(in_contact)

        first_contact = (self.feet_air_time > 0.0) & in_contact
        r_feet_air_time = W_FEET_AIR_TIME * float(np.sum((self.feet_air_time - TARGET_AIR_TIME) * first_contact))
        excess_air_time = np.clip(self.feet_air_time - MAX_AIR_TIME, 0, None)
        r_stuck_leg = -W_STUCK_LEG * float(np.sum(excess_air_time ** 2))

        self.feet_air_time += CONTROL_DT
        self.feet_air_time[in_contact] = 0.0

        reference_angles = self._referenceJointAngles()
        r_imitation = W_IMITATION * math.exp(-float(np.sum((joint_pos - reference_angles) ** 2)) / IMITATION_SIGMA)

        reward = (W_ALIVE + r_tracking + r_lin_vel_z + r_ang_vel_xy + r_ang_vel_z
                  + r_orientation + r_torque + r_dof_vel + r_dof_acc
                  + r_action_rate + r_dof_pos_limits + r_base_height
                  + r_foot_slip + r_feet_air_time + r_stuck_leg + r_imitation)
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
            "reward_foot_slip": r_foot_slip, "reward_feet_air_time": r_feet_air_time,
            "reward_stuck_leg": r_stuck_leg, "reward_imitation": r_imitation,
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
