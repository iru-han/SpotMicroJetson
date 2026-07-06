"""
Train a single PPO policy to track a forward velocity command on the
MuJoCo SpotMicroAI model (see gym_spotmicro_mujoco.py for the env/reward).

Usage:
    python3 train_ppo_mujoco.py --timesteps 2000000 --n-envs 8
"""

import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

from gym_spotmicro_mujoco import SpotMicroMuJoCoEnv

LOG_DIR = os.path.join(os.path.dirname(__file__), "ppo_runs")


class RenderCallback(BaseCallback):
    """Syncs the MuJoCo viewer every step — only meaningful with n_envs=1."""

    def _on_step(self):
        self.training_env.envs[0].render()
        return True


class RewardComponentLoggingCallback(BaseCallback):
    """Breaks ep_rew_mean down into its pieces (see gym_spotmicro_mujoco.py's
    step() info dict) so you can see e.g. whether torque cost or velocity
    tracking is dominating, instead of just one combined number."""

    KEYS = ("reward_tracking", "reward_lin_vel_z", "reward_ang_vel_xy", "reward_orientation",
            "reward_torque", "reward_dof_vel", "reward_dof_acc", "reward_action_rate",
            "reward_dof_pos_limits", "reward_base_height", "reward_foot_slip",
            "reward_feet_air_time", "reward_stuck_leg", "reward_imitation", "vel_error")

    def _on_step(self):
        for info in self.locals.get("infos", []):
            for key in self.KEYS:
                if key in info:
                    self.logger.record_mean(f"reward_components/{key}", info[key])
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--n-envs", type=int, default=16)
    parser.add_argument("--run-name", type=str, default="spotmicro_ppo")
    parser.add_argument("--render", action="store_true",
                         help="Show the MuJoCo viewer while training. Forces --n-envs 1 "
                              "and is much slower — use only to sanity-check behavior.")
    args = parser.parse_args()

    if args.render:
        args.n_envs = 1

    os.makedirs(LOG_DIR, exist_ok=True)

    env_kwargs = {"render_mode": "human"} if args.render else {}
    # SubprocVecEnv actually spreads envs across OS processes/CPU cores;
    # the default DummyVecEnv steps them one after another in this single
    # process, so n_envs>1 would give no real speedup. Rendering needs the
    # viewer in this process, so it stays on DummyVecEnv (n_envs is 1 anyway).
    vec_env_cls = None if args.render else SubprocVecEnv
    vec_env = make_vec_env(SpotMicroMuJoCoEnv, n_envs=args.n_envs, env_kwargs=env_kwargs, vec_env_cls=vec_env_cls)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        tensorboard_log=LOG_DIR,
        n_steps=1024,
        batch_size=256,
        learning_rate=3e-4,
        policy_kwargs=dict(net_arch=[128, 128]),
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=max(50_000 // args.n_envs, 1),
        save_path=os.path.join(LOG_DIR, args.run_name),
        name_prefix="ckpt",
    )

    callbacks = [checkpoint_cb, RewardComponentLoggingCallback()]
    if args.render:
        callbacks.append(RenderCallback())
    model.learn(total_timesteps=args.timesteps, callback=callbacks, tb_log_name=args.run_name)

    model.save(os.path.join(LOG_DIR, args.run_name, "final_model"))
    vec_env.save(os.path.join(LOG_DIR, args.run_name, "vecnormalize.pkl"))
    print(f"Done. Model saved under {os.path.join(LOG_DIR, args.run_name)}")


if __name__ == "__main__":
    main()
