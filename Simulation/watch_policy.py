"""
Watch a trained SpotMicroAI PPO policy walk in the MuJoCo viewer.

Usage:
    python3 watch_policy.py ppo_runs/spotmicro_ppo/final_model.zip
    python3 watch_policy.py ppo_runs/spotmicro_ppo/ckpt_100000_steps.zip
"""

import argparse
import os
import time

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from gym_spotmicro_mujoco import SpotMicroMuJoCoEnv, CONTROL_DT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("--vecnormalize", default=None,
                         help="Path to vecnormalize.pkl (defaults to the sibling file next to the model)")
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()

    vecnorm_path = args.vecnormalize
    if vecnorm_path is None:
        candidate = os.path.join(os.path.dirname(args.model_path), "vecnormalize.pkl")
        vecnorm_path = candidate if os.path.exists(candidate) else None

    env = DummyVecEnv([lambda: SpotMicroMuJoCoEnv(render_mode="human")])
    if vecnorm_path:
        env = VecNormalize.load(vecnorm_path, env)
        env.training = False
        env.norm_reward = False

    model = PPO.load(args.model_path)

    for ep in range(args.episodes):
        obs = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, info = env.step(action)
            done = bool(dones[0])
            total_reward += float(reward[0])
            env.envs[0].render()
            time.sleep(CONTROL_DT)
        print(f"episode {ep}: reward={total_reward:.1f}")


if __name__ == "__main__":
    main()
