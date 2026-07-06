"""
Run a trained SpotMicroAI policy and log per-step data to a CSV for offline
inspection (foot contacts, joint angles, position, reward breakdown) —
optionally watching it live in the MuJoCo viewer at the same time.

Usage:
    python3 collect_policy_data.py ppo_runs/spotmicro_ppo_v4/final_model.zip
    python3 collect_policy_data.py ppo_runs/spotmicro_ppo_v4/final_model.zip --no-render --episodes 3
"""

import argparse
import csv
import os
import time

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from gym_spotmicro_mujoco import SpotMicroMuJoCoEnv, CONTROL_DT, LEGS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("--vecnormalize", default=None)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--out", default="policy_data.csv")
    parser.add_argument("--no-render", action="store_true", help="Skip the MuJoCo viewer (faster, headless)")
    args = parser.parse_args()

    vecnorm_path = args.vecnormalize
    if vecnorm_path is None:
        candidate = os.path.join(os.path.dirname(args.model_path), "vecnormalize.pkl")
        vecnorm_path = candidate if os.path.exists(candidate) else None

    render_mode = None if args.no_render else "human"
    env = DummyVecEnv([lambda: SpotMicroMuJoCoEnv(render_mode=render_mode)])
    if vecnorm_path:
        env = VecNormalize.load(vecnorm_path, env)
        env.training = False
        env.norm_reward = False

    model = PPO.load(args.model_path)
    raw_env = env.envs[0]

    fieldnames = (["episode", "step", "t", "x", "y", "z", "vx_body", "cmd_vx"]
                  + [f"contact_{leg}" for leg in LEGS]
                  + ["reward", "vel_error"])

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for ep in range(args.episodes):
            obs = env.reset()
            done = False
            step = 0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, dones, info = env.step(action)
                done = bool(dones[0])

                in_contact = raw_env._footContacts()
                linvel_body, _, _ = raw_env._bodyFrameVectors()
                row = {
                    "episode": ep, "step": step, "t": round(step * CONTROL_DT, 3),
                    "x": raw_env.data.qpos[0], "y": raw_env.data.qpos[1], "z": raw_env.data.qpos[2],
                    "vx_body": linvel_body[0], "cmd_vx": raw_env.command_vx,
                    "reward": float(reward[0]), "vel_error": info[0].get("vel_error"),
                }
                for i, leg in enumerate(LEGS):
                    row[f"contact_{leg}"] = int(in_contact[i])
                writer.writerow(row)

                if render_mode:
                    raw_env.render()
                    time.sleep(CONTROL_DT)
                step += 1

    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
