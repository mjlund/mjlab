# scripts/eval_velocity_tracking.py

from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from rsl_rl.runners import OnPolicyRunner

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.utils.torch import configure_torch_backends


# === EDIT THESE CONSTANTS TO MATCH YOUR PROJECT ==============================

TASK_NAME = "go1_velocity_flat"   # <-- EDIT ME: whatever task you trained
CHECKPOINT_PATH = (
    "logs/rsl_rl/go1_velocity_flat/checkpoints/Checkpoint_XXX.pt"
)  # <-- EDIT ME: your final checkpoint

COMMAND_TERM_KEY = "velocity"     # <-- EDIT ME if the key is different
#   You can find the correct key by temporarily printing:
#   print(env.unwrapped.command_manager.terms.keys())

# ============================================================================


def apply_play_overrides(cfg: ManagerBasedRlEnvCfg) -> None:
    """Similar to play._apply_play_env_overrides, but minimal."""
    cfg.episode_length_s = int(1e9)

    # Clean observations (no corruption / noise) if present
    if "policy" in cfg.observations:
        cfg.observations["policy"].enable_corruption = False

    # Remove random pushes if defined
    if cfg.events is not None:
        cfg.events.pop("push_robot", None)

    # Disable terrain curriculum if present
    if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
        tg = cfg.scene.terrain.terrain_generator
        tg.curriculum = False
        tg.num_cols = 5
        tg.num_rows = 5
        tg.border_width = 10.0


def make_env_and_policy(device: str):
    # Load configs
    env_cfg = load_env_cfg(TASK_NAME)
    agent_cfg = load_rl_cfg(TASK_NAME)

    # Evaluation settings
    env_cfg.scene.num_envs = 1  # single environment for tracking
    apply_play_overrides(env_cfg)

    # Build env
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # Build runner + load trained policy
    log_dir = Path("logs") / "rsl_rl" / agent_cfg.experiment_name
    checkpoint = Path(CHECKPOINT_PATH)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    runner = OnPolicyRunner(
        env, asdict(agent_cfg), log_dir=str(log_dir), device=device
    )
    runner.load(str(checkpoint), map_location=device)
    policy = runner.get_inference_policy(device=device)

    return env, policy


def get_velocity_command_term(env):
    """Return the velocity command term object inside the command manager."""
    cmd_terms = env.unwrapped.command_manager.terms
    print("Available command terms:", cmd_terms.keys())

    if COMMAND_TERM_KEY not in cmd_terms:
        raise KeyError(
            f"COMMAND_TERM_KEY='{COMMAND_TERM_KEY}' not found. "
            f"Available keys: {list(cmd_terms.keys())}"
        )

    vel_term = cmd_terms[COMMAND_TERM_KEY]
    return vel_term


def build_command_schedule(num_steps_per_segment: int = 125):
    """Return commanded (vx, vy, wz) for each env step."""
    cmds = []

    # 1) Forward walking: (vx, vy, wz) = (0 -> 0.6, 0, 0)
    for k in range(num_steps_per_segment):
        vx = 0.6 * k / (num_steps_per_segment - 1)
        cmds.append((vx, 0.0, 0.0))

    # 2) Lateral walking: (0, 0.4, 0)
    for _ in range(num_steps_per_segment):
        cmds.append((0.0, 0.4, 0.0))

    # 3) Turning: (0, 0, 0.4)
    for _ in range(num_steps_per_segment):
        cmds.append((0.0, 0.0, 0.4))

    # 4) Mixed: (0.5, 0, 0.3)
    for _ in range(num_steps_per_segment):
        cmds.append((0.5, 0.0, 0.3))

    cmds = np.array(cmds, dtype=np.float32)  # [T, 3]
    return cmds


def main():
    configure_torch_backends()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    env, policy = make_env_and_policy(device)
    vel_term = get_velocity_command_term(env)
    robot = vel_term.robot
    dt = env.unwrapped.step_dt

    # Reset env and get initial obs
    obs, _ = env.reset()

    cmd_seq = build_command_schedule(num_steps_per_segment=125)
    T = cmd_seq.shape[0]

    # Logs
    t_log = np.arange(T) * dt
    cmd_log = np.zeros((T, 3), dtype=np.float32)
    act_log = np.zeros((T, 3), dtype=np.float32)

    with torch.no_grad():
        for i in range(T):
            vx_cmd, vy_cmd, wz_cmd = cmd_seq[i]

            # --- 1) Write commanded velocity into the command term ---
            # shape: [num_envs, 3], we have num_envs=1 so index 0
            vel_term.vel_command_b[0, 0] = vx_cmd
            vel_term.vel_command_b[0, 1] = vy_cmd
            vel_term.vel_command_b[0, 2] = wz_cmd

            # --- 2) Policy inference ---
            action = policy(obs)
            obs, _, _, _ = env.step(action)

            # --- 3) Read actual velocities in base frame ---
            lin_b = robot.data.root_link_lin_vel_b[0]  # (3,)
            ang_b = robot.data.root_link_ang_vel_b[0]  # (3,)

            cmd_log[i] = np.array([vx_cmd, vy_cmd, wz_cmd], dtype=np.float32)
            act_log[i] = np.array(
                [
                    float(lin_b[0].item()),
                    float(lin_b[1].item()),
                    float(ang_b[2].item()),
                ],
                dtype=np.float32,
            )

    env.close()

    # === Plotting ============================================================
    labels = ["vx", "vy", "wz"]
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 8))

    for k in range(3):
        ax = axes[k]
        ax.plot(t_log, cmd_log[:, k], "--", label=f"command {labels[k]}")
        ax.plot(t_log, act_log[:, k], "-", label=f"actual {labels[k]}")
        ax.set_ylabel(labels[k])
        ax.grid(True)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("time [s]")
    fig.suptitle(f"Velocity command tracking – {TASK_NAME}")
    fig.tight_layout()
    out_path = Path("velocity_tracking_curve.png")
    fig.savefig(out_path, dpi=200)
    print(f"Saved plot to {out_path.resolve()}")


if __name__ == "__main__":
    main()
