
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm
from mjlab.managers.manager_term_config import CommandTermCfg
from mjlab.third_party.isaaclab.isaaclab.utils.math import (
  matrix_from_quat,
  quat_apply,
  wrap_to_pi,
)

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


class UniformVelocityCommand(CommandTerm):
  cfg: UniformVelocityCommandCfg

  def __init__(self, cfg: UniformVelocityCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    if self.cfg.heading_command and self.cfg.ranges.heading is None:
      raise ValueError("heading_command=True but ranges.heading is set to None.")
    if self.cfg.ranges.heading and not self.cfg.heading_command:
      raise ValueError("ranges.heading is set but heading_command=False.")

    self.robot: Entity = env.scene[cfg.asset_name]

    self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
    self.heading_target = torch.zeros(self.num_envs, device=self.device)
    self.heading_error = torch.zeros(self.num_envs, device=self.device)
    self.is_heading_env = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self.is_standing_env = torch.zeros_like(self.is_heading_env)

    self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)

    # Simple plotting setup
    if self.cfg.enable_plotting:
      self.plot_times = []
      self.plot_cmd = []
      self.plot_actual = []
      self.plot_foot_pos = []
      self.plot_counter = 0
      
      import signal, atexit
      atexit.register(self.save_plot)
      signal.signal(signal.SIGINT, lambda s, f: (self.save_plot(), exit(0)))

  @property
  def command(self) -> torch.Tensor:
    return self.vel_command_b

  def _update_metrics(self) -> None:
    max_command_time = self.cfg.resampling_time_range[1]
    max_command_step = max_command_time / self._env.step_dt
    self.metrics["error_vel_xy"] += (
      torch.norm(
        self.vel_command_b[:, :2] - self.robot.data.root_link_lin_vel_b[:, :2], dim=-1
      )
      / max_command_step
    )
    self.metrics["error_vel_yaw"] += (
      torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_link_ang_vel_b[:, 2])
      / max_command_step
    )

    # Store vals
    if self.cfg.enable_plotting:
      self.plot_counter += 1
      if self.plot_counter % self.cfg.plot_decimation == 0:
        idx = self.cfg.plot_env_idx
        self.plot_times.append(self.plot_counter * self._env.step_dt)
        self.plot_cmd.append([
          self.vel_command_b[idx, 0].item(),
          self.vel_command_b[idx, 1].item(),
          self.vel_command_b[idx, 2].item()
        ])
        self.plot_actual.append([
          self.robot.data.root_link_lin_vel_b[idx, 0].item(),
          self.robot.data.root_link_lin_vel_b[idx, 1].item(),
          self.robot.data.root_link_ang_vel_b[idx, 2].item()
        ])
        
 
        foot_pos = None
        if hasattr(self.robot.data, 'site_pos_w'):
          foot_pos = self.robot.data.site_pos_w[idx, self.cfg.foot_index]
        elif hasattr(self.robot.data, 'body_pos_w'):
          foot_pos = self.robot.data.body_pos_w[idx, self.cfg.foot_index]
        
        if foot_pos is not None:
          self.plot_foot_pos.append([
            foot_pos[0].item(),
            foot_pos[1].item(),
            foot_pos[2].item()
          ])
        else:
          self.plot_foot_pos.append([0.0, 0.0, 0.0])

  def save_plot(self):
    if not self.cfg.enable_plotting or len(self.plot_times) == 0:
      return
    
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import os
    
    os.makedirs(self.cfg.plot_output_dir, exist_ok=True)
    
    t = np.array(self.plot_times)
    cmd = np.array(self.plot_cmd)
    act = np.array(self.plot_actual)
    foot = np.array(self.plot_foot_pos)
    
    # X velocity
    plt.figure(figsize=(10, 4))
    plt.plot(t, cmd[:, 0], 'b-', label='Command', linewidth=2)
    plt.plot(t, act[:, 0], 'r--', label='Actual', linewidth=1.5)
    plt.ylabel('Vel X (m/s)')
    plt.xlabel('Time (s)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{self.cfg.plot_output_dir}/vel_x.png", dpi=150)
    plt.close()
    
    # Y velocity
    plt.figure(figsize=(10, 4))
    plt.plot(t, cmd[:, 1], 'b-', label='Command', linewidth=2)
    plt.plot(t, act[:, 1], 'r--', label='Actual', linewidth=1.5)
    plt.ylabel('Vel Y (m/s)')
    plt.xlabel('Time (s)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{self.cfg.plot_output_dir}/vel_y.png", dpi=150)
    plt.close()
    
    # Angular velocity
    plt.figure(figsize=(10, 4))
    plt.plot(t, cmd[:, 2], 'b-', label='Command', linewidth=2)
    plt.plot(t, act[:, 2], 'r--', label='Actual', linewidth=1.5)
    plt.ylabel('Vel Yaw (rad/s)')
    plt.xlabel('Time (s)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{self.cfg.plot_output_dir}/vel_yaw.png", dpi=150)
    plt.close()
    
    # Front left foot position
    plt.figure(figsize=(10, 4))
    #plt.plot(t, foot[:, 0], 'g-', label='X', linewidth=1.5)
    #plt.plot(t, foot[:, 1], 'b-', label='Y', linewidth=1.5)
    plt.plot(t, foot[:, 2], 'r-', label='Z (height)', linewidth=1.5)
    plt.ylabel('Foot Position (m)')
    plt.xlabel('Time (s)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{self.cfg.plot_output_dir}/foot_pos.png", dpi=150)
    plt.close()
    

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    r = torch.empty(len(env_ids), device=self.device)
    self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
    self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
    self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)
    if self.cfg.heading_command:
      assert self.cfg.ranges.heading is not None
      self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
      self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
    self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

    init_vel_mask = r.uniform_(0.0, 1.0) < self.cfg.init_velocity_prob
    init_vel_env_ids = env_ids[init_vel_mask]
    if len(init_vel_env_ids) > 0:
      root_pos = self.robot.data.root_link_pos_w[init_vel_env_ids]
      root_quat = self.robot.data.root_link_quat_w[init_vel_env_ids]
      lin_vel_b = self.robot.data.root_link_lin_vel_b[init_vel_env_ids]
      lin_vel_b[:, :2] = self.vel_command_b[init_vel_env_ids, :2]
      root_lin_vel_w = quat_apply(root_quat, lin_vel_b)
      root_ang_vel_b = self.robot.data.root_link_ang_vel_b[init_vel_env_ids]
      root_ang_vel_b[:, 2] = self.vel_command_b[init_vel_env_ids, 2]
      root_state = torch.cat(
        [root_pos, root_quat, root_lin_vel_w, root_ang_vel_b], dim=-1
      )
      self.robot.write_root_state_to_sim(root_state, init_vel_env_ids)

  # def _resample_command(self, env_ids: torch.Tensor) -> None:
  #   r = torch.empty(len(env_ids), device=self.device)

  #   step = (self._env._sim_step_counter)/96
  #   if step >=0 and step <125:
  #     self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x1)
  #     self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y1)
  #     self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z1)
  #   elif step >=125 and step <250:
  #     self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x2)
  #     self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y2)
  #     self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z2)
  #   elif step >=250 and step <375:
  #     self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x3)
  #     self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y3)
  #     self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z3)
  #   else:
  #     self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x4)
  #     self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y4)
  #     self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z4)
    
  #   if self.cfg.heading_command:
  #     assert self.cfg.ranges.heading is not None
  #     self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
  #     self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
  #   self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

  #   init_vel_mask = r.uniform_(0.0, 1.0) < self.cfg.init_velocity_prob
  #   init_vel_env_ids = env_ids[init_vel_mask]
  #   if len(init_vel_env_ids) > 0:
  #     root_pos = self.robot.data.root_link_pos_w[init_vel_env_ids]
  #     root_quat = self.robot.data.root_link_quat_w[init_vel_env_ids]
  #     lin_vel_b = self.robot.data.root_link_lin_vel_b[init_vel_env_ids]
  #     lin_vel_b[:, :2] = self.vel_command_b[init_vel_env_ids, :2]
  #     root_lin_vel_w = quat_apply(root_quat, lin_vel_b)
  #     root_ang_vel_b = self.robot.data.root_link_ang_vel_b[init_vel_env_ids]
  #     root_ang_vel_b[:, 2] = self.vel_command_b[init_vel_env_ids, 2]
  #     root_state = torch.cat(
  #       [root_pos, root_quat, root_lin_vel_w, root_ang_vel_b], dim=-1
  #     )
  #     self.robot.write_root_state_to_sim(root_state, init_vel_env_ids)

  def _update_command(self) -> None:
    if self.cfg.heading_command:
      self.heading_error = wrap_to_pi(self.heading_target - self.robot.data.heading_w)
      env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
      self.vel_command_b[env_ids, 2] = torch.clip(
        self.cfg.heading_control_stiffness * self.heading_error[env_ids],
        min=self.cfg.ranges.ang_vel_z[0],
        max=self.cfg.ranges.ang_vel_z[1],
      )
    standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
    self.vel_command_b[standing_env_ids, :] = 0.0

  # def _update_command(self) -> None:
  #   if self.cfg.heading_command:
  #     self.heading_error = wrap_to_pi(self.heading_target - self.robot.data.heading_w)
  #     env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
      
  #     step = (self._env._sim_step_counter)/96

  #     if step >=0 and step <125:
  #       min_val, max_val = self.cfg.ranges.ang_vel_z1
  #     elif step >=125 and step <250:
  #       min_val, max_val = self.cfg.ranges.ang_vel_z2
  #     elif step >=250 and step <375:
  #       min_val, max_val = self.cfg.ranges.ang_vel_z3
  #     else:
  #       min_val, max_val = self.cfg.ranges.ang_vel_z4

  #     self.vel_command_b[env_ids, 2] = torch.clip(
  #       self.cfg.heading_control_stiffness * self.heading_error[env_ids],
  #       min=min_val,max=max_val,
  #     )
  #   standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
  #   self.vel_command_b[standing_env_ids, :] = 0.0

  def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
    """Draw velocity command and actual velocity arrows."""
    batch = visualizer.env_idx
    if batch >= self.num_envs:
      return

    cmds = self.command.cpu().numpy()
    base_pos_ws = self.robot.data.root_link_pos_w.cpu().numpy()
    base_quat_w = self.robot.data.root_link_quat_w
    base_mat_ws = matrix_from_quat(base_quat_w).cpu().numpy()
    lin_vel_bs = self.robot.data.root_link_lin_vel_b.cpu().numpy()
    ang_vel_bs = self.robot.data.root_link_ang_vel_b.cpu().numpy()

    base_pos_w = base_pos_ws[batch]
    base_mat_w = base_mat_ws[batch]
    cmd = cmds[batch]
    lin_vel_b = lin_vel_bs[batch]
    ang_vel_b = ang_vel_bs[batch]

    if np.linalg.norm(base_pos_w) < 1e-6:
      return

    def local_to_world(vec: np.ndarray) -> np.ndarray:
      return base_pos_w + base_mat_w @ vec

    scale = self.cfg.viz.scale
    z = self.cfg.viz.z_offset

    # Draw arrows
    base = local_to_world(np.array([0, 0, z]) * scale)
    visualizer.add_arrow(base, local_to_world((np.array([0, 0, z]) + np.array([cmd[0], cmd[1], 0])) * scale), 
                        color=(0.2, 0.2, 0.6, 0.6), width=0.015)
    visualizer.add_arrow(base, local_to_world((np.array([0, 0, z]) + np.array([0, 0, cmd[2]])) * scale), 
                        color=(0.2, 0.6, 0.2, 0.6), width=0.015)
    visualizer.add_arrow(base, local_to_world((np.array([0, 0, z]) + np.array([lin_vel_b[0], lin_vel_b[1], 0])) * scale), 
                        color=(0.0, 0.6, 1.0, 0.7), width=0.015)
    visualizer.add_arrow(base, local_to_world((np.array([0, 0, z]) + np.array([0, 0, ang_vel_b[2]])) * scale), 
                        color=(0.0, 1.0, 0.4, 0.7), width=0.015)


@dataclass(kw_only=True)
class UniformVelocityCommandCfg(CommandTermCfg):
  asset_name: str
  heading_command: bool = False
  heading_control_stiffness: float = 1.0
  rel_standing_envs: float = 0.1
  rel_heading_envs: float = 0.3
  init_velocity_prob: float = 0.0
  resampling_time_range: tuple[float, float] = (3.0, 8.0)
  
  # For plotting
  enable_plotting: bool = True
  plot_decimation: int = 10
  plot_env_idx: int = 0
  plot_output_dir: str = "plots"
  foot_index: int = 0  

  class_type: type[CommandTerm] = UniformVelocityCommand

  @dataclass 
  class Ranges:
    # lin_vel_x1: tuple[float, float] = (0.0, 0.6) #changed, must be within (-1.0, 1.0)
    # lin_vel_y1: tuple[float, float] = (0.0, 0.0) #changed, must be within (-1.0, 1.0)
    # ang_vel_z1: tuple[float, float] = (0.0, 0.0) #changed,  must be within (-1.0, 1.0)

    # lin_vel_x2: tuple[float, float] = (0.0, 0.0) #changed, must be within (-1.0, 1.0)
    # lin_vel_y2: tuple[float, float] = (0.4, 0.4) #changed, must be within (-1.0, 1.0)
    # ang_vel_z2: tuple[float, float] = (0.0, 0.0) #changed,  must be within (-1.0, 1.0)

    # lin_vel_x3: tuple[float, float] = (-0.05, 0.05) #changed, must be within (-1.0, 1.0)
    # lin_vel_y3: tuple[float, float] = (-0.05, 0.05) #changed, must be within (-1.0, 1.0)
    # ang_vel_z3: tuple[float, float] = (0.4, 0.4) #changed,  must be within (-1.0, 1.0)

    # lin_vel_x4: tuple[float, float] = (0.6, 0.6) #changed, must be within (-1.0, 1.0)
    # lin_vel_y4: tuple[float, float] = (0.0, 0.0) #changed, must be within (-1.0, 1.0)
    # ang_vel_z4: tuple[float, float] = (0.3, 0.3) #changed,  must be within (-1.0, 1.0)

    #training values
    # lin_vel_x1: tuple[float, float] = (-0.2, 0.8) #changed, must be within (-1.0, 1.0)
    # lin_vel_y1: tuple[float, float] = (-0.5, 0.5) #changed, must be within (-1.0, 1.0)
    # ang_vel_z1: tuple[float, float] = (-0.5, 0.5) #changed,  must be within (-1.0, 1.0)

    # lin_vel_x2: tuple[float, float] = (-0.2, 0.8) #changed, must be within (-1.0, 1.0)
    # lin_vel_y2: tuple[float, float] = (-0.5, 0.5) #changed, must be within (-1.0, 1.0)
    # ang_vel_z2: tuple[float, float] = (-0.5, 0.5) #changed,  must be within (-1.0, 1.0)

    # lin_vel_x3: tuple[float, float] = (-0.2, 0.8) #changed, must be within (-1.0, 1.0)
    # lin_vel_y3: tuple[float, float] = (-0.5, 0.5) #changed, must be within (-1.0, 1.0)
    # ang_vel_z3: tuple[float, float] = (-0.5, 0.5) #changed,  must be within (-1.0, 1.0)

    # lin_vel_x4: tuple[float, float] = (-0.2, 0.8) #changed, must be within (-1.0, 1.0)
    # lin_vel_y4: tuple[float, float] = (-0.5, 0.5) #changed, must be within (-1.0, 1.0)
    # ang_vel_z4: tuple[float, float] = (-0.5, 0.5) #changed,  must be within (-1.0, 1.0)

    # lin_vel_x: tuple[float, float] = (0.6, 0.6) #changed, must be within (-1.0, 1.0)
    # lin_vel_y: tuple[float, float] = (0.0, 0.0) #changed, must be within (-1.0, 1.0)
    # ang_vel_z: tuple[float, float] = (0.3, 0.3) #changed,  must be within (-1.0, 1.0)



    lin_vel_x: tuple[float, float] = (0.1, 0.4)
    lin_vel_y: tuple[float, float] = (-0.1, 0.1)
    ang_vel_z: tuple[float, float] = (-0.1, 0.1)
    heading: tuple[float, float] | None = (-np.pi, np.pi)

  ranges: Ranges = field(default_factory=Ranges)

  @dataclass
  class VizCfg:
    z_offset: float = 0.2
    scale: float = 0.5

  viz: VizCfg = field(default_factory=VizCfg)

  def __post_init__(self):
    if self.heading_command and self.ranges.heading is None:
      raise ValueError(
        "The velocity command has heading commands active (heading_command=True) but "
        "the `ranges.heading` parameter is set to None."
      )
