from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np
import torch
from .backflip_command import BackflipCommandCfg

from mjlab.entity import Entity
from mjlab.managers.manager_term_config import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor, ContactSensor
from mjlab.third_party.isaaclab.isaaclab.utils.math import quat_apply_inverse
from mjlab.third_party.isaaclab.isaaclab.utils.string import (
  resolve_matching_names_values,
)
from mjlab.third_party.isaaclab.isaaclab.utils.math import wrap_to_pi

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

##############backflip###########

def backflip_phase_reward(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    #assert command is not None, f"Command '{command_name}' not found."

    # 1. Access the Command Term and the Robot Asset
    command_term = env.command_manager.get_term("backflip")
    assert command_term is not None, "Backflip command term not found in command manager!"
    
    # 2. Extract Phase and State Information
    phi = command_term.metrics["phi"]
    pitch = command_term.metrics["root_euler_w"][:, 1]
    
    # Velocities
    lin_vel_w = asset.data.root_link_lin_vel_w
    ang_vel_b = asset.data.root_link_ang_vel_b
    
    reward = torch.zeros_like(phi)

    # --- Phase 1: Takeoff (0.0 -> 0.25) ---
    # Goal: Maximize vertical impulse and start the rotation
    takeoff_mask = (phi <= 0.25)
    if takeoff_mask.any():
        # Reward upward world velocity
        reward[takeoff_mask] += 2.0 * lin_vel_w[takeoff_mask, 2] 
        # Reward negative pitch velocity (backwards rotation)
        reward[takeoff_mask] += 0.5 * torch.abs(ang_vel_b[takeoff_mask, 1])

    # --- Phase 2: Flight (0.25 -> 0.75) ---
    # Goal: Match the rotation profile and maintain height
    flight_mask = (phi > 0.25) & (phi <= 0.75)
    if flight_mask.any():
        # Target pitch: progresses from 0 to -2*PI
        # We match the profile: -2 * pi * (phi - 0.25) / 0.5
        target_pitch = -2 * np.pi * torch.clamp((phi[flight_mask] - 0.25) / 0.5, 0.0, 1.0)
        pitch_error = torch.abs(wrap_to_pi(pitch[flight_mask] - target_pitch))
        
        reward[flight_mask] += 5.0 * torch.exp(-pitch_error / 0.5)
        
        # Reward for staying high in the air
        root_pos_w = asset.data.root_link_pos_w
        reward[flight_mask] += 3.0 * root_pos_w[flight_mask, 2]

    # --- Phase 3: Landing (0.75 -> 1.0) ---
    # Goal: Become upright and stop 
    landing_mask = (phi > 0.75)
    if landing_mask.any():
        # Reward being upright (pitch near 0 or -2PI)
        upright_error = torch.abs(wrap_to_pi(pitch[landing_mask]))
        reward[landing_mask] += 10.0 * torch.exp(-upright_error / 0.2)
        
        # Penalize horizontal and vertical velocity 
        vel_norm = torch.norm(lin_vel_w[landing_mask], dim=-1)
        reward[landing_mask] -= 0.5 * vel_norm

    return reward

def stability_penalty(env: ManagerBasedRlEnv):
    """Global stability penalty: keep roll and yaw at zero regardless of phase."""
    # asset_name = command_term.cfg.robot
    command_term = env.command_manager.get_term("backflip")
    assert command_term is not None, "Backflip command term not found in command manager!"
    euler = command_term.metrics["root_euler_w"]
    
    # Penalize any Roll (index 0) or Yaw (index 2)
    # Using square ensures smaller errors are penalized less than large deviations
    roll_error = torch.square(euler[:, 0])
    yaw_error = torch.square(euler[:, 2])
    
    return -2.0 * (roll_error + yaw_error)

def track_linear_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward for tracking the commanded base linear velocity.

  The commanded z velocity is assumed to be zero.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_lin_vel_b
  xy_error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
  z_error = torch.square(actual[:, 2])
  lin_vel_error = xy_error + z_error
  return torch.exp(-lin_vel_error / std**2)


def track_angular_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward heading error for heading-controlled envs, angular velocity for others.

  The commanded xy angular velocities are assumed to be zero.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_ang_vel_b
  z_error = torch.square(command[:, 2] - actual[:, 2])
  xy_error = torch.sum(torch.square(actual[:, :2]), dim=1)
  ang_vel_error = z_error + xy_error
  return torch.exp(-ang_vel_error / std**2)

def default_joint_position(
  env,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
):
  asset: Entity = env.scene[asset_cfg.name]
  current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
  desired_joint_pos = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
  error_squared = torch.square(current_joint_pos - desired_joint_pos)
  return torch.sum(torch.abs(current_joint_pos - desired_joint_pos), dim=1)


def flat_orientation(
  env: ManagerBasedRlEnv,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward flat base orientation (robot being upright).

  If asset_cfg has body_ids specified, computes the projected gravity
  for that specific body. Otherwise, uses the root link projected gravity.
  """
  asset: Entity = env.scene[asset_cfg.name]

  # If body_ids are specified, compute projected gravity for that body.
  if asset_cfg.body_ids:
    body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]  # [B, N, 4]
    body_quat_w = body_quat_w.squeeze(1)  # [B, 4]
    gravity_w = asset.data.gravity_vec_w  # [3]
    projected_gravity_b = quat_apply_inverse(body_quat_w, gravity_w)  # [B, 3]
    xy_squared = torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)
  else:
    # Use root link projected gravity.
    xy_squared = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
  return torch.exp(-xy_squared / std**2)


def base_z(
  env: ManagerBasedRlEnv,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward flat base orientation (robot being upright).

  If asset_cfg has body_ids specified, computes the projected gravity
  for that specific body. Otherwise, uses the root link projected gravity.
  """
  asset: Entity = env.scene[asset_cfg.name]

    # Use root link projected gravity.
  z_error = torch.square(asset.data.root_link_pos_w[:, 2] - 0.3)
  return torch.exp(-z_error / std**2)



def self_collision_cost(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Penalize self-collisions.

  Returns the number of self-collisions detected by the specified contact sensor.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  return sensor.data.found.squeeze(-1)


def body_angular_velocity_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize excessive body angular velocities."""
  asset: Entity = env.scene[asset_cfg.name]
  ang_vel = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids, :]
  ang_vel = ang_vel.squeeze(1)
  ang_vel_xy = ang_vel[:, :2]  # Don't penalize z-angular velocity.
  return torch.sum(torch.square(ang_vel_xy), dim=1)


def angular_momentum_penalty(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Penalize whole-body angular momentum to encourage natural arm swing."""
  angmom_sensor: BuiltinSensor = env.scene[sensor_name]
  angmom = angmom_sensor.data
  angmom_magnitude_sq = torch.sum(torch.square(angmom), dim=-1)
  angmom_magnitude = torch.sqrt(angmom_magnitude_sq)
  env.extras["log"]["Metrics/angular_momentum_mean"] = torch.mean(angmom_magnitude)
  return angmom_magnitude_sq


def feet_air_time(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  threshold_min: float = 0.05,
  threshold_max: float = 0.5,
  command_name: str | None = None,
  command_threshold: float = 0.5,
) -> torch.Tensor:
  """Reward feet air time."""
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  current_air_time = sensor_data.current_air_time
  assert current_air_time is not None
  in_range = (current_air_time > threshold_min) & (current_air_time < threshold_max)
  reward = torch.sum(in_range.float(), dim=1)
  in_air = current_air_time > 0
  num_in_air = torch.sum(in_air.float())
  mean_air_time = torch.sum(current_air_time * in_air.float()) / torch.clamp(
    num_in_air, min=1
  )
  env.extras["log"]["Metrics/air_time_mean"] = mean_air_time
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      scale = (total_command > command_threshold).float()
      reward *= scale
  return reward


def feet_clearance(
  env: ManagerBasedRlEnv,
  target_height: float,
  command_name: str | None = None,
  command_threshold: float = 0.01,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize deviation from target clearance height, weighted by foot velocity."""
  asset: Entity = env.scene[asset_cfg.name]
  foot_z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]  # [B, N]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, N, 2]
  vel_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, N]
  delta = torch.abs(foot_z - target_height)  # [B, N]
  cost = torch.sum(delta * vel_norm, dim=1)  # [B]
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      active = (total_command > command_threshold).float()
      cost = cost * active
  return cost


def soft_landing(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str | None = None,
  command_threshold: float = 0.05,
) -> torch.Tensor:
  """Penalize high impact forces at landing to encourage soft footfalls."""
  contact_sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = contact_sensor.data
  assert sensor_data.force is not None
  forces = sensor_data.force  # [B, N, 3]
  force_magnitude = torch.norm(forces, dim=-1)  # [B, N]
  first_contact = contact_sensor.compute_first_contact(dt=env.step_dt)  # [B, N]
  landing_impact = force_magnitude * first_contact.float()  # [B, N]
  cost = torch.sum(landing_impact, dim=1)  # [B]
  num_landings = torch.sum(first_contact.float())
  mean_landing_force = torch.sum(landing_impact) / torch.clamp(num_landings, min=1)
  env.extras["log"]["Metrics/landing_force_mean"] = mean_landing_force
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      active = (total_command > command_threshold).float()
      cost = cost * active
  return cost





