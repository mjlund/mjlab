from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.backflip.rl import BackflipOnPolicyRunner

from .env_cfgs import (
  UNITREE_GO2_FLAT_ENV_CFG,
  UNITREE_GO2_ROUGH_ENV_CFG,
)
from .rl_cfg import UNITREE_GO2_PPO_RUNNER_CFG

register_mjlab_task(
  task_id="Mjlab-Backflip-Rough-Unitree-Go2",
  env_cfg=UNITREE_GO2_ROUGH_ENV_CFG,
  rl_cfg=UNITREE_GO2_PPO_RUNNER_CFG,
  runner_cls=BackflipOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Backflip-Flat-Unitree-Go2",
  env_cfg=UNITREE_GO2_FLAT_ENV_CFG,
  rl_cfg=UNITREE_GO2_PPO_RUNNER_CFG,
  runner_cls=BackflipOnPolicyRunner,
)

