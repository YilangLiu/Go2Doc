# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Train a PPO agent using JAX on the specified environment."""

from datetime import datetime
import functools
import json
import os
import time
import warnings
import yaml
import numpy as np 

from absl import app
from absl import flags
from absl import logging
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.acme import running_statistics
from brax.training.checkpoint import load
from etils import epath
from flax.training import orbax_utils
import jax
import jax.numpy as jp
import mediapy as media
from ml_collections import config_dict
import mujoco
from orbax import checkpoint as ocp
from tensorboardX import SummaryWriter
import wandb

import mujoco_playground
from mujoco_playground import registry
from mujoco_playground import wrapper
from mujoco_playground.config import dm_control_suite_params
from mujoco_playground.config import locomotion_params
from mujoco_playground.config import manipulation_params

xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = "egl"

# Ignore the info logs from brax
logging.set_verbosity(logging.WARNING)

# Suppress warnings

# Suppress RuntimeWarnings from JAX
warnings.filterwarnings("ignore", category=RuntimeWarning, module="jax")
# Suppress DeprecationWarnings from JAX
warnings.filterwarnings("ignore", category=DeprecationWarning, module="jax")
# Suppress UserWarnings from absl (used by JAX and TensorFlow)
warnings.filterwarnings("ignore", category=UserWarning, module="absl")

# Generate unique experiment name
now = datetime.now()
timestamp = now.strftime("%Y%m%d-%H%M%S")
exp_name = f"RL_Flip-{timestamp}"
print(f"Experiment name: {exp_name}")

# Set up logging directory
logdir = epath.Path("logs").resolve() / exp_name
logdir.mkdir(parents=True, exist_ok=True)
print(f"Logs are being stored in: {logdir}")

def load_rl_policy(env_name: str, checkpoint_path: str):
  # Load environment configuration
  ppo_params = locomotion_params.brax_ppo_config(env_name)

  ckpt_path = epath.Path("logs").resolve() / checkpoint_path / "checkpoints" # epath.Path(_LOAD_CHECKPOINT_PATH.value).resolve()
  if ckpt_path.is_dir():
    latest_ckpts = list(ckpt_path.glob("*"))
    latest_ckpts = [ckpt for ckpt in latest_ckpts if ckpt.is_dir()]
    latest_ckpts.sort(key=lambda x: int(x.name))
    latest_ckpt = latest_ckpts[-1]
    restore_checkpoint_path = latest_ckpt
    print(f"Restoring from: {restore_checkpoint_path}")

  # Set up checkpoint directory
  ckpt_path = logdir / "checkpoints"
  ckpt_path.mkdir(parents=True, exist_ok=True)
  print(f"Checkpoint path: {ckpt_path}")

  network_factory=functools.partial(
    ppo_networks.make_ppo_networks,
        **ppo_params.network_factory,
        preprocess_observations_fn=running_statistics.normalize,
      )
  env_cfg = registry.get_default_config(env_name)
  env = registry.load(env_name, config=env_cfg)

  obs_size = env.observation_size
  act_size = env.action_size
  ppo_network = network_factory(obs_size, act_size)
  params = load(restore_checkpoint_path)
  
  # Create inference function
  make_inference_fn = ppo_networks.make_inference_fn(ppo_network)
  inference_fn = make_inference_fn(params, deterministic=True)
  jit_inference_fn = jax.jit(inference_fn)
  return jit_inference_fn, env_cfg

def main(argv):
  """Run training and evaluation for the specified environment."""
  del argv

  # stance to footstand
  policy_1, env_cfg = load_rl_policy("Go2Footstand", "Go2Footstand-20250908-202459")

  # footstand to handstand 
  policy_2, env_cfg = load_rl_policy("Go2FlipRL", "Go2FlipRL-20250908-215914")
  
  # handstand to stance
  policy_3, _ = load_rl_policy("Go2RestoreRL", "Go2RestoreRL-20250908-221049")

  # Prepare for evaluation
  eval_env = registry.load("Go2Handstand", config=env_cfg)

  jit_reset = jax.jit(eval_env.reset)
  jit_step = jax.jit(eval_env.step)

  rng = jax.random.PRNGKey(123)
  rng, reset_rng = jax.random.split(rng)
  
  state = jit_reset(reset_rng)
  rollout = [state]
  # import pdb; pdb.set_trace()
  # Run evaluation rollout
  for t in range(env_cfg.episode_length):
    # act_rng, rng = jax.random.split(rng)
    # ctrl, _ = policy_2(state.obs, act_rng)
    # state = jit_step(state, ctrl)
    
    if t < 50:
      act_rng, rng = jax.random.split(rng)
      ctrl, _ = policy_1(state.obs, act_rng)
      state = jit_step(state, ctrl)

    elif t >= 50 and t < 150:
      act_rng, rng = jax.random.split(rng)
      ctrl, _ = policy_2(state.obs, act_rng)
      state = jit_step(state, ctrl)
    
    else:
      act_rng, rng = jax.random.split(rng)
      ctrl, _ = policy_3(state.obs, act_rng)
      state = jit_step(state, ctrl)

    rollout.append(state)

    # if state.done:
    #   print("rollout done")
    #   break

  # Render and save the rollout
  render_every = 1
  fps = 1.0 / eval_env.dt / render_every
  print(f"FPS for rendering: {fps}")

  traj = rollout[::render_every]

  scene_option = mujoco.MjvOption()
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = False
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False

  frames = eval_env.render(
      traj, 
      camera="track",
      height=480, 
      width=640, 
      # scene_option=scene_option
  )
  video_path = logdir/"video.mp4"
  media.write_video(f"{video_path}", frames, fps=fps)
  print("Rollout video saved!")

  traj_path = logdir/"state.npy"
  traj_path_vel = logdir/"state_vel.npy"
  traj_state = []
  traj_state_vel = []
  for state in traj:
    traj_state.append(state.data.qpos)
    traj_state_vel.append(state.data.qvel)
  traj_state = np.array(traj_state)
  traj_state_vel = np.array(traj_state_vel)
  np.savetxt(traj_path, traj_state)
  np.savetxt(traj_path_vel, traj_state_vel)

if __name__ == "__main__":
  app.run(main)
