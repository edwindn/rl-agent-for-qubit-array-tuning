import numpy as np
import jax
from jax import device_get as to_cpu
import os
import pickle
from pathlib import Path
import json
import matplotlib.image as mpimg
from functools import partial

from utils import sigmoid
from qarray_ndot_env_v2 import QuantumDeviceEnv as LargeArrayEnv
from qarray_2dot_env import QuantumDeviceEnv as _2DotEnv
from inference import load_agent, make_env, load_config

"""
Runs inference on large array using pairwise predictions on the voltages

currently merges action voltages using mean
"""

def run_inference(ckpt, ndots, max_steps, platform='cuda', merge_type='mean', done_threshold=0.5):
    assert merge_type in ['centered', 'mean', 'override']
    assert merge_type == 'mean', "Currently only 'mean' merge type is supported"

    dreamer_config = load_config()
    dreamer_config = dreamer_config.update({
        'task': 'custom_qarray2dot',
        'jax': {'platform': platform, 'debug': False, 'prealloc': False},
        'logdir': '/tmp/inference_logdir',
    })
    agent = load_agent(_2DotEnv, dreamer_config, ckpt)
    return

    if ndots == 2:
        array_env = _2DotEnv
    else:
        array_env = partial(LargeArrayEnv, ndots=ndots)
    num_scans = ndots - 1

    env = make_env(array_env, dreamer_config, render_mode='rgb_array')

    agent = load_agent(_2DotEnv, dreamer_config, ckpt) # for action size compatibility

    obs, _ = env.reset()
    carry = agent.init_policy(batch_size=1) # initial encoder embeddings
    carry_list = [carry] * num_scans

    reward = 0.0
    ep_reward = 0.0

    step = 0
    while True:
        step += 1
        is_first = (step == 1)

        meta_dict = {
            'is_first': np.array([is_first], dtype=bool),
            'is_last': np.array([False], dtype=bool),
            'is_terminal': np.array([False], dtype=bool),
            'reward': np.array([reward], dtype=np.float32)
        }

        outs = []
        img_list = []
        dones = []

        for i in range(num_scans):
            img = obs['image'][:,:,i:i+1]
            img = np.expand_dims(img, axis=0)
            img_list.append(img)
            
            v = obs['obs_voltages'][i:i+2]
            assert len(v) == 2, f"Expected 2 voltages, got {len(v)}"
            v = np.expand_dims(v, axis=0)

            input_obs = {'image': img, 'obs_voltages': v, **meta_dict}

            carry, acts, _ = agent.policy(carry_list[i], input_obs)
            carry_list[i] = carry

            out_voltages = acts['action'].flatten()
            outs.append(out_voltages)

            done_prob = acts.get('done', float('-inf'))
            done = sigmoid(done_prob) > done_threshold
            dones.append(done)

        if merge_type == 'mean':
            out_voltages = [outs[0][0]]
            for i in range(num_scans-1):
                out_voltages.append((outs[i][1]+outs[i+1][0])/2)
            out_voltages.append(outs[-1][1])
        else:
            raise NotImplementedError

        action = {
            'action': np.array(out_voltages, dtype=np.float32),
            'reset': False
        }

        env_out = env.step(action)
        obs = {
            'image': env_out['image'],
            'obs_voltages': env_out['obs_voltages']
        }
        reward = env_out['reward']
        truncated = env_out['is_last']
        terminated = env_out['is_terminal']

        ep_reward += reward

        dones = np.array(dones, dtype=np.bool)

        if np.all(dones) or terminated or truncated or (max_steps is not None and step >= max_steps):
            print(f"Episode terminated with reward {ep_reward:.2f} ({reward:.2f}) after {step} steps")
            break

    for i, img in enumerate(img_list):
        mpimg.imsave(f'final_img_{i}.png', img.squeeze())


def run_inference2(ckpt, max_steps, platform='cuda'):
    dreamer_config = load_config()
    dreamer_config = dreamer_config.update({
        'task': 'custom_qarray2dot',
        'jax': {'platform': platform, 'debug': False, 'prealloc': False},
        'logdir': '/tmp/inference_logdir',
    })

    env = make_env(_2DotEnv, dreamer_config, render_mode='rgb_array')

    agent = load_agent(_2DotEnv, dreamer_config, ckpt) # for action size compatibility

    obs, _ = env.reset()
    carry = agent.init_policy(batch_size=1) # initial encoder embeddings
    reward = 0.0
    ep_reward = 0.0

    step = 0
    while True:
        step += 1
        is_first = (step == 1)

        img = obs['image']
        img = np.expand_dims(img, axis=0)

        v = obs['obs_voltages']
        v = np.expand_dims(v, axis=0)

        meta_dict = {
            'is_first': np.array([is_first], dtype=bool),
            'is_last': np.array([False], dtype=bool),
            'is_terminal': np.array([False], dtype=bool),
            'reward': np.array([reward], dtype=np.float32)
        }

        obs = {'image': img, 'obs_voltages': v, **meta_dict}

        carry, acts, _ = agent.policy(carry, obs)

        action = {
            'action': acts['action'].flatten(),
            'reset': False
        }

        env_out = env.step(action)
        obs = {
            'image': env_out['image'],
            'obs_voltages': env_out['obs_voltages']
        }
        reward = env_out['reward']
        truncated = env_out['is_last']
        terminated = env_out['is_terminal']

        ep_reward += reward

        if terminated:
            print(f"Episode terminated successfully with reward {ep_reward} after {step} steps")
            break

        if truncated or (max_steps is not None and step >= max_steps):
            print(f"Episode truncated with reward {ep_reward} after {step} steps")
            break


def main():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--max_steps", type=int, default=None, help="Maximum number of steps for agent")
    parser.add_argument("--ndots", type=int, default=4, help="Number of dots in the quantum device")
    args = parser.parse_args()
    # later extend to large arrays

    run_inference(args.ckpt, args.ndots, args.max_steps, merge_type='mean')


if __name__ == "__main__":
    main()