import numpy as np
import jax
from jax import device_get as to_cpu
import os
import pickle
from pathlib import Path
import json
import matplotlib.image as mpimg

from qarray_4dot_env import QuantumDeviceEnv as LargeArrayEnv
from qarray_2dot_env import QuantumDeviceEnv as _2DotEnv
from inference import load_agent, make_env, load_config

"""
Runs inference on large array using pairwise predictions on the voltages

currently merges action voltages using mean
"""

def run_inference(ckpt, max_steps, platform='cuda', merge_type='centered'):
    assert merge_type in ['centered', 'mean', 'override']

    dreamer_config = load_config()
    dreamer_config = dreamer_config.update({
        'task': 'custom_qarray2dot',
        'jax': {'platform': platform, 'debug': False, 'prealloc': False},
        'logdir': '/tmp/inference_logdir',
    })

    env = make_env(LargeArrayEnv, dreamer_config, render_mode='rgb_array')

    agent = load_agent(_2DotEnv, dreamer_config, ckpt) # for action size compatibility

    obs, _ = env.reset()
    carry = agent.init_policy(batch_size=1) # initial encoder embeddings
    carry1 = carry
    carry2 = carry
    carry3 = carry
    reward = 0.0
    ep_reward = 0.0

    step = 0
    while True:
        step += 1
        is_first = (step == 1)

        img1 = obs['image'][:,:,0:1]
        img1 = np.expand_dims(img1, axis=0)
        img2 = obs['image'][:,:,1:2]
        img2 = np.expand_dims(img2, axis=0)
        img3 = obs['image'][:,:,2:3]
        img3 = np.expand_dims(img3, axis=0)

        v1 = obs['obs_voltages'][0:2]
        v1 = np.expand_dims(v1, axis=0)
        v2 = obs['obs_voltages'][1:3]
        v2 = np.expand_dims(v2, axis=0)
        v3 = obs['obs_voltages'][2:]
        v3 = np.expand_dims(v3, axis=0)

        meta_dict = {
            'is_first': np.array([is_first], dtype=bool),
            'is_last': np.array([False], dtype=bool),
            'is_terminal': np.array([False], dtype=bool),
            'reward': np.array([reward], dtype=np.float32)
        }

        obs1 = {'image': img1, 'obs_voltages': v1, **meta_dict}
        obs2 = {'image': img2, 'obs_voltages': v2, **meta_dict}
        obs3 = {'image': img3, 'obs_voltages': v3, **meta_dict}

        carry1, acts1, _ = agent.policy(carry1, obs1)
        carry2, acts2, _ = agent.policy(carry2, obs2)
        carry3, acts3, _ = agent.policy(carry3, obs3)

        out_voltages1 = acts1['action'].flatten()
        out_voltages2 = acts2['action'].flatten()
        out_voltages3 = acts3['action'].flatten()
        if merge_type == 'centered':
            out_voltages = [out_voltages1[0]] + out_voltages2.tolist() + [out_voltages3[1]]
        elif merge_type == 'mean':
            out_voltages = [out_voltages1[0], (out_voltages1[1]+out_voltages2[0])/2, (out_voltages2[1]+out_voltages3[0])/2, out_voltages3[1]]
        elif merge_type == 'override':
            out_voltages = [out_voltages1[0]] + [out_voltages2[0]] + out_voltages3.tolist()

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

        if terminated:
            print(f"Episode terminated successfully with reward {ep_reward} after {step} steps")
            break

        if truncated or (max_steps is not None and step >= max_steps):
            print(f"Episode truncated with reward {ep_reward} after {step} steps")
            break

    mpimg.imsave('final_img1.png', img1.squeeze())
    mpimg.imsave('final_img2.png', img2.squeeze())
    mpimg.imsave('final_img3.png', img3.squeeze())


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
    args = parser.parse_args()
    # later extend to large arrays

    run_inference(args.ckpt, args.max_steps, merge_type='mean')


if __name__ == "__main__":
    main()