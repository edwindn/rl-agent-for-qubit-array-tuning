import numpy as np

from qarray_4dot_env import QuantumDeviceEnv
from inference import load_agent, make_env, load_config

"""
Runs inference on large array using pairwise predictions on the voltages
"""

def run_inference(ckpt, max_steps, platform='cuda'):
    env = QuantumDeviceEnv()

    dreamer_config = load_config()
    dreamer_config = dreamer_config.update({
        'task': 'custom_qarray2dot',
        'jax': {'platform': platform, 'debug': False, 'prealloc': False},
        'logdir': '/tmp/inference_logdir',
    })

    agent = load_agent(QuantumDeviceEnv, dreamer_config, ckpt)


    obs, _ = env.reset()
    for step in range(max_steps):
        img1, img2, img3 = obs[:,:,0:1], obs[:,:,1:2], obs[:,:,2:3]
        voltage_outs1 = agent.act(img1)
        voltage_outs2 = agent.act(img2)
        voltage_outs3 = agent.act(img3)
        voltages_outs = [voltage_outs1[0], (voltage_outs1[1]+voltage_outs2[0])/2, (voltage_outs2[1]+voltage_outs3[0])/2, voltage_outs3[1]]

        obs, reward, terminated, truncated, _ = env.step(voltage_outs)

        if terminated:
            print(f"Episode terminated successfully after {step} steps with reward {reward}")
        if truncated:
            print(f"Episode truncated with reward {reward}")

        if terminated or truncated: break


def test_run(platform='cuda'):

    dreamer_config = load_config()
    config = config.update({
        'task': 'custom_qarray2dot',
        'jax': {'platform': platform, 'debug': False, 'prealloc': False},
        'logdir': '/tmp/inference_logdir',
    })

    agent = load_agent(dreamer_config, args.ckpt)
    make_env_fn = lambda: make_env(QuantumDeviceEnv, config, render_mode='rgb_array')
    driver = embodied.Driver([make_env_fn], parallel=False)

    frame_data = {'step': 0, 'output_dir': output_dir, 'episode_step': 0}
    episode_data = {'count': 0, 'target': num_episodes, 'current_steps': 0, 'current_reward': 0.0}

    def rollout_callback(tran, worker):
        if worker == 0:
            episode_data['current_steps'] += 1
            episode_data['current_reward'] += tran.get('reward', 0.0)
            frame_data['episode_step'] += 1

            is_last = tran.get('is_last', False)
            is_terminal = tran.get('is_terminal', False)

            # save frames logic here

    driver.on_step(rollout_callback)
    policy = lambda *args: agent.policy(*args, mode='eval')
    driver.reset(agent.init_policy)
    driver(policy, episodes=1)
    driver.close()
    print('\nInference completed')


def main():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--max_steps", type=int, default=100, help="Maximum number of steps for agent")
    args = parser.parse_args()
    # later extend to large arrays

    run_inference(args.ckpt, args.max_steps)


if __name__ == "__main__":
    main()