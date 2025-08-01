import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qarray_env import QuantumDeviceEnv
import numpy as np
from tqdm import tqdm
import multiprocessing as mp

# test with fixed capacitance matrices
# need to randomise and pass in capacitances as well


def rollout(env, max_steps=50):
    voltages = []
    states = []
    env.reset()

    for _ in range(max_steps):
        action = env.action_space.sample()
        state, *_ = env.step(action)

        voltages.append(state['voltages'])

        image = state['image']
        image = np.array(image, dtype=np.float32) / 255.0
        states.append(image)
    
    return np.array(voltages), np.array(states)

def process_rollout(args):
    env, max_steps, save_dir, idx = args
    voltages, states = rollout(env=env, max_steps=max_steps)
    save_dir = f"{save_dir}/rollout_{idx}"

    os.makedirs(save_dir, exist_ok=True)
    np.savez_compressed(f"{save_dir}/data.npz", voltages=voltages, states=states)


def main():
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Collect CSD data")
    parser.add_argument("--rollouts", type=int, required=True, help="Number of rollouts to collect")
    parser.add_argument("--max_steps", type=int, default=50, help="Maximum steps per rollout")
    parser.add_argument("--save_dir", type=str, default="./eval_data", help="Directory to save collected data")
    parser.add_argument("--num_workers", type=int, default=mp.cpu_count(), help="Number of parallel workers")
    args = parser.parse_args()

    print(f"Using {args.num_workers} workers out of {mp.cpu_count()} available")

    env = QuantumDeviceEnv()
    print("Created environment")

    tasks = [(env, args.max_steps, args.save_dir, i) for i in range(args.rollouts)]

    with mp.Pool(processes=args.num_workers) as pool:
        for result in tqdm(pool.imap(process_rollout, tasks), total=args.rollouts):
            pass

    print("Done")


if __name__ == "__main__":
    main()
    
