import os
import numpy as np
from tqdm import tqdm

def load_data(data_dir):
    """
    Load training data from the specified directory.
    """
    dataset = []

    shard_folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
    for f in tqdm(shard_folders, desc="Loading data"):
        data_path = os.path.join(data_dir, f, 'data.npz')
        data = np.load(data_path, allow_pickle=True)
        voltages, images = data['voltages'], data['states']
        for v, i in zip(voltages, images):
            dataset.append({'voltages': v, 'image': i})

    print(f"Loaded {len(dataset)} samples from {data_dir}")
    return dataset