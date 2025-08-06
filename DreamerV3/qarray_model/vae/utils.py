import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import colormaps  # Use the updated colormaps API
from multiprocessing import Pool, cpu_count
from torch.utils.data import TensorDataset
import torch


def process_file(file_path):
    """
    Process a single data file and return the processed data.
    """
    data = np.load(file_path, allow_pickle=True)
    voltages, images = data['voltages'], data['states']
    processed_data = []

    viridis = colormaps.get_cmap('viridis')

    for v, i in zip(voltages, images):
        i_normalized = (i - i.min()) / (i.max() - i.min() + 1e-8)  # Avoid division by zero
        # i_colored = viridis(i_normalized.squeeze())[:, :, :3]
        # i_colored = (i_colored * 255).astype(np.uint8)
        i_processed = np.transpose(i_normalized, (2, 0, 1))
        processed_data.append({'voltages': v, 'image': i_processed})

    return processed_data


def load_data(data_dir, num_samples=None):
    """
    Load training data from the specified directory using multiprocessing
    and convert grayscale images to 3-channel RGB using viridis colormap.
    """
    dataset = []

    # Get all data file paths
    shard_folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
    data_files = [os.path.join(data_dir, f, 'data.npz') for f in shard_folders]
    if num_samples is not None:
        data_files = data_files[:num_samples]

    # Use multiprocessing to process files in parallel
    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap(process_file, data_files), total=len(data_files), desc="Loading data"))

    # Combine results from all processes
    for result in results:
        dataset.extend(result)

    print(f"Loaded {len(dataset)} samples from {data_dir}")
    return dataset

    
def load_dataset(data_dir: str, num_samples: int) -> (TensorDataset, TensorDataset):
    """
    Load the dataset from the specified directory.
    Args:
        data_dir (str): Directory to load the dataset from.

    Returns:
        TensorDataset: Training and testing datasets.
    """
    dataset = load_data(data_dir, num_samples=num_samples)
    train_size = int(0.99 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_images, train_labels = zip(*[(torch.tensor(item['image']), torch.tensor(item['voltages'])) for item in train_dataset])
    test_images, test_labels = zip(*[(torch.tensor(item['image']), torch.tensor(item['voltages'])) for item in test_dataset])
    train_images = torch.stack(train_images)
    train_labels = torch.stack(train_labels)
    test_images = torch.stack(test_images)
    test_labels = torch.stack(test_labels)

    train_dataset = TensorDataset(train_images, train_labels)
    test_dataset = TensorDataset(test_images, test_labels)
    return train_dataset, test_dataset

if __name__ == "__main__":
    dataset = load_data('./data')
    all_voltages = np.array([d['voltages'] for d in dataset])
    max_v = np.max(all_voltages, axis=0)
    min_v = np.min(all_voltages, axis=0)
    print(f"Max voltages: {max_v}, Min voltages: {min_v}")

    exit()
    # Plot an image
    data = dataset[0]
    print(data['image'].shape)
    plt.imshow(dataset[0]['image'])
    plt.title(f"Voltages: {dataset[0]['voltages']}")
    plt.axis('off')
    plt.savefig('sample_image.png')
