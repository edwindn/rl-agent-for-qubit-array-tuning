import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import colormaps  # Use the updated colormaps API

def load_data(data_dir):
    """
    Load training data from the specified directory and convert grayscale images to 3-channel RGB using viridis colormap.
    """
    dataset = []

    # Initialize the viridis colormap
    viridis = colormaps.get_cmap('viridis')  # Updated to use colormaps.get_cmap

    shard_folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
    
    for f in tqdm(shard_folders, desc="Loading data"):
        data_path = os.path.join(data_dir, f, 'data.npz')
        data = np.load(data_path, allow_pickle=True)
        voltages, images = data['voltages'], data['states']
        for v, i in zip(voltages, images):
            # Normalize the grayscale image to the range [0, 1]
            i_normalized = (i - i.min()) / (i.max() - i.min() + 1e-8)  # Avoid division by zero
            
            # Apply the viridis colormap
            i_colored = viridis(i_normalized.squeeze())[:, :, :3]  # Get RGB channels (ignore alpha)
            
            # Convert to uint8 format (optional, for consistency)
            i_colored = (i_colored * 255).astype(np.uint8)
            
            dataset.append({'voltages': v, 'image': i_colored})

    print(f"Loaded {len(dataset)} samples from {data_dir}")
    return dataset

if __name__ == "__main__":
    dataset = load_data('./data')
    print(f"Dataset contains {len(dataset)} samples.")
    # Plot an image
    data = dataset[0]
    print(data['image'].shape)
    plt.imshow(dataset[0]['image'])
    plt.title(f"Voltages: {dataset[0]['voltages']}")
    plt.axis('off')
    plt.savefig('sample_image.png')
