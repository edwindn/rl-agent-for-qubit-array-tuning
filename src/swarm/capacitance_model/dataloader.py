import torch
import torch.utils.data as data
import numpy as np
import os
import glob
from typing import Tuple, List, Optional, Union
from sklearn.model_selection import train_test_split

from capacitance_utils import get_channel_targets


class CapacitanceDataset(data.Dataset):
    """
    PyTorch dataset for loading multi-channel capacitance images and Cgd targets.
    
    Loads data from batched NPY files containing multi-channel images and CGD matrices.
    Each image channel corresponds to different charge sensor measurements with different target CGD values.
    """
    
    def __init__(
        self, 
        data_dirs: Union[str, List[str]], 
        transform: Optional[callable] = None,
        load_to_memory: bool = False,
        validate_data: bool = True
    ):
        """
        Initialize dataset.
        
        Args:
            data_dir: Path to dataset directory containing images/ and parameters/ (can be a list of datasets)
            transform: Optional image transforms
            load_to_memory: If True, load all data to memory (requires ~11.5GB)
            validate_data: If True, validate data consistency during initialization
        """
        self.data_dirs = data_dirs if isinstance(data_dirs, list) else [data_dirs]
        self.transform = transform
        self.load_to_memory = load_to_memory

        print(f"Loading data files from the following directories: {' '.join(self.data_dirs)}")

        # Get all batch files
        self.image_files = []
        self.cgd_files = []
        for data_dir in self.data_dirs:
            self.image_files.extend(sorted(glob.glob(os.path.join(data_dir, "images", "batch_*.npy"))))
            self.cgd_files.extend(sorted(glob.glob(os.path.join(data_dir, "cgd_matrices", "batch_*.npy"))))
        
        assert len(self.image_files) == len(self.cgd_files), \
            f"Mismatch: {len(self.image_files)} image files vs {len(self.cgd_files)} cgd files"
        
        print(f"Found {len(self.image_files)} batch files")
        
        # Build index mapping global_idx -> (file_idx, local_idx, channel_idx)
        self._build_index()
        
        if validate_data:
            self._validate_data()
        
        if load_to_memory:
            print("Loading all data to memory...")
            self._load_to_memory()
        
    def _build_index(self):
        """Build mapping from global index to (batch_file_idx, local_idx, channel_idx)"""
        self.index_map = []
        self.batch_sizes = []
        
        for file_idx, (img_file, cgd_file) in enumerate(zip(self.image_files, self.cgd_files)):
            # Load images to get batch size and number of channels
            images = np.load(img_file, allow_pickle=True)
            batch_size = images.shape[0]
            num_channels = images.shape[-1] if len(images.shape) == 4 else 1  # (batch, H, W, channels)
            
            self.batch_sizes.append(batch_size)
            
            # Create index entries for each sample and each channel
            for local_idx in range(batch_size):
                for channel_idx in range(num_channels):
                    self.index_map.append((file_idx, local_idx, channel_idx))
        
        self.total_samples = len(self.index_map)
        print(f"Total samples: {self.total_samples} (including all channels)")
        
    def _validate_data(self):
        """Validate data consistency across a few sample batches"""
        print("Validating data consistency...")
        
        # Check first, middle, and last batch
        check_indices = [0, len(self.image_files) // 2, len(self.image_files) - 1]
        
        for idx in check_indices:
            img_file = self.image_files[idx]
            cgd_file = self.cgd_files[idx]
            
            images = np.load(img_file, allow_pickle=True)
            cgd_matrices = np.load(cgd_file, allow_pickle=True)
            
            # Check shapes and structure
            assert len(images) == len(cgd_matrices), f"Batch {idx}: image/cgd count mismatch"
            assert images.ndim == 4, f"Batch {idx}: expected 4D images (batch, H, W, channels), got {images.ndim}D"
            assert images.shape[1] == images.shape[2], f"Batch {idx}: images not square"
            
            # Check CGD matrix structure
            sample_cgd = cgd_matrices[0]
            assert sample_cgd.ndim == 2, f"Batch {idx}: expected 2D CGD matrix, got {sample_cgd.ndim}D"
            assert sample_cgd.shape[0] >= 3, f"Batch {idx}: CGD matrix too small {sample_cgd.shape}"
            assert sample_cgd.shape[1] >= 3, f"Batch {idx}: CGD matrix too small {sample_cgd.shape}"
            
        print("Data validation passed!")
        
    def _load_to_memory(self):
        """Load all data to memory for faster access"""
        self.memory_images = []
        self.memory_cgd_matrices = []
        
        for file_idx in range(len(self.image_files)):
            images = np.load(self.image_files[file_idx], allow_pickle=True)
            cgd_matrices = np.load(self.cgd_files[file_idx], allow_pickle=True)
            
            self.memory_images.append(images.astype(np.float32))
            self.memory_cgd_matrices.append(cgd_matrices.astype(np.float32))
            
        print("Data loaded to memory!")
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            image: (1, H, W) tensor
            targets: (3,) tensor containing [Cgd_1_2, Cgd_1_3, Cgd_0_2]
        """
        file_idx, local_idx, channel_idx = self.index_map[idx]
        
        if self.load_to_memory:
            # Load from memory
            full_image = self.memory_images[file_idx][local_idx]  # Shape: (H, W, channels)
            cgd_matrix = self.memory_cgd_matrices[file_idx][local_idx]  # Shape: (num_dots, num_dots+1)
        else:
            # Load from disk
            images = np.load(self.image_files[file_idx], allow_pickle=True)
            cgd_matrices = np.load(self.cgd_files[file_idx], allow_pickle=True)
            
            full_image = images[local_idx].astype(np.float32)  # Shape: (H, W, channels)
            cgd_matrix = cgd_matrices[local_idx].astype(np.float32)  # Shape: (num_dots, num_dots+1)
        
        # Extract single channel from the multi-channel image
        image = full_image[:, :, channel_idx]  # Shape: (H, W)
        
        # Get targets for this specific channel
        num_dots = cgd_matrix.shape[0]
        targets = get_channel_targets(channel_idx, cgd_matrix, num_dots)
        
        # Convert to torch tensors
        image = torch.from_numpy(image).unsqueeze(0)  # Add channel dimension (1, H, W)
        targets = torch.from_numpy(targets)
        
        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)
            
        return image, targets


def create_data_loaders(
    data_dirs: Union[str, List[str]],
    batch_size: int = 32,
    val_split: float = 0.2,
    num_workers: int = 4,
    load_to_memory: bool = False,
    transform: Optional[callable] = None,
    random_state: int = 42
) -> Tuple[data.DataLoader, data.DataLoader]:
    """
    Create train and validation data loaders.
    
    Args:
        data_dirs: Path(s) to dataset directory
        batch_size: Batch size for data loaders
        val_split: Fraction of data to use for validation
        num_workers: Number of worker processes for data loading
        load_to_memory: Whether to load all data to memory
        transform: Optional image transforms
        random_state: Random seed for train/val split
        
    Returns:
        train_loader, val_loader
    """
    # Create full dataset
    full_dataset = CapacitanceDataset(
        data_dirs=data_dirs,
        transform=transform,
        load_to_memory=load_to_memory
    )
    
    # Split dataset
    total_size = len(full_dataset)
    val_size = int(val_split * total_size)
    train_size = total_size - val_size
    
    train_dataset, val_dataset = data.random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(random_state)
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


# Image transforms for data augmentation and normalization
class PercentileNormalize:
    """
    Normalize image data using percentile-based normalization to match env.py.
    
    Normalizes from 0 to 1 based on the middle 99% of data, 
    clipping the outer 0.5% to 0 and 1 on either end.
    """
    
    def __call__(self, tensor):
        """
        Args:
            tensor (torch.Tensor): Input tensor of shape (C, H, W) or (H, W)
            
        Returns:
            torch.Tensor: Normalized tensor with values in [0, 1]
        """
        # Convert to numpy for percentile calculation
        if isinstance(tensor, torch.Tensor):
            data = tensor.numpy()
        elif isinstance(tensor, np.ndarray):
            data = tensor
        else:
            raise TypeError("Input must be a torch.Tensor or np.ndarray")

        # Calculate percentiles for the middle 99% of data
        p_low = np.percentile(data, 0.5)   # 0.5th percentile  
        p_high = np.percentile(data, 99.5) # 99.5th percentile
        
        # Normalize to [0, 1] based on middle 99% range
        if p_high > p_low:
            normalized = (data - p_low) / (p_high - p_low)
        else:
            # Handle edge case where all values are the same
            normalized = np.zeros_like(data)
        
        # Clip to [0, 1] range (this clips the outer 0.5% on each end)
        normalized = np.clip(normalized, 0.0, 1.0)
        
        # Convert back to tensor and ensure float32
        return torch.from_numpy(normalized).float()


def get_transforms(normalize: bool = True):
    """Get image transforms for preprocessing that match env.py normalization"""
    import torchvision.transforms as transforms
    
    transform_list = []
    
    if normalize:
        # Use percentile-based normalization to match env.py
        # This normalizes based on middle 99% of data, clipping outer 0.5% on each end
        transform_list.append(PercentileNormalize())
    
    return transforms.Compose(transform_list) if transform_list else None


if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend for file output
    import matplotlib.pyplot as plt
    import random
    
    # Test the dataset
    data_dir = "/home/edn/rl-agent-for-qubit-array-tuning/src/swarm/qarray_dataset/dataset"
    
    print("Testing dataset loading...")
    dataset = CapacitanceDataset(data_dir, load_to_memory=False)
    
    # Test first few samples
    for i in range(3):
        image, targets = dataset[i]
        print(f"Sample {i}: image shape {image.shape}, targets {targets}")
    
    print("\nTesting data loaders...")
    train_loader, val_loader = create_data_loaders(
        data_dir, 
        batch_size=4, 
        load_to_memory=False
    )
    
    # Test one batch
    for batch_imgs, batch_targets in train_loader:
        print(f"Batch: images {batch_imgs.shape}, targets {batch_targets.shape}")
        print(f"Target range: {batch_targets.min():.3f} to {batch_targets.max():.3f}")
        break
    
    def save_random_images(loader, loader_name: str, num_images: int = 16):
        """Save random images from a dataloader to a PNG file"""
        # Collect images and targets from loader
        all_images = []
        all_targets = []
        
        # Sample random batches to get 9+ images
        sampled_batches = 0
        max_batches = min(5, len(loader))  # Limit to avoid loading too much
        
        for batch_imgs, batch_targets in loader:
            all_images.append(batch_imgs.cpu())
            all_targets.append(batch_targets.cpu())
            sampled_batches += 1
            if sampled_batches >= max_batches:
                break
        
        if not all_images:
            print(f"No images found in {loader_name} loader")
            return
            
        # Concatenate all collected images and targets
        all_images = torch.cat(all_images, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Randomly select 9 images
        total_available = len(all_images)
        num_to_plot = min(num_images, total_available)
        random_indices = random.sample(range(total_available), num_to_plot)
        
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        axes = axes.flatten()
        
        for plot_idx, img_idx in enumerate(random_indices):
            image = all_images[img_idx].squeeze().numpy()  # Remove channel dim and convert to numpy
            targets = all_targets[img_idx].numpy()
            
            # Plot the image
            im = axes[plot_idx].imshow(image, cmap='viridis', aspect='equal')
            axes[plot_idx].set_title(f'Sample {img_idx}\nTargets: [{targets[0]:.3f}, {targets[1]:.3f}, {targets[2]:.3f}]', 
                                   fontsize=8)
            axes[plot_idx].set_xlabel('Gate Voltage')
            axes[plot_idx].set_ylabel('Gate Voltage') 
            
            # Add colorbar
            plt.colorbar(im, ax=axes[plot_idx], shrink=0.8)
        
        for idx in range(num_to_plot, 16):
            axes[idx].axis('off')
            
        plt.suptitle(f'{loader_name.title()} Loader: {num_to_plot} Random Charge Stability Diagrams', 
                    fontsize=14)
        plt.tight_layout()
        
        # Save to file
        output_path = f'{loader_name}_random_images.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved {num_to_plot} random {loader_name} images to: {output_path}")
        plt.close()
    
    # Save random images from both loaders
    print("\nSaving random images from dataloaders...")
    save_random_images(train_loader, "train")
    save_random_images(val_loader, "validation") 