import torch.multiprocessing as mp
mp.set_start_method("spawn", force=True)

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader, TensorDataset

from vae.vqvae import create_vqvae2_large, VQVAELoss

def main():
    # Initialize accelerator
    accelerator = Accelerator()
    print(f"Process {accelerator.process_index}: Starting minimal test")
    
    # Create simple dummy dataset - no custom wrapper, no splitting
    dummy_data = torch.randn(100, 1, 128, 128)
    dummy_labels = torch.zeros(100)
    dataset = TensorDataset(dummy_data, dummy_labels)
    
    # Simple DataLoader
    train_loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
    val_loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
    
    if accelerator.is_main_process:
        print("Created dummy dataset and loaders")
        
        # Test one batch
        for batch in train_loader:
            print(f"Batch type: {type(batch)}")
            print(f"Batch shapes: {[x.shape for x in batch]}")
            break
    
    # Create minimal model and optimizer
    model = create_vqvae2_large(image_size=128, in_channels=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    if accelerator.is_main_process:
        print("Calling accelerator.prepare...")
    
    # Prepare with accelerator
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )
    
    if accelerator.is_main_process:
        print("SUCCESS: accelerator.prepare completed!")
    
    # Test one forward pass
    if accelerator.is_main_process:
        print("Testing forward pass...")
        for batch in train_loader:
            images = batch[0]  # TensorDataset returns tuples
            print(f"Forward pass with shape: {images.shape}")
            outputs = model(images)
            print(f"Output shape: {outputs[0].shape}")  # VQVAE returns tuple
            break
    
    print(f"Process {accelerator.process_index}: Done!")

if __name__ == "__main__":
    main()