import torch
import os
import numpy as np
import sys

from vqvae import create_vqvae2_large
from utils import load_data
from qarray_env import QuantumDeviceEnv


def main():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device to use for inference")
    parser.add_argument("--checkpoint", type=str, default="./qarray_checkpoints/vqvae_epoch_5.pth", help="Path to the model checkpoint")
    parser.add_argument("--num_samples", type=int, default=12, help="Number of samples to generate")
    parser.add_argument("--data_type", type=str, default="unseen", choices=["unseen", "train"], help="Whether to infer on train or unseen data")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Inferring on {args.data_type} samples")

    model = create_vqvae2_large(image_size=128, in_channels=1).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint

    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace('module.', '') if k.startswith('module.') else k
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict)
    model.eval()

    num_images = args.num_samples

    max_cuda_batch_size = 4

    # --- test on new data ---
    if args.data_type == "unseen":
        env = QuantumDeviceEnv(render_mode='rgb_array')
        images = []
        env.reset()
        for _ in range(num_images):
            action = env.action_space.sample()
            state, *_ = env.step(action)
            image = state['image']
            image = np.array(image, dtype=np.float32) / 255.0
            images.append(image)
        images = torch.tensor(images).permute(0, 3, 1, 2)
        print(images.shape)


    # --- test on training data ---
    elif args.data_type == "train":
        this_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(this_dir, '../data')
        dataset = load_data(data_dir, num_samples=1)
        np.random.shuffle(dataset)
        images = [d['image'] for d in dataset[:num_images]]
        images = np.array(images)
        images = torch.tensor(images).to(device)

    images = images + torch.randn_like(images) * 0.025  # Adding white noise

    if num_images > max_cuda_batch_size:
        recons = []
        images = images.cpu()
        
        with torch.no_grad():  # Disable gradient computation
            for i in range(0, num_images, max_cuda_batch_size):
                # Clear cache before each batch
                torch.cuda.empty_cache()
                
                batch = images[i:i + max_cuda_batch_size].to(device).float()
                output = model(batch)
                tq, bq = output['top_indices'], output['bottom_indices']
                recon_batch = model.decode_from_indices(tq, bq)
                
                # Move to CPU immediately and delete GPU tensors
                recons.append(recon_batch.cpu())
                del batch, output, tq, bq, recon_batch
                torch.cuda.empty_cache()
                
        recon = torch.cat(recons, dim=0)
    else:
        with torch.no_grad():
            images = images.to(device).float()
            output = model(images)
            tq, bq = output['top_indices'], output['bottom_indices']
            recon = model.decode_from_indices(tq, bq)
    print(recon.shape)
    
    # plot the input images next to their reconstructions
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, num_images, figsize=(16, 4))
    recon = recon.detach().cpu().numpy()
    images = images.cpu().numpy()
    for i in range(num_images):
        axs[0, i].imshow(images[i].squeeze(), cmap='viridis')
        axs[0, i].axis('off')
        axs[1, i].imshow(recon[i].squeeze(), cmap='viridis')
        axs[1, i].axis('off')
    axs[0, 0].set_title("Input Images")
    axs[1, 0].set_title("Reconstructed Images")
    plt.tight_layout()
    plt.savefig("vqvae_reconstruction.png")



if __name__ == "__main__":
    main()