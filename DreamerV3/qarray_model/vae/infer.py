import torch
import os
import numpy as np

from vqvae import create_vqvae2_large
from utils import load_data

def main():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu", help="Device to use for inference")
    parser.add_argument("--checkpoint", type=str, default="./qarray_checkpoints/vqvae_epoch_2.pth", help="Path to the model checkpoint")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = create_vqvae2_large(image_size=128, in_channels=1).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint

    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace('module.', '') if k.startswith('module.') else k
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict)
    model.eval()

    num_images = 4

    this_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(this_dir, '../data')
    dataset = load_data(data_dir, num_samples=1)
    images = [d['image'] for d in dataset[:num_images]]
    images = np.array(images)
    images = torch.tensor(images).to(device)

    output = model(images)
    tq, bq = output['top_indices'], output['bottom_indices']
    #print(tq) # ranges from 0 to 1023

    recon = model.decode_from_indices(tq, bq)
    print(recon.shape)  # Check the shape of the reconstructed output
    
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