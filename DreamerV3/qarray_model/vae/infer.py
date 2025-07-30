import torch

from vqvae import create_vqvae2_large
from mnist import load_mnist

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_vqvae2_large(image_size=28, in_channels=1).to(device)
    model.load_state_dict(torch.load("vqvae_mnist.pth"))
    model.eval()

    _, eval_data = load_mnist(batch_size=8)
    sample = next(iter(eval_data))
    images, _ = sample
    images = images.to(device)
    output = model(images)
    tq, bq = output['top_indices'], output['bottom_indices']
    #print(tq) # ranges from 0 to 1023

    recon = model.decode_from_indices(tq, bq)
    print(recon.shape)  # Check the shape of the reconstructed output
    
    # plot the 8 input images next to their reconstructions
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 8, figsize=(16, 4))
    recon = recon.detach().cpu().numpy()
    images = images.cpu().numpy()

    for i in range(8):
        axs[0, i].imshow(images[i].squeeze(), cmap='gray')
        axs[0, i].axis('off')
        axs[1, i].imshow(recon[i].squeeze(), cmap='gray')
        axs[1, i].axis('off')
    axs[0, 0].set_title("Input Images")
    axs[1, 0].set_title("Reconstructed Images")
    plt.tight_layout()
    plt.savefig("vqvae_reconstruction.png")



if __name__ == "__main__":
    main()