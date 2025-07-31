import torch
import torch.nn.functional as F

from vqvae_mnist import create_vqvae2_large
from train_mnist import load_mnist
from latent_predictor_mnist import PixelCNN

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_vqvae2_large(image_size=28, in_channels=1).to(device)
    model.load_state_dict(torch.load("vqvae_mnist.pth"))
    model.eval()

    cnn = PixelCNN(num_classes=10, small_dim=7, large_dim=14).to(device)
    cnn.load_state_dict(torch.load("pixel_cnn_mnist.pth"))
    cnn.eval()

    # _, eval_data = load_mnist(batch_size=8)
    # sample = next(iter(eval_data))
    # images, labels = sample
    # images = images.to(device)
    # labels = labels.to(device)
    # output = model(images, labels)
    # tq, bq = output['top_indices'], output['bottom_indices']
    labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).to(device)
    labels_one_hot = F.one_hot(labels, num_classes=10).to(device)
    tq, bq = cnn(labels_one_hot.float())
    tq, bq = tq * 1023, bq * 1023
    tq, bq = tq.long().squeeze(), bq.long().squeeze()

    print(labels.shape)
    print(tq.shape, bq.shape)

    recon = model.decode_from_indices(tq, bq, labels_one_hot)
    print(recon.shape)  # Check the shape of the reconstructed output
    
    # plot the 8 input images next to their reconstructions
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 8, figsize=(16, 4))
    recon = recon.detach().cpu().numpy()

    images = torch.randn(recon.shape)
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