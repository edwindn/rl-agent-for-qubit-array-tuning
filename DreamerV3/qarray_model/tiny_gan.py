import torch
import torch.nn as nn
import numpy as np
from torchvision.utils import save_image
import wandb
from dotenv import load_dotenv
from tqdm import tqdm
import random
import matplotlib.pyplot as plt

from tinyGAN.model import Generator, Discriminator, VGGLoss
from tinyGAN.utils import *
from utils import load_data

load_dotenv()

"""
training script for tiny GAN

note: we keep the noise fixed to ensure deterministic outputs
pass in y embeds rather than x label

todo:
better handling of grayscale (viridis or better reshape to c=1)
"""

class VoltageProjector(nn.Module):
    def __init__(self, in_dim, out_dim, latent_dim=256):
        super(VoltageProjector, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(in_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, out_dim)
        )

    def forward(self, z):
        return self.net(z)


def infer(G, label, noise, z_dim=128):
    G.eval()
    with torch.no_grad():
        out = G(noise, label).detach().cpu()
    return denorm(out)


def train(dataloader, G, D, voltage_projector, g_optimizer, d_optimizer, device, use_wandb, epochs=1, save_dir='./checkpoints'):
    os.makedirs(save_dir, exist_ok=True)

    # Loss functions
    adversarial_loss = nn.BCEWithLogitsLoss()
    reconstruction_loss = nn.L1Loss()
    vgg_loss = VGGLoss()
    
    # Fixed noise for consistent generation during training
    fixed_noise = torch.load('tinygan_noise.pt').to(device)
    
    # Training loop
    G.train()
    voltage_projector.train()
    D.train()
    
    for epoch in range(epochs):
        total_g_loss = 0
        total_d_loss = 0
        
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_idx, batch in progress_bar:
            # Convert voltages to torch.float
            voltages = batch['voltages'].to(device).float()  # Ensure dtype is torch.float
            
            # Convert real_images to torch.float and normalize to [0, 1]
            real_images = batch['image'].to(device).float() / 255.0  # Normalize uint8 to [0, 1]
            real_images = real_images.permute(0, 3, 1, 2)  # Convert to (batch_size, channels, height, width)
            
            # Ensure real_images has 3 channels
            if real_images.shape[1] != 3:
                if real_images.shape[1] == 1:  # Grayscale images
                    real_images = real_images.repeat(1, 3, 1, 1)  # Convert to 3 channels
                else:
                    raise ValueError(f"Unexpected number of channels in real_images: {real_images.shape[1]}")
            
            batch_size = voltages.size(0)
            
            # Expand fixed noise to match batch size
            if batch_size != fixed_noise.size(0):
                noise = fixed_noise.expand(batch_size, -1)
            else:
                noise = fixed_noise
            
            # Create labels for adversarial loss
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)
            
            # Generate fake class labels (using random labels from pretrained classes)
            fake_class_labels = torch.randint(0, 398, (batch_size,)).to(device)
            
            # ================== Train Discriminator ================== #
            d_optimizer.zero_grad()
            
            # Project voltages to embedding space
            voltage_embeddings = voltage_projector(voltages)
            
            # Generate fake images
            fake_images = G(noise, voltage_embeddings)
            
            # Discriminator outputs for real images
            d_real_out, _ = D(real_images, fake_class_labels)
            d_real_loss = adversarial_loss(d_real_out, real_labels)
            
            # Discriminator outputs for fake images (detach to avoid backprop through generator)
            d_fake_out, _ = D(fake_images.detach(), fake_class_labels)
            d_fake_loss = adversarial_loss(d_fake_out, fake_labels)
            
            # Total discriminator loss
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            d_optimizer.step()
            
            # ================== Train Generator & Voltage Projector ================== #
            g_optimizer.zero_grad()
            
            # Generate fake images again (without detach for backprop)
            voltage_embeddings = voltage_projector(voltages)
            fake_images = G(noise, voltage_embeddings)
            
            # Adversarial loss for generator
            d_fake_out, _ = D(fake_images, fake_class_labels)
            g_adv_loss = adversarial_loss(d_fake_out, real_labels)
            
            # Reconstruction loss (L1 loss between fake and real images)
            g_recon_loss = reconstruction_loss(fake_images, real_images)
            
            # Perceptual loss using VGG features
            g_vgg_loss = vgg_loss(fake_images, real_images)
            
            # Total generator loss (weighted combination)
            g_loss = g_adv_loss + 10.0 * g_recon_loss + 1.0 * g_vgg_loss
            
            g_loss.backward()
            g_optimizer.step()
            
            # Accumulate losses
            total_g_loss += g_loss.item()
            total_d_loss += d_loss.item()
            
            # Update tqdm progress bar
            progress_bar.set_postfix({
                'G_Loss': g_loss.item(),
                'D_Loss': d_loss.item(),
            })
            
            if use_wandb:
                wandb.log({
                    'G_Loss': g_loss.item(),
                    'D_Loss': d_loss.item(),
                })

        # Save test run of generated images at the end of the epoch
        with torch.no_grad():
            sample_voltage = voltages[0:1]  # Take first sample
            sample_embedding = voltage_projector(sample_voltage)
            sample_noise = fixed_noise[0:1].expand(1, 128)  # Ensure correct shape
            sample_fake = G(sample_noise, sample_embedding)
            sample_real = real_images[0:1]  # Take first sample

            # Remove batch dimension for visualization
            sample_real = sample_real.squeeze(0)  # Shape: (channels, height, width)
            sample_fake = sample_fake.squeeze(0)  # Shape: (channels, height, width)

            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(sample_real.permute(1, 2, 0).cpu().numpy())
            plt.title("Real Image")
            plt.axis('off')
            plt.subplot(1, 2, 2)
            plt.imshow(sample_fake.permute(1, 2, 0).cpu().numpy())
            plt.title("Fake Image")
            plt.axis('off')
            plt.savefig(f'{save_dir}/test_run_epoch{epoch+1}.png')
            plt.close()
            
        # Print epoch summary
        avg_g_loss = total_g_loss / len(dataloader)
        avg_d_loss = total_d_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{epochs}] - Average G_Loss: {avg_g_loss:.4f}, Average D_Loss: {avg_d_loss:.4f}')
        
        # Save model checkpoints
        if (epoch + 1) % 5 == 0:
            torch.save({
                'generator': G.state_dict(),
                'voltage_projector': voltage_projector.state_dict(),
                'discriminator': D.state_dict(),
                'g_optimizer': g_optimizer.state_dict(),
                'd_optimizer': d_optimizer.state_dict(),
            }, f'{save_dir}/checkpoint_epoch_{epoch+1}.pth')
            print(f'Saved checkpoint for epoch {epoch+1}')
    
    print("Training completed!")
    
    # Save final models
    torch.save({
        'generator': G.state_dict(),
        'voltage_projector': voltage_projector.state_dict(),
        'discriminator': D.state_dict(),
    }, f'{save_dir}/final_models.pth')
    print("Final models saved!")


def visualise_images(dataset, device):
    """
    Visualize 10 random images from the dataset after processing them
    the same way as in the training loop.
    """
    # Select 10 random samples from the dataset
    random_samples = random.sample(dataset, 10)

    # Process the images
    processed_images = []
    voltages_list = []
    for sample in random_samples:
        voltages = sample['voltages']
        image = sample['image']

        # Convert image to torch.float and normalize to [0, 1]
        image = torch.tensor(image).to(device).float() / 255.0  # Normalize uint8 to [0, 1]
        image = image.permute(2, 0, 1)  # Convert to (channels, height, width)

        processed_images.append(image)
        voltages_list.append(voltages)

    # Plot the images
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    for i, (image, voltages) in enumerate(zip(processed_images, voltages_list)):
        # Convert image back to numpy for plotting
        image_np = image.permute(1, 2, 0).cpu().numpy()  # Convert to (height, width, channels)
        axes[i].imshow(image_np)
        axes[i].set_title(f"Voltages: {voltages}")
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig('random_samples.png')


def main():
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Tiny GAN finetuning script")
    parser.add_argument("--voltage_dim", type=int, default=2, help="Dimensionality of the voltage input")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs for training")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate for the optimizer")
    parser.add_argument("--use_wandb", action='store_true', default=False, help="Use Weights & Biases for logging")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    voltage_dim = args.voltage_dim
    lr = args.lr
    use_wandb = args.use_wandb

    noise = torch.load('tinygan_noise.pt').to(device)
    z_dim = 128
    G = Generator(image_size=128, conv_dim=32, z_dim=z_dim, c_dim=128, repeat_num=5)
    D = Discriminator(image_size=128, conv_dim=32, repeat_num=5)
    voltage_projector = VoltageProjector(in_dim=voltage_dim, out_dim=z_dim).to(device)

    g_optimizer = torch.optim.Adam(
        list(G.parameters()) + list(voltage_projector.parameters()), 
        lr=lr, betas=(0.5, 0.999)
    )
    d_optimizer = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

    restore_model(30, 'gan/models', G, D)
    G.to(device)
    D.to(device)

    dataset = load_data('./data')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    if use_wandb:
        wandb.init(
            project="qarray_tiny_gan",
        )

    print('Training ...')
    train(dataloader, G, D, voltage_projector, g_optimizer, d_optimizer, device, use_wandb, epochs=args.epochs)
    print('Finished training')
    exit()

    # noise = torch.FloatTensor(truncated_normal(z_dim)).to(device)
    # n_row = 1
    # label = np.random.choice(398, n_row, replace=False)
    # label = [100]
    # label = torch.tensor(label).to(device)
    voltages = [-6.254, 0.5345]
    voltages = torch.tensor(voltages).to(device).unsqueeze(0)

    emb = voltage_projector(voltages)
    image = infer(G, emb, noise)
    save_image(image, 'demo.png', nrow=1)


if __name__ == "__main__":
    main()