import torch
import torch.nn as nn
import numpy as np
from torchvision.utils import save_image
from tinyGAN.model import Generator, Discriminator, VGGLoss
from tinyGAN.utils import *
from utils import load_data

"""
training script for tiny GAN

note: we keep the noise fixed to ensure deterministic outputs
pass in y embeds rather than x label
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


def train(dataloader, G, D, voltage_projector, g_optimizer, d_optimizer, device, epochs=1, save_dir='./checkpoints'):
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
        
        for batch_idx, batch in enumerate(dataloader):
            voltages = batch['voltages'].to(device)  # Shape: (batch_size, voltage_dim)
            real_images = batch['image'].to(device)  # Shape: (batch_size, 3, height, width)
            
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
            
            # Print progress
            if batch_idx % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Batch [{batch_idx}/{len(dataloader)}], '
                      f'G_Loss: {g_loss.item():.4f}, D_Loss: {d_loss.item():.4f}, '
                      f'G_Adv: {g_adv_loss.item():.4f}, G_Recon: {g_recon_loss.item():.4f}, '
                      f'G_VGG: {g_vgg_loss.item():.4f}')
            
            # Save sample images periodically
            if batch_idx % 50 == 0:
                with torch.no_grad():
                    sample_voltage = voltages[0:1]  # Take first sample
                    sample_embedding = voltage_projector(sample_voltage)
                    sample_noise = fixed_noise[0:1]
                    sample_fake = G(sample_noise, sample_embedding)
                    sample_real = real_images[0:1]
                    
                    # Save comparison
                    comparison = torch.cat([sample_real, sample_fake], dim=0)
                    save_image(comparison, f'{save_dir}/training_samples_epoch{epoch}_batch{batch_idx}.png', 
                             nrow=2, normalize=True, range=(-1, 1))
        
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


def main():
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Tiny GAN finetuning script")
    parser.add_argument("--voltage_dim", type=int, default=2, help="Dimensionality of the voltage input")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs for training")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate for the optimizer")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    voltage_dim = args.voltage_dim
    lr = args.lr

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

    print('Training ...')
    train(dataloader, G, D, voltage_projector, g_optimizer, d_optimizer, device, epochs=args.epochs)
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