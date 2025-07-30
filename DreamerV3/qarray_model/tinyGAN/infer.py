import torch
import matplotlib.pyplot as plt

from tiny_gan import VoltageProjector, infer
from model import Generator, Discriminator, VGGLoss
from utils import *


def main():
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Run inference on the tinyGAN model.")
    parser.add_argument("--checkpoint", type=str, default='./checkpoints/final_models.pth')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    checkpoint = torch.load(args.checkpoint, map_location=device) # generator, voltage_projector, discriminator

    noise = torch.load('tinygan_noise.pt').to(device)
    voltage_dim = 2
    z_dim = 128
    G = Generator(image_size=128, conv_dim=32, z_dim=z_dim, c_dim=128, repeat_num=5).to(device)
    D = Discriminator(image_size=128, conv_dim=32, repeat_num=5).to(device)
    voltage_projector = VoltageProjector(in_dim=voltage_dim, out_dim=z_dim).to(device)

    G.load_state_dict(checkpoint['generator'])
    D.load_state_dict(checkpoint['discriminator'])
    voltage_projector.load_state_dict(checkpoint['voltage_projector'])

    voltages = [-6.254, 0.5345]
    voltages = torch.tensor(voltages).to(device).unsqueeze(0)
    emb = voltage_projector(voltages)
    image = infer(G, emb, noise)
    image = image.cpu().detach().numpy().squeeze().transpose(1, 2, 0)
    plt.imshow(image)
    plt.axis('off')
    plt.savefig('inference_result.png', bbox_inches='tight', pad_inches=0.1)

if __name__ == "__main__":
    main()