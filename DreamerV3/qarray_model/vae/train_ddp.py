import os
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import datasets, transforms
import signal
import sys
from typing import List, Dict
import subprocess
from tqdm import tqdm
import json

from vqvae import create_vqvae2_large, VQVAELoss
from utils import load_data


def setup(rank: int, world_size: int) -> None:
    """
    Initialize the distributed process group for multi-GPU training.

    Args:
        rank (int): The rank of the current process.
        world_size (int): The total number of processes.

    """
    os.environ['MASTER_ADDR'] = '127.0.0.1' # 'oums-dlgpu1.materials.ox.ac.uk'
    os.environ['MASTER_PORT'] = '50001'  # or any free port number
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup() -> None:
    """
    Clean up the distributed process group.
    """
    if dist.is_initialized():
        dist.destroy_process_group()
        print('Process group destroyed')
    else:
        print('No group found')


def check_gpu_memory(min_memory_gb: float) -> List[int]:
    """
    Check available GPU memory and return GPUs with at least min_memory_gb free memory.

    Args:
        min_memory_gb (float): Minimum required free memory in GB.

    Returns:
        List[int]: List of GPU indices with sufficient free memory.
    """
    available_gpus = []
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader'],
                            stdout=subprocess.PIPE)
    memory_free = result.stdout.decode('utf-8').strip().split('\n')
    memory_free = [int(x) for x in memory_free]
    print("GPU Memory Availability:")
    for i, mem in enumerate(memory_free):
        mem_gb = mem / 1024  # Convert MB to GB
        print(f"GPU {i}: {mem_gb:.2f} GB free")
        if mem_gb >= min_memory_gb:
            available_gpus.append(i)
    return available_gpus


def signal_handler(sig, frame) -> None:
    """
    Handle termination signals to clean up and exit gracefully.

    Args:
        sig: Signal number.
        frame: Current stack frame.
    """
    print('Received signal to terminate. Cleaning up...')
    cleanup()
    sys.exit(0)


def GPU_setup(min_memory_gb: float, use_gpus: List[int]) -> int:
    """
    Set up available GPUs based on minimum required free memory.

    Args:
        min_memory_gb (float): Minimum required free memory in GB per GPU.

    Returns:
        int: Number of GPUs available (world_size).
    """
    available_gpus = check_gpu_memory(min_memory_gb)
    available_gpus = [gpu for gpu in available_gpus if gpu in use_gpus]

    world_size = len(available_gpus)

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, available_gpus))
    if not available_gpus:
        print(f"No GPUs with at least {min_memory_gb} GB of memory available.")
    else:
        gpustring = ', '.join(str(gpu) for gpu in available_gpus)
        print(f'Using {world_size} GPUs: {gpustring}')

    return world_size


def build_model(image_size: int, in_channels: int, rank: int) -> torch.nn.Module:
    """
    Build the VQ-VAE model.

    Args:
        image_size (int): Size of the input images.
        in_channels (int): Number of input channels.
        rank (int): Rank of the current process.

    Returns:
        torch.nn.Module: The VQ-VAE model.
    """
    model = create_vqvae2_large(image_size=image_size, in_channels=in_channels)
    model = model.to(rank)
    return model


def load_dataset(data_dir: str, num_samples: int) -> (TensorDataset, TensorDataset):
    """
    Load the dataset from the specified directory.
    Args:
        data_dir (str): Directory to load the dataset from.

    Returns:
        TensorDataset: Training and testing datasets.
    """
    dataset = load_data(data_dir, num_samples=num_samples)
    train_size = int(0.99 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_images, train_labels = zip(*[(torch.tensor(item['image']), torch.tensor(item['voltages'])) for item in train_dataset])
    test_images, test_labels = zip(*[(torch.tensor(item['image']), torch.tensor(item['voltages'])) for item in test_dataset])
    train_images = torch.stack(train_images)
    train_labels = torch.stack(train_labels)
    test_images = torch.stack(test_images)
    test_labels = torch.stack(test_labels)

    train_dataset = TensorDataset(train_images, train_labels)
    test_dataset = TensorDataset(test_images, test_labels)
    return train_dataset, test_dataset


def save_test_images(model: torch.nn.Module, test_loader: DataLoader, save_dir: str) -> None:
    """
    Save reconstructed images from the test set.

    Args:
        model (torch.nn.Module): The trained VQ-VAE model.
        test_loader (DataLoader): DataLoader for the test set.
        save_dir (str): Directory to save the images.
    """
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving test images to {save_dir}")

    model.eval()
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(test_loader):
            images = images.to(rank).float()
            outputs = model(images)

        for i in range(len(images)):
            img = outputs[i].cpu().numpy().transpose(1, 2, 0)  # Convert to HWC format
            img_path = os.path.join(save_dir, f"test_image_{batch_idx * len(images) + i}.png")
            torchvision.utils.save_image(img, img_path)


def train(rank: int, world_size: int, train: TensorDataset, test: TensorDataset, config: Dict) -> None:
    """
    Main training function to be executed by each process.
    """
    setup(rank, world_size)
    print(f"Rank {rank} of {world_size} is starting training...")

    batch_size = config['batch_size']
    epochs = config['epochs']
    save_dir = config['save_dir']
    use_wandb = config['use_wandb']
    noise_weight = config['noise_weight']
    consistency_weight = config['consistency_weight']

    train_sampler = DistributedSampler(train, num_replicas=world_size, rank=rank)
    test_sampler = DistributedSampler(test, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train, batch_size=batch_size, sampler=train_sampler)
    test_loader = DataLoader(test, batch_size=batch_size, sampler=test_sampler)

    model = build_model(**config['model_params'], rank=rank)
    ddp_model = DDP(model, device_ids=[rank])#, find_unused_parameters=True)
    ddp_model.train()

    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=config['lr'])
    criterion = VQVAELoss(consistency_weight=consistency_weight).to(rank)

    if rank == 0 and use_wandb:
        import wandb
        wandb.init(project="vqvae-qarray", config=config)

    for epoch in range(epochs):
        epoch_loss = 0.0

        if rank == 0:
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)
        else:
            progress_bar = train_loader

        ddp_model.train()

        for batch_idx, (images, voltages) in enumerate(progress_bar):
            images = images.to(rank).float()
            voltages = voltages.to(rank).float()

            noise = torch.randn_like(images) * noise_weight
            outputs = ddp_model(images + noise)
            loss_dict = criterion(outputs, images, voltages)
            loss = loss_dict['total_loss']

            reconstruction_loss = loss_dict['reconstruction_loss']
            vq_loss = loss_dict['vq_loss']
            consistency_loss = loss_dict['consistency_loss']
            vq_loss_top = loss_dict['vq_loss_top']
            vq_loss_bottom = loss_dict['vq_loss_bottom']

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            dist.all_reduce(loss)
            dist.all_reduce(reconstruction_loss)
            dist.all_reduce(vq_loss)
            dist.all_reduce(consistency_loss)
            dist.all_reduce(vq_loss_top)
            dist.all_reduce(vq_loss_bottom)

            loss /= world_size
            consistency_loss /= world_size
            reconstruction_loss /= world_size
            vq_loss /= world_size

            epoch_loss += loss.item()

            if rank == 0:
                progress_bar.set_postfix({
                    "Batch Loss": f"{loss.item():.4f}",
                    "Avg Loss": f"{epoch_loss / (batch_idx + 1):.4f}"
                })

                if use_wandb:
                    wandb.log({
                        "train/epoch": epoch + 1,
                        "train/total_loss": loss.item(),
                        "train/reconstruction_loss": reconstruction_loss.item(),
                        "train/vq_loss": vq_loss.item(),
                        "train/consistency_loss": consistency_loss.item(),
                    })

        ddp_model.eval()

        for images, voltages in test_loader:
            images = images.to(rank).float()
            voltages = voltages.to(rank).float()

            noise = torch.randn_like(images) * noise_weight
            outputs = ddp_model(images + noise)
            loss_dict = criterion(outputs, images, voltages)
            loss = loss_dict['total_loss']

            reconstruction_loss = loss_dict['reconstruction_loss']
            vq_loss = loss_dict['vq_loss']
            consistency_loss = loss_dict['consistency_loss']
            vq_loss_top = loss_dict['vq_loss_top']
            vq_loss_bottom = loss_dict['vq_loss_bottom']

            dist.all_reduce(loss)
            dist.all_reduce(reconstruction_loss)
            dist.all_reduce(vq_loss)
            dist.all_reduce(consistency_loss)
            dist.all_reduce(vq_loss_top)
            dist.all_reduce(vq_loss_bottom)

            loss /= world_size
            consistency_loss /= world_size
            reconstruction_loss /= world_size
            vq_loss /= world_size

            if rank == 0 and use_wandb:
                wandb.log({
                    "eval/total_loss": loss.item(),
                    "eval/reconstruction_loss": reconstruction_loss.item(),
                    "eval/vq_loss": vq_loss.item(),
                    "eval/consistency_loss": consistency_loss.item(),
                })


        if rank == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_loader):.4f}")
            save_dict = {
                'model': ddp_model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(save_dict, f"{save_dir}/vqvae_epoch_{epoch + 1}.pth")
    
    print(f"Rank {rank} finished training.")
    cleanup()


def train2(rank: int, world_size: int, config: Dict) -> None:
    setup(rank, world_size)
    print(f"Rank {rank} of {world_size} is starting training...")

    # Your training code here
    # ...
    model = build_model(**config['model_params'], rank=rank)
    print("Wrapping model with DDP...")
    ddp_model = DDP(model, device_ids=[rank])
    print("DDP wrapped.")
    # ddp_model = model
    # ddp_model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    ddp_model.train()

    print(f"Rank {rank} finished training.")
    cleanup()


def main() -> None:
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Train VQ-VAE on qarray data")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs for training")
    parser.add_argument("--gpus", type=int, nargs='+', default=[0,1,2,3,4,5,6,7], help="List of GPU indices to use")
    parser.add_argument("--min_memory_gb", type=float, default=14.0, help="Minimum required free memory in GB per GPU")
    parser.add_argument("--datasize", type=int, default=2000, help="Size of the dataset")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for the optimizer")
    parser.add_argument("--white_noise", type=float, default=0.1, help="Weight of white noise added to images")
    parser.add_argument("--consistency_weight", type=float, default=0.1, help="Weight of consistency loss")
    parser.add_argument("--wandb", action='store_true', help="Use wandb for logging")
    parser.add_argument("--data_dir", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data'), help="Directory to load data from")
    parser.add_argument("--save_dir", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'qarray_checkpoints'), help="Directory to save models")
    args = parser.parse_args()

    train_dataset, test_dataset = load_dataset(args.data_dir, args.datasize)

    use_gpus = args.gpus
    min_memory_gb = args.min_memory_gb
    world_size = GPU_setup(min_memory_gb, use_gpus)

    if world_size < 1:
        print("No suitable GPUs found. Exiting.")
        sys.exit(1)

    config = {
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'lr': args.lr,
        'save_dir': args.save_dir,
        'use_wandb': args.wandb,
        'noise_weight': args.white_noise,
        'consistency_weight': args.consistency_weight,
        'model_params': {
            'image_size': 128,
            'in_channels': 1,
        }
    }

    os.makedirs(config['save_dir'], exist_ok=True)

    # save config to save_dir as a json file
    with open(os.path.join(os.path.dirname(__file__), config['save_dir'], 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    print("Training ...")
    torch.multiprocessing.spawn(train, args=(world_size, train_dataset, test_dataset, config), nprocs=world_size, join=True)
    

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    main()