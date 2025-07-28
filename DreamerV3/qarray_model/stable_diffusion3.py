import numpy as np
import torch
import torch.nn as nn
import os
from diffusers import StableDiffusion3Pipeline, SD3Transformer2DModel
from diffusers.models import AutoencoderKL
from transformers import CLIPTokenizer, CLIPTextModel, T5Tokenizer, T5EncoderModel
from dotenv import load_dotenv
from PIL import Image

load_dotenv()


class VoltageProjectionLayer(nn.Module):
    def __init__(self, input_dim, output_dim, latent_dim=512):
        super().__init__()

        self.net = nn.Sequential(

        )

    def forward(self, x):
        return self.net(x)


def load_model():
    """
    Loads the Stable Diffusion 3 model components for fine-tuning.
    Returns only the trainable components (VAE and Transformer) without tokenizers.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    try:
        # Load the full pipeline first to get the components
        pipeline = StableDiffusion3Pipeline.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers",
            torch_dtype=dtype,
            cache_dir="./model_cache"
        )
        
        # Extract individual components for training
        # VAE (Variational Autoencoder) - for encoding/decoding images
        vae = pipeline.vae
        vae.requires_grad_(True)  # Enable gradients for training
        
        # Transformer (Main diffusion model) - core model to fine-tune
        transformer = pipeline.transformer
        transformer.requires_grad_(True)  # Enable gradients for training
        
        # Scheduler (for noise scheduling during training)
        scheduler = pipeline.scheduler
        
        # Move models to device
        vae = vae.to(device)
        transformer = transformer.to(device)
        
        print(f"Transformer parameters: {sum(p.numel() for p in transformer.parameters() if p.requires_grad):,}")
        
        return {
            'vae': vae,
            'transformer': transformer,
            'scheduler': scheduler,
            'device': device,
            'dtype': dtype,
            'text_encoder': pipeline.text_encoder.to(device, dtype=dtype),
            'text_encoder_2': pipeline.text_encoder_2.to(device, dtype=dtype),
            'text_encoder_3': pipeline.text_encoder_3.to(device, dtype=dtype),
            'tokenizer': pipeline.tokenizer,
            'tokenizer_2': pipeline.tokenizer_2,
            'tokenizer_3': pipeline.tokenizer_3
        }
        
    except Exception as e:
        print(f"Error loading model components: {e}")
        raise Exception("Could not load Stable Diffusion 3 model for training")


def setup_training(model_components, learning_rate=1e-5):
    """
    Setup optimizers and loss functions for training.
    """
    vae = model_components['vae']
    transformer = model_components['transformer']
    
    # Set up optimizers
    vae_optimizer = torch.optim.AdamW(vae.parameters(), lr=learning_rate)
    transformer_optimizer = torch.optim.AdamW(transformer.parameters(), lr=learning_rate)
    
    # Loss function (MSE for diffusion training)
    criterion = nn.MSELoss()
    
    return {
        'vae_optimizer': vae_optimizer,
        'transformer_optimizer': transformer_optimizer,
        'criterion': criterion
    }


def get_clip_embeds(tokenizer, text_encoder, prompt, device, dtype, max_length=77):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids.to(device)
    with torch.no_grad():
        prompt_embeds = text_encoder(text_input_ids, output_hidden_states=True)
        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds.hidden_states[-2]  # Use second-to-last layer
    prompt_embeds = prompt_embeds.to(dtype=dtype)
    pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=dtype)
    return prompt_embeds, pooled_prompt_embeds

def get_t5_embeds(tokenizer, text_encoder, prompt, device, dtype, max_length=256):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids.to(device)
    with torch.no_grad():
        prompt_embeds = text_encoder(text_input_ids)[0]
    prompt_embeds = prompt_embeds.to(dtype=dtype)
    return prompt_embeds

def infer(model_components, prompt, negative_prompt="", height=1024, width=1024, num_inference_steps=28, guidance_scale=7.0, generator=None):
    """
    Generate an image from a text prompt using Stable Diffusion 3 model components.

    Args:
        model_components (dict): Dictionary containing 'vae', 'transformer', 'scheduler', 
                                 'text_encoder', 'text_encoder_2', 'text_encoder_3', 
                                 'tokenizer', 'tokenizer_2', 'tokenizer_3', 'device', 'dtype'.
        prompt (str): The text prompt to guide image generation.
        negative_prompt (str, optional): Negative prompt for classifier-free guidance. Defaults to "".
        height (int, optional): Height of the generated image in pixels. Defaults to 1024.
        width (int, optional): Width of the generated image in pixels. Defaults to 1024.
        num_inference_steps (int, optional): Number of denoising steps. Defaults to 28.
        guidance_scale (float, optional): Guidance scale for CFG. Defaults to 7.0.
        generator (torch.Generator, optional): Random generator for reproducibility.

    Returns:
        PIL.Image.Image: The generated image.
    """
    # Extract model components
    vae = model_components['vae']
    transformer = model_components['transformer']
    scheduler = model_components['scheduler']
    text_encoder = model_components['text_encoder']
    text_encoder_2 = model_components['text_encoder_2']
    text_encoder_3 = model_components['text_encoder_3']
    tokenizer = model_components['tokenizer']
    tokenizer_2 = model_components['tokenizer_2']
    tokenizer_3 = model_components['tokenizer_3']
    device = model_components['device']
    dtype = model_components['dtype']

    # Ensure prompt is a string
    if isinstance(prompt, list):
        prompt = prompt[0]
    negative_prompt = negative_prompt or ""

    # Encode positive prompt
    prompt_embed, pooled_prompt_embed = get_clip_embeds(tokenizer, text_encoder, prompt, device, dtype)
    prompt_2_embed, pooled_prompt_2_embed = get_clip_embeds(tokenizer_2, text_encoder_2, prompt, device, dtype)
    t5_prompt_embed = get_t5_embeds(tokenizer_3, text_encoder_3, prompt, device, dtype)
    clip_prompt_embeds = torch.cat([prompt_embed, prompt_2_embed], dim=-1)
    pad_dim = t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1]
    clip_prompt_embeds = torch.nn.functional.pad(clip_prompt_embeds, (0, pad_dim))
    prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=1)  # Concatenate along sequence length
    pooled_prompt_embeds = torch.cat([pooled_prompt_embed, pooled_prompt_2_embed], dim=-1)
    
    print(prompt_embeds.shape)
    print(pooled_prompt_embeds.shape)

    # Encode negative prompt
    neg_prompt_embed, neg_pooled_prompt_embed = get_clip_embeds(tokenizer, text_encoder, negative_prompt, device, dtype)
    neg_prompt_2_embed, neg_pooled_prompt_2_embed = get_clip_embeds(tokenizer_2, text_encoder_2, negative_prompt, device, dtype)
    neg_t5_prompt_embed = get_t5_embeds(tokenizer_3, text_encoder_3, negative_prompt, device, dtype)
    neg_clip_prompt_embeds = torch.cat([neg_prompt_embed, neg_prompt_2_embed], dim=-1)
    pad_dim = neg_t5_prompt_embed.shape[-1] - neg_clip_prompt_embeds.shape[-1]
    neg_clip_prompt_embeds = torch.nn.functional.pad(neg_clip_prompt_embeds, (0, pad_dim))
    negative_prompt_embeds = torch.cat([neg_clip_prompt_embeds, neg_t5_prompt_embed], dim=1)
    negative_pooled_prompt_embeds = torch.cat([neg_pooled_prompt_embed, neg_pooled_prompt_2_embed], dim=-1)

    print(negative_prompt_embeds.shape)
    print(negative_pooled_prompt_embeds.shape)

    # Concatenate for classifier-free guidance
    prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
    pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)


    # need to project voltages up to these two vectors
    print(f"Prompt embeds shape: {prompt_embeds.shape}")
    print(f"Pooled prompt embeds shape: {pooled_prompt_embeds.shape}")

    print(prompt_embeds.max(), prompt_embeds.min())
    print(pooled_prompt_embeds.max(), pooled_prompt_embeds.min())

    # exit()

    # Prepare initial latents
    vae_scale_factor = 8  # Standard for Stable Diffusion
    latent_height = height // vae_scale_factor
    latent_width = width // vae_scale_factor
    latent_shape = (1, transformer.config.in_channels, latent_height, latent_width)
    latents = torch.randn(latent_shape, generator=generator, device=device, dtype=dtype)

    # Set up timesteps
    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps

    # Denoising loop
    for t in timesteps:
        # Expand latents for guidance
        latent_model_input = torch.cat([latents] * 2)
        timestep = t.expand(latent_model_input.shape[0])

        # Predict noise
        with torch.no_grad():
            noise_pred = transformer(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                return_dict=False,
            )[0]

        # Apply classifier-free guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # Update latents
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    # Decode latents to image
    with torch.no_grad():
        latents = (latents / vae.config.scaling_factor) + getattr(vae.config, 'shift_factor', 0)
        image = vae.decode(latents).sample

    # Convert to PIL Image
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()[0]
    image = (image * 255).astype(np.uint8)
    pil_image = Image.fromarray(image)

    return pil_image


def main():
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Train model on CSD data")
    parser.add_argument("--data_dir", type=str, default='./data', help="Directory containing training data")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs for training")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for training")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_components = load_model()

    # Test inference with text prompt
    text_prompt = "A futuristic quantum device with glowing circuits"
    text_prompt = "Voltages: [-1.18318284  0.85538369]"
    print(f"Generating image from prompt: '{text_prompt}'")

    import time
    start_time = time.time()
    
    image = infer(model_components, text_prompt)
    print(f"Inference time: {time.time() - start_time:.2f} seconds")
    image.save("generated_image.png")
    print("Image saved as 'generated_image.png'")
    
    exit()
    
    # Continue with training setup...
    training_components = setup_training(model_components, args.learning_rate)
    data = load_data(args.data_dir)
    print("Training setup complete!")


if __name__ == '__main__':
    main()