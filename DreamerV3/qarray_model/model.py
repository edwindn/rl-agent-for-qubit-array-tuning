import numpy as np
import torch
import os
from diffusers import StableDiffusion3Pipeline
from dotenv import load_dotenv

# required packages:
# import accelerate
# import protobuf
# import sentencepiece

load_dotenv()

def load_model():
    """
    Loads the Stable Diffusion 3 model from local files and returns a callable pipeline for inference.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # First, try to download the complete model structure
    print("Downloading Stable Diffusion 3 model...")
    try:
        pipeline = StableDiffusion3Pipeline.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers",
            torch_dtype=torch.float16,
            cache_dir="./model_cache"  # Cache locally for future use
        )
        pipeline = pipeline.to(device)
        print("Model loaded successfully from Hugging Face.")
        return pipeline
        
    except Exception as e:
        print(f"Error loading model from Hugging Face: {e}")
        
        # Alternative: try with different model variant
        try:
            print("Trying alternative model...")
            pipeline = StableDiffusion3Pipeline.from_pretrained(
                "stabilityai/stable-diffusion-3-medium",
                torch_dtype=torch.float16,
                cache_dir="./model_cache"
            )
            pipeline = pipeline.to(device)
            print("Alternative model loaded successfully.")
            return pipeline
            
        except Exception as e2:
            print(f"Error with alternative model: {e2}")
            raise Exception("Could not load any Stable Diffusion 3 model")


def load_data():
    pass


def main():
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Train model on CSD data")
    parser.add_argument("--data_dir", type=str, default='./data', help="Directory containing training data")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs for training")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading model...")
    model = load_model()
    model.to(device)
    print(f"Model device: {model.device}")

    # Example inference
    prompt = "A futuristic cityscape at sunset"
    image = model(prompt).images[0]
    image.save("output.png")
    print("Inference completed. Image saved as output.png.")


if __name__ == '__main__':
    main()