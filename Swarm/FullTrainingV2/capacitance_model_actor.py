"""
Ray actor for hosting capacitance models on individual GPUs.

This actor provides a serialization-safe interface to the capacitance model
by accepting numpy arrays and returning numpy arrays, avoiding the need to
serialize the PyTorch model or its threading components.
"""

import os
import sys
import torch
import numpy as np
import ray
from typing import Tuple
from pathlib import Path

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# Import the setup function from train.py

def setup_capacitance_model(checkpoint: str, batch_window_ms: int = 100, max_batch_size: int = 128):
    try:
        from ..CapacitanceModel import CapacitancePredictionModel
    except ImportError:
        # Fallback for direct execution - try absolute imports with path adjustment
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(current_dir)  # Swarm directory
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)
            from CapacitanceModel.CapacitancePrediction import CapacitancePredictionModel
        except ImportError:
            # Final fallback - individual module imports with path adjustment
            try:
                capacitance_dir = os.path.join(parent_dir, 'CapacitanceModel')
                if capacitance_dir not in sys.path:
                    sys.path.insert(0, capacitance_dir)
                from CapacitancePrediction import CapacitancePredictionModel
            except ImportError:
                raise RuntimeError("Could not initialise capacitance prediction model")

    # Load pre-trained weights - use absolute path from project root
    if 'SWARM_PROJECT_ROOT' in os.environ:
        # Ray distributed mode: use environment variable set by training script
        swarm_dir = os.environ['SWARM_PROJECT_ROOT']
    else:
        # Local development mode: find Swarm directory from current file
        swarm_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
    weights_path = os.path.join(
        swarm_dir,
        'CapacitanceModel', 
        checkpoint
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print("Warning: CUDA device not found, initialising capacitance model on CPU ...")
    
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found at: {weights_path}")

    ml_model = CapacitancePredictionModel()

    checkpoint = torch.load(weights_path, map_location=device)
            
    # Extract model state dict from checkpoint
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    ml_model.load_state_dict(state_dict)
    ml_model.to(device)
    ml_model.eval()  # Set to evaluation mode

    # Import and wrap the ML model for thread-safe batched inference
    try:
        # Try relative import from same directory
        from .batched_capacitance_wrapper import BatchedCapacitanceModelWrapper
    except ImportError:
        # Fallback import for direct execution
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        from batched_capacitance_wrapper import BatchedCapacitanceModelWrapper
    
    # Wrap the model for thread-safe concurrent access
    batched_model = BatchedCapacitanceModelWrapper(
        ml_model, 
        batch_window_ms=batch_window_ms,  # batching window
        max_batch_size=max_batch_size   # Max batch size before forcing processing
    )
    
    print(f"[CAPACITANCE DEBUG] Wrapped model with batching (window: 10ms, max_batch: 128)")
    return batched_model
    


@ray.remote(num_gpus=1)
class CapacitanceModelActor:
    """
    Ray actor that hosts a capacitance model on a specific GPU.
    
    This actor provides a thread-safe, serialization-friendly interface
    to the capacitance model by handling numpy array conversions.
    """
    
    def __init__(self, checkpoint_path: str, gpu_id: int, batch_window_ms: int = 100, max_batch_size: int = 128):
        """
        Initialize the capacitance model actor on a specific GPU.
        
        Args:
            checkpoint_path: Path to model checkpoint relative to Swarm directory
            gpu_id: GPU ID to use for this actor
            batch_window_ms: Batching window for concurrent requests
            max_batch_size: Maximum batch size before forcing processing
        """
        self.gpu_id = gpu_id
        self.checkpoint_path = checkpoint_path
        
        # Set CUDA device for this actor
        # Ray makes the allocated GPU visible as cuda:0 regardless of physical GPU ID
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            torch.cuda.set_device(0)
            print(f"[ACTOR GPU {gpu_id}] Initialized on {self.device} (Ray-allocated)")
        else:
            self.device = torch.device("cpu")
            print(f"[ACTOR GPU {gpu_id}] WARNING: CUDA not available, using CPU")
        
        # Initialize the model using the existing setup function
        self._setup_model(batch_window_ms, max_batch_size)
        
    def _setup_model(self, batch_window_ms: int, max_batch_size: int):
        """Setup the capacitance model with batching using existing setup function."""
        try:
            # Use the existing setup_capacitance_model function from train.py
            self.model = setup_capacitance_model(
                checkpoint=self.checkpoint_path,
                batch_window_ms=batch_window_ms,
                max_batch_size=max_batch_size
            )
            
            # Ensure model is on the correct device
            self.model.to(self.device)
            
            print(f"[ACTOR GPU {self.gpu_id}] Model loaded successfully with batching "
                  f"(window: {batch_window_ms}ms, max_batch: {max_batch_size})")
            
        except Exception as e:
            print(f"[ACTOR GPU {self.gpu_id}] Failed to setup model: {e}")
            raise
    
    def predict(self, input_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict capacitance values from input images.
        
        Args:
            input_data: Numpy array of shape (num_dots-1, 1, height, width)
            
        Returns:
            Tuple of (values, log_vars) as numpy arrays
        """
        try:
            # Convert numpy to tensor
            input_tensor = torch.from_numpy(input_data).float().to(self.device)
            
            # Run prediction
            with torch.no_grad():
                values, log_vars = self.model(input_tensor)
            
            # Convert back to numpy for serialization
            values_numpy = values.detach().cpu().numpy()
            log_vars_numpy = log_vars.detach().cpu().numpy()
            
            return values_numpy, log_vars_numpy
            
        except Exception as e:
            print(f"[ACTOR GPU {self.gpu_id}] Prediction error: {e}")
            raise
    
    def get_device_info(self) -> dict:
        """Get information about this actor's device."""
        info = {
            'gpu_id': self.gpu_id,
            'device': str(self.device),
            'cuda_available': torch.cuda.is_available(),
            'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'current_device': torch.cuda.current_device() if torch.cuda.is_available() else None,
            'process_id': os.getpid()
        }
        
        # Add physical GPU identification
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')
        info['cuda_visible_devices'] = cuda_visible
        
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            props = torch.cuda.get_device_properties(0)
            info.update({
                'gpu_memory_mb': props.total_memory // (1024**2),
                'gpu_name': props.name,
                'compute_capability': f"{props.major}.{props.minor}"
            })
        
        return info
    
    def cleanup(self):
        """Clean up resources."""
        if hasattr(self.model, 'close'):
            self.model.close()
        print(f"[ACTOR GPU {self.gpu_id}] Cleaned up successfully")


def create_capacitance_actor(checkpoint_path: str, gpu_id: int, **kwargs) -> ray.ObjectRef:
    """
    Factory function to create a capacitance model actor.
    
    Args:
        checkpoint_path: Path to model checkpoint
        gpu_id: GPU ID for this actor
        **kwargs: Additional arguments for the actor
        
    Returns:
        Ray object reference to the created actor
    """
    return CapacitanceModelActor.remote(checkpoint_path, gpu_id, **kwargs)