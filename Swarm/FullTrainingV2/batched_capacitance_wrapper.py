"""
Thread-safe batched wrapper for the capacitance model.

This wrapper maintains the exact same interface as the original PyTorch model
but internally queues requests and processes them in batches for thread safety
and improved GPU efficiency.
"""

import torch
import threading
import time
import uuid
from collections import deque
from typing import Dict, Tuple, Any, Optional
import numpy as np


class BatchedCapacitanceModelWrapper:
    """
    A thread-safe wrapper around the capacitance model that automatically batches
    concurrent forward passes while maintaining the exact same interface.
    Do NOT use this class to train the model, this is meant for inference only.
    
    Usage:
        # Wrap the original model
        wrapped_model = BatchedCapacitanceModelWrapper(original_model, batch_window_ms=10)
        
        # Use exactly like the original model
        values, log_vars = wrapped_model(input_tensor)
    """
    
    def __init__(self, model: torch.nn.Module, batch_window_ms: int = 10, max_batch_size: int = 128):
        """
        Initialize the batched wrapper.
        
        Args:
            model: The original PyTorch capacitance model
            batch_window_ms: Time window to collect requests for batching (milliseconds)
            max_batch_size: Maximum batch size before forcing processing
        """
        self.model = model
        self.model.eval()

        self.batch_window_ms = batch_window_ms
        self.max_batch_size = max_batch_size
        
        # Request queue and processing
        self.request_queue = deque()
        self.pending_results: Dict[str, threading.Event] = {}
        self.results: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        
        # Thread synchronization
        self.queue_lock = threading.Lock()
        self.processing_lock = threading.Lock()
        
        # Batch processing thread
        self.batch_processor_thread = None
        self.stop_processing = threading.Event()
        self._start_batch_processor()
        
        # Forward model attributes to make this a drop-in replacement
        # Get device from model parameters since PyTorch models don't have a direct device attribute
        self.device = next(model.parameters()).device
        
    def _start_batch_processor(self):
        """Start the background batch processing thread."""
        self.batch_processor_thread = threading.Thread(
            target=self._batch_processor_loop,
            daemon=True
        )
        self.batch_processor_thread.start()
        
    def _batch_processor_loop(self):
        """Main loop for processing batched requests."""
        while not self.stop_processing.is_set():
            try:
                # Wait for either timeout or max batch size
                start_time = time.time()
                batch_requests = []
                
                # Collect requests within the time window
                while (time.time() - start_time) * 1000 < self.batch_window_ms:
                    with self.queue_lock:
                        if self.request_queue:
                            batch_requests.append(self.request_queue.popleft())
                            
                            # Force processing if we hit max batch size
                            if len(batch_requests) >= self.max_batch_size:
                                break
                    
                    # Short sleep to avoid busy waiting
                    time.sleep(0.001)
                
                # Collect any remaining requests
                with self.queue_lock:
                    while self.request_queue and len(batch_requests) < self.max_batch_size:
                        batch_requests.append(self.request_queue.popleft())
                
                # Process the batch if we have requests
                if batch_requests:
                    self._process_batch(batch_requests)
                else:
                    # No requests, sleep a bit longer
                    time.sleep(0.005)
                    
            except Exception as e:
                print(f"[BATCH PROCESSOR] Error in batch processing: {e}")
                # Continue processing other requests
                continue
    
    def _process_batch(self, requests: list):
        """
        Process a batch of requests together.
        
        Args:
            requests: List of (request_id, input_tensor) tuples
        """
        try:
            # Extract tensors and request IDs
            request_ids = [req[0] for req in requests]
            input_tensors = [req[1] for req in requests]

            # Ensure inputs are on the model device and contiguous
            for i, t in enumerate(input_tensors):
                if t.device != self.device:
                    input_tensors[i] = t.to(self.device, non_blocking=False).contiguous()
                else:
                    input_tensors[i] = t.contiguous()
            
            # Stack tensors into a batch
            # Each input is shape (num_dots-1, 1, H, W)
            # Batch becomes (batch_size * (num_dots-1), 1, H, W)
            batch_tensor = torch.cat(input_tensors, dim=0)
            
            # Run the model on the entire batch
            with torch.no_grad():
                batch_values, batch_log_vars = self.model(batch_tensor)
            
            # Split results back to individual requests
            channels_per_request = input_tensors[0].shape[0]  # num_dots-1
            
            for i, request_id in enumerate(request_ids):
                start_idx = i * channels_per_request
                end_idx = start_idx + channels_per_request
                
                # Extract this request's results
                request_values = batch_values[start_idx:end_idx].detach().clone()
                request_log_vars = batch_log_vars[start_idx:end_idx].detach().clone()

                # Store results and signal completion
                self.results[request_id] = (request_values, request_log_vars)
                
                # Signal that this request is complete
                if request_id in self.pending_results:
                    self.pending_results[request_id].set()
                    
        except Exception as e:
            print(f"[BATCH PROCESSOR] Error processing batch: {e}")
            # Signal all requests in this batch as failed
            for request_id, _ in requests:
                if request_id in self.pending_results:
                    self.pending_results[request_id].set()
    
    def forward(self, input_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model (batched).
        
        Args:
            input_tensor: Input tensor of shape (num_dots-1, 1, H, W)
            
        Returns:
            Tuple of (values, log_vars) tensors
        """
        return self.__call__(input_tensor)
    
    def __call__(self, input_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Call the model (batched). Maintains exact same interface as original model.
        
        Args:
            input_tensor: Input tensor of shape (num_dots-1, 1, H, W)
            
        Returns:
            Tuple of (values, log_vars) tensors
        """
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        
        # Create completion event
        completion_event = threading.Event()
        self.pending_results[request_id] = completion_event
        
        # Add request to queue
        with self.queue_lock:
            self.request_queue.append((request_id, input_tensor))
        
        # Wait for processing to complete
        completion_event.wait(timeout=30.0)  # 30 second timeout
        
        # Get results
        if request_id in self.results:
            result = self.results[request_id]
            # Clean up
            del self.results[request_id]
            del self.pending_results[request_id]
            return result
        else:
            # Timeout or error occurred
            if request_id in self.pending_results:
                del self.pending_results[request_id]
            raise RuntimeError(f"Capacitance model inference failed or timed out for request {request_id}")
    
    def eval(self):
        """Set model to evaluation mode."""
        return self.model.eval()
    
    def train(self, mode: bool = True):
        """Set model to training mode."""
        return self.model.train(mode)
    
    def to(self, device):
        """Move model to specified device."""
        self.model = self.model.to(device)
        self.device = device
        return self
    
    def parameters(self):
        """Return model parameters."""
        return self.model.parameters()
    
    def state_dict(self):
        """Return model state dictionary."""
        return self.model.state_dict()
    
    def load_state_dict(self, state_dict):
        """Load model state dictionary."""
        return self.model.load_state_dict(state_dict)
    
    def close(self):
        """Clean shutdown of the batch processor."""
        self.stop_processing.set()
        if self.batch_processor_thread:
            self.batch_processor_thread.join(timeout=5.0)
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        self.close()


def create_batched_capacitance_model(original_model, **kwargs) -> BatchedCapacitanceModelWrapper:
    """
    Factory function to create a batched capacitance model wrapper.
    
    Args:
        original_model: The original PyTorch capacitance model
        **kwargs: Additional arguments for BatchedCapacitanceModelWrapper
        
    Returns:
        BatchedCapacitanceModelWrapper instance
    """
    return BatchedCapacitanceModelWrapper(original_model, **kwargs)