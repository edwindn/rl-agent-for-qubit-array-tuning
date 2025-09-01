"""
Manager for capacitance model actors across multiple GPUs.

This manager creates and manages Ray actors hosting capacitance models,
providing automatic GPU assignment and load balancing for Ray workers.
"""

import os
import torch
import ray
from typing import List, Dict, Optional
import hashlib

from capacitance_model_actor import create_capacitance_actor


class CapacitanceActorManager:
    """
    Manager for capacitance model actors distributed across GPUs.
    
    This class creates one actor per GPU and provides methods for Ray workers
    to get the appropriate actor reference for their GPU assignment.
    """
    
    def __init__(self, checkpoint_path: str, gpu_list: List[int], batch_window_ms: int = 100, max_batch_size: int = 128):
        """
        Initialize the actor manager.
        
        Args:
            checkpoint_path: Path to model checkpoint
            gpu_list: List of GPU IDs to use
            batch_window_ms: Batching window for actors
            max_batch_size: Maximum batch size for actors
        """
        self.checkpoint_path = checkpoint_path
        self.gpu_list = gpu_list
        self.batch_window_ms = batch_window_ms
        self.max_batch_size = max_batch_size
        
        # Dictionary mapping GPU ID to actor reference
        self.actors: Dict[int, ray.ObjectRef] = {}
        
        # Create actors
        self._create_actors()
        
    def _create_actors(self):
        """Create one actor per GPU."""
        print(f"Creating capacitance model actors on GPUs: {self.gpu_list}")
        
        for gpu_id in self.gpu_list:
            try:
                actor_ref = create_capacitance_actor(
                    checkpoint_path=self.checkpoint_path,
                    gpu_id=gpu_id,
                    batch_window_ms=self.batch_window_ms,
                    max_batch_size=self.max_batch_size
                )
                self.actors[gpu_id] = actor_ref
                print(f"✓ Created actor for GPU {gpu_id}")
                
            except Exception as e:
                print(f"✗ Failed to create actor for GPU {gpu_id}: {e}")
                raise
        
        # Wait for all actors to initialize and verify they're working
        self._verify_actors()
    
    def _verify_actors(self):
        """Verify all actors are properly initialized."""
        print("Verifying actor initialization...")
        
        for gpu_id, actor_ref in self.actors.items():
            try:
                # Get device info to verify initialization
                device_info = ray.get(actor_ref.get_device_info.remote())
                print(f"✓ GPU {gpu_id} actor: {device_info}")
                
            except Exception as e:
                print(f"✗ GPU {gpu_id} actor verification failed: {e}")
                raise
    
    def get_actor_for_worker(self) -> ray.ObjectRef:
        """
        Get the appropriate actor reference for the current Ray worker.
        
        This method uses Ray's GPU assignment to determine which actor
        the current worker should use.
        
        Returns:
            Ray object reference to the appropriate capacitance model actor
        """
        try:
            # Get GPU IDs assigned to this Ray worker
            worker_gpus = ray.get_gpu_ids()
            
            if worker_gpus:
                # Use the first GPU assigned to this worker
                worker_gpu = worker_gpus[0]
                
                # Map to the actual GPU in our list (Ray may renumber GPUs)
                if worker_gpu < len(self.gpu_list):
                    actual_gpu_id = self.gpu_list[worker_gpu]
                else:
                    # Fallback: round-robin assignment
                    actual_gpu_id = self.gpu_list[worker_gpu % len(self.gpu_list)]
                
                if actual_gpu_id in self.actors:
                    return self.actors[actual_gpu_id]
                else:
                    # Fallback to first available actor
                    return self.actors[self.gpu_list[0]]
            else:
                # No GPU assigned to worker, use first actor (probably CPU)
                return self.actors[self.gpu_list[0]]
                
        except Exception as e:
            print(f"Warning: Failed to determine worker GPU assignment: {e}")
            # Fallback to first actor
            return self.actors[self.gpu_list[0]]
    
    def get_actor_by_gpu(self, gpu_id: int) -> Optional[ray.ObjectRef]:
        """Get actor reference for a specific GPU."""
        return self.actors.get(gpu_id)
    
    def get_all_actors(self) -> Dict[int, ray.ObjectRef]:
        """Get all actor references."""
        return self.actors.copy()
    
    def get_worker_assignment_info(self) -> dict:
        """Get information about how this worker should be assigned."""
        try:
            worker_gpus = ray.get_gpu_ids()
            assigned_actor_gpu = None
            
            if worker_gpus and worker_gpus[0] < len(self.gpu_list):
                assigned_actor_gpu = self.gpu_list[worker_gpus[0]]
            elif self.gpu_list:
                assigned_actor_gpu = self.gpu_list[0]
            
            return {
                'worker_gpus': worker_gpus,
                'assigned_actor_gpu': assigned_actor_gpu,
                'available_actors': list(self.actors.keys())
            }
        except Exception as e:
            return {'error': str(e)}
    
    def cleanup(self):
        """Clean up all actors."""
        print("Cleaning up capacitance model actors...")
        
        cleanup_futures = []
        for gpu_id, actor_ref in self.actors.items():
            try:
                future = actor_ref.cleanup.remote()
                cleanup_futures.append((gpu_id, future))
            except Exception as e:
                print(f"Failed to initiate cleanup for GPU {gpu_id} actor: {e}")
        
        # Wait for all cleanups to complete
        for gpu_id, future in cleanup_futures:
            try:
                ray.get(future, timeout=10)
                print(f"✓ GPU {gpu_id} actor cleaned up")
            except Exception as e:
                print(f"✗ GPU {gpu_id} actor cleanup failed: {e}")
    
    def __del__(self):
        """Cleanup when manager is destroyed."""
        try:
            self.cleanup()
        except:
            pass  # Ignore cleanup errors during destruction


def create_capacitance_actor_manager(gpu_list: List[int], checkpoint_path: str = "artifacts/best_model.pth", **kwargs) -> CapacitanceActorManager:
    """
    Factory function to create a capacitance actor manager.
    
    Args:
        gpu_list: List of GPU IDs to use
        checkpoint_path: Path to model checkpoint
        **kwargs: Additional arguments for actors
        
    Returns:
        CapacitanceActorManager instance
    """
    return CapacitanceActorManager(checkpoint_path, gpu_list, **kwargs)