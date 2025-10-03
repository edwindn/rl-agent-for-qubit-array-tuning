#!/usr/bin/env python
"""
Modal runner script - triggers training on the deployed GPU cluster.
Run with: python run_modal.py (after deploying modal_init.py)
"""

import modal

# Look up the deployed app and function
train_function = modal.Function.lookup("qarray-rl-training", "train")

# Trigger the training job
print("🚀 Starting RL training job on Modal GPU cluster...")
call = train_function.spawn()
print(f"Training job submitted. Call ID: {call.object_id}")
print(f"View logs at: https://modal.com/apps")
print(f"\nTo stream logs, run: modal app logs qarray-rl-training")
