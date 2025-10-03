Setup API key on Modal web

Activate profile with associated api key on command line using:
modal profile activate vradam

Add wandb API key to modal using
modal secret create wandb-secret WANDB_API_KEY=********************

The run using:
modal run modal_train.py