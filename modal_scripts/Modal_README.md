Setup API key on Modal web

Activate profile with associated api key on command line using:
modal profile activate vradam

Add wandb API key to modal using
modal secret create wandb-secret WANDB_API_KEY=********************

Add PAT token
 modal secret create github-read-private GITHUB_TOKEN=github_pat_**********

The run using:
modal run modal_scripts/modal_train.py