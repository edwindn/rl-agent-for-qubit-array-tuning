## Slurm Commands

To submit a job: sbatch scriptname.slurm


To check queued jobs (on htc): squeue -M htc -u $USER

To kill a job: scancel -M htc <job_id>

## Running order:

create .env file with WANDB_API_KEY=key:
* echo "WANDB_API_KEY=<your_key_here> > .env

Generate ssh key and add to github account
Check by running ssh -T git@github.com 
(needed for install of private qarray_latched repo)

Create a /logs directory in the project root

then with cwd as root of the repo run:
* sbatch train_scripts/env_arc.slurm
* sbatch train_scripts/arc.slurm
