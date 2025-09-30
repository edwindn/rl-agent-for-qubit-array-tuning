## Slurm Commands

To submit a job: sbatch scriptname.slurm


To check queued jobs (on htc): squeue -M htc -u $USER

To kill a job: scancel -M htc <job_id>

## Running order:

create .env file with WANDB_API_KEY=key

authenticate git ssh

then with cwd as root of the repo run

sbatch train_scripts/env_arc.slurm
sbatch train_scripts/arc.slurm