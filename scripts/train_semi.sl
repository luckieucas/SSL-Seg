#!/bin/bash
#SBATCH --job-name=train_semi   # Job name
#SBATCH --time=14:00:00           # Maximum runtime
#SBATCH --nodes=1                 # Number of nodes
#SBATCH --ntasks=1                # Number of tasks
#SBATCH --cpus-per-task=6         # Number of CPU cores per task
#SBATCH --gres=gpu:1                  # Number of GPUs per task
#SBATCH --mem=32gb                # Memory allocation
#SBATCH --partition=weilab   # Partition to submit the job
#SBATCH --mail-type=BEGIN,END,FAIL  # Email notifications (on job start, end, or fail)
#SBATCH --mail-user=liupen@bc.edu  # Email address for notifications
#SBATCH --output=/projects/weilab/liupeng/semi-seg/log/train_semi_%j.out  
# Activate the Conda environment
source ~/miniconda3/bin/activate
conda activate ssl_seg

# Print debug information
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "Start time: $(date)"

# Run the nnUNet training command
python train_semi-supervised.py --c
# Print end time
echo "Training completed at $(date)"
