#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --time=00:30:00
#SBATCH --job-name=pelec-post
#SBATCH --output=pelec-post_%j.out
#SBATCH --error=pelec-post_%j.err
#SBATCH --partition=campus
#SBATCH --account="ACF-UTK0011"

## Bash script to run the PeleC Python post-processing pipeline using SLURM
## on the Campus cluster. Adjust the parameters below as needed.

## Set run variables
PythonScript="./pelec_post.py"        # Path to the post-processing script
WorkingDir="/lustre/isaac24/scratch/sbrollia/PeleC-Python"  # Working directory
CondaEnv="base"                       # Conda environment to activate (use "base" or your env name)

## --- Print job information --- ##
echo "PeleC Post-Processing SLURM Job Script"
echo "Starting job: $SLURM_JOB_NAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on nodes: $SLURM_NODELIST"
echo "Number of tasks: $SLURM_NTASKS"
echo "Time limit: $SLURM_TIMELIMIT"
echo ""
echo ""

## --- Print run configuration --- ##
echo "Post-Processing Configuration:"
echo "Working Directory: $WorkingDir"
echo "Python Script: $PythonScript"
echo "Conda Environment: $CondaEnv"
echo ""

cd $WorkingDir

## --- Load necessary modules --- ##
echo "Loading modules..."
module load gcc/13.4.0 openmpi/4.1.8-gcc13

## --- Initialize conda --- ##
echo "Initializing conda..."
eval "$(conda shell.bash hook)"
conda activate $CondaEnv

## Verify Python environment
echo "Python executable: $(which python3)"
echo "Python version: $(python3 --version)"
echo ""

## --- Run post-processing --- ##
echo "Running PeleC post-processing..."
python3 $PythonScript
echo "Post-processing completed."
