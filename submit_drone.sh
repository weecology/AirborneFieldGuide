#!/bin/bash
#SBATCH --job-name=EvergladesDroneIterate   # Job name
#SBATCH --mail-type=END               # Mail events
#SBATCH --mail-user=ben.weinstein@weecology.org # Where to send mail
#SBATCH --account=ewhite
#SBATCH --nodes=1                 # Number of MPI r
#SBATCH --cpus-per-task=10
#SBATCH --mem=100GB
#SBATCH --time=48:00:00       #Time limit hrs:min:sec
#SBATCH --output=/home/b.weinstein/logs/drone_pipeline_%j.out   # Standard output and error log
#SBATCH --error=/home/b.weinstein/logs/drone_pipeline_%j.err
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1

source activate AirborneFieldGuide
python -m cProfile -o "/home/b.weinstein/logs/drone_profile.prof" drone_pipeline.py