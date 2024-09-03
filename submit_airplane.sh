#!/bin/bash
#SBATCH --job-name=AirplaneIterate   # Job name
#SBATCH --mail-type=END               # Mail events
#SBATCH --mail-user=ben.weinstein@weecology.org # Where to send mail
#SBATCH --account=ewhite
#SBATCH --nodes=1                 # Number of MPI r
#SBATCH --cpus-per-task=10
#SBATCH --mem=60GB
#SBATCH --time=48:00:00       #Time limit hrs:min:sec
#SBATCH --output=/home/b.weinstein/logs/Airplane_pipeline_%j.out   # Standard output and error log
#SBATCH --error=/home/b.weinstein/logs/Airplane_pipeline_%j.err
#SBATCH --partition=gpu
#SBATCH --gpus=1

source activate AirborneFieldGuide
cd /home/b.weinstein/AirborneFieldGuide/
python airplane_pipeline.py
