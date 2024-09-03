#!/bin/bash
#SBATCH --job-name=LabelStudio   # Job name
#SBATCH --mail-type=END               # Mail events
#SBATCH --mail-user=ben.weinstein@weecology.org # Where to send mail
#SBATCH --account=ewhite
#SBATCH --nodes=1                 # Number of MPI r
#SBATCH --cpus-per-task=3
#SBATCH --mem=80GB
#SBATCH --time=48:00:00       #Time limit hrs:min:sec
#SBATCH --output=/home/b.weinstein/logs/label_pipeline_%j.out   # Standard output and error log
#SBATCH --error=/home/b.weinstein/logs/label_pipeline_%j.err
#SBATCH --partition=gpu
#SBATCH --gpus=1

export PATH="$PATH:/orange/ewhite/b.weinstein/miniconda3_new/bin"

source activate AirborneFieldGuide
cd /home/b.weinstein/AirborneFieldGuide/
python run.py $1
