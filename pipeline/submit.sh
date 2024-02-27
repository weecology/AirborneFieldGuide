#!/bin/bash
#SBATCH --job-name=EvergladesIterate   # Job name
#SBATCH --mail-type=END               # Mail events
#SBATCH --mail-user=ben.weinstein@weecology.org # Where to send mail
#SBATCH --account=ewhite
#SBATCH --nodes=1                 # Number of MPI r
#SBATCH --cpus-per-task=10
#SBATCH --mem=60GB
#SBATCH --time=48:00:00       #Time limit hrs:min:sec
#SBATCH --output=/blue/ewhite/everglades/EvergladesSpeciesModel/logs/EvergladesSpeciesModel_%j.out   # Standard output and error log
#SBATCH --error=/blue/ewhite/everglades/EvergladesSpeciesModel/logs/EvergladesSpeciesModel_%j.err
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1

ulimit -c 0
source activate EvergladesTools
cd /blue/ewhite/everglades/EvergladesTools
python pipeline.py