#!/bin/sh
#SBATCH --time=168:00:00
#SBATCH --partition=gpu
#SBATCH --mem=0
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --job-name=baseline
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=ayush.agrawal7661@gmail.com
#SBATCH --output=/work/netthinker/shared/out_files/ssl-resnet-test-cifar10.out

export PYTHONPATH=$WORK/tf-gpu-pkgs
module load singularity
singularity exec docker://lordvoldemort28/pytorch-opencv:dev python -u initialization.py
