#!/bin/sh
#SBATCH --time=168:00:00
#SBATCH --partition=gpu
#SBATCH --mem=0
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --constraint=gpu_32gb
#SBATCH --job-name=tst-freeze-r50
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=ayush.agrawal7661@gmail.com
#SBATCH --output=./out_log/tst-freeze-ssl-r-c10.out

export PYTHONPATH=$WORK/tf-gpu-pkgs
module load singularity
singularity exec docker://lordvoldemort28/pytorch-opencv:dev python -u initialization.py
