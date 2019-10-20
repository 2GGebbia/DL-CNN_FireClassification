#!/bin/bash

#SBATCH --job-name="fire_cnn"
#SBATCH --qos=training
#SBATCH --workdir=.
#SBATCH --output=results/fire_%j.out
#SBATCH --error=results/fire_%j.err
#SBATCH --ntasks=4
#SBATCH --gres gpu:1
#SBATCH --time=10:00:00

module purge; module load K80/default impi/2018.1 mkl/2018.1 cuda/8.0 CUDNN/7.0.3 python/3.6.3_ML

python model/fire.py 
