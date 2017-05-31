#!/bin/bash

#Sbatch for: Human pose detection CNN
#################
#set a job name
#SBATCH --job-name=pose_cnn
#SBATCH --mail-user=akma327@stanford.edu --mail-type=ALL
#################
# A file for job output, you can check job progress
#SBATCH --output=/scratch/PI/rondror/akma327/classes/CS231A/project/Squatty/src/batch/053117_cnn_large/pose_cnn_progress.out
#################
# A file for errors from the job
#SBATCH --error=/scratch/PI/rondror/akma327/classes/CS231A/project/Squatty/src/batch/053117_cnn_large/pose_cnn_error.out
#################
#SBATCH --time=3:00:00
#################
#SBATCH --partition=rondror
#SBATCH --qos=rondror
#################
#number of nodes you are requesting
#SBATCH --tasks=4
#SBATCH --ntasks-per-socket=6
#SBATCH --mem=45000
#################

echo "Starting..."

module load tensorflow/0.12.1

cd /scratch/PI/rondror/akma327/classes/CS231A/project/Squatty/src
/share/PI/rondror/software/miniconda/bin/python pose_cnn.py
