#!/bin/bash

#SBATCH --account=def-dmg
#SBATCH --gres=gpu:lgpu:4
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --time=0-30:00            # time (DD-HH:MM)

python solution.py --mode train --log_dir ./logs_EmotioNet --save_dir ./save_EmotioNet --data_dir EmotioNet --resize True
