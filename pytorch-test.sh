#!/bin/bash
#SBATCH --gres=gpu:v100l:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=6  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32000M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=2-00:00
#SBATCH --output=%N-%j.out



# Log GPU memory usage every minute
while true; do
    echo "Memory usage at $(date):" >> gpu_memory_usage.log
    nvidia-smi >> gpu_memory_usage.log
    sleep 60
done &



python3 main.py --train_task sen trans qa --test_task sen trans qa


