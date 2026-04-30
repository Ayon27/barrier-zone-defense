#!/bin/bash
#SBATCH -p uri-gpu
#SBATCH -A pi_kaleel_mahmood_uri_edu 
#SBATCH --job-name=barz-square-4gpu
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:4
#SBATCH --time=240:00:00
#SBATCH --output=barz/results/slurm_%x_%j.out
#SBATCH --error=barz/results/slurm_%x_%j.err

set -euo pipefail

cd /home/syed_ayon_uri_edu/code/ballot_black_box

module load conda/latest
conda activate torch311

srun --exclusive --ntasks=1 --gres=gpu:1 \
    python3 -u barz/run_barz_square.py &

srun --exclusive --ntasks=1 --gres=gpu:1 \
    python3 -u barz/run_barz_square_ramp_ce.py &

srun --exclusive --ntasks=1 --gres=gpu:1 \
    python3 -u barz/run_barz_square_ramp_ce_ensemble.py &

srun --exclusive --ntasks=1 --gres=gpu:1 \
    python3 -u barz/run_barz_square_diffusion_linf.py &

wait

echo "All 4 Square jobs completed."