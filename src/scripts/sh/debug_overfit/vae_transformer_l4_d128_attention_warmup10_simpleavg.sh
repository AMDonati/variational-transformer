#!/bin/bash
#SBATCH --job-name=warmup5-vae-attn-l4-d128-transformer
#SBATCH --qos=qos_gpu-t3
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8g
#SBATCH --output=slurm_out/warmup5-vae-transformer-attn-l4-d128-%j.out
#SBATCH --error=slurm_out/warmup5-vae-transformer-attn-l4-d128-%j.err
#SBATCH --time=20:00:00
#SBATCH -A ktz@gpu

export TMPDIR=$JOBSCRATCH
module purge
module load  pytorch-gpu/py3/1.7.1
conda activate smc-t

export PYTHONPATH=src:${PYTHONPATH}

MODEL="VAE"
OUTPUT_PATH="output/debug_overfit/simple_avg"
NUM_LAYERS=4
D_MODEL=128
DFF=512
BS=32
EP=30
LATENT="attention"
BETA_SCHEDULE="warmup"
N_CYCLE=10

srun python -u src/scripts/run_transformer.py -model $MODEL -latent $LATENT -num_layers $NUM_LAYERS -d_model $D_MODEL -dff $DFF -bs $BS -ep $EP -output_path $OUTPUT_PATH -beta_schedule $BETA_SCHEDULE -n_cycle $N_CYCLE -simple_average 1
