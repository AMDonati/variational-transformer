#!/bin/bash
#SBATCH --job-name=warmup10-vae-attn-l1-d32-transformer
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
OUTPUT_PATH="output/debug_overfit_smalldim_valdata5000"
NUM_LAYERS=1
D_MODEL=32
NUM_HEADS=1
DFF=32
BS=32
EP=30
LATENT="attention"
BETA_SCHEDULE="warmup"
N_CYCLE=10

srun python -u src/scripts/run_transformer.py -model $MODEL -latent $LATENT -num_layers $NUM_LAYERS -d_model $D_MODEL -dff $DFF -bs $BS -ep $EP -output_path $OUTPUT_PATH -beta_schedule $BETA_SCHEDULE -n_cycle $N_CYCLE -num_heads $NUM_HEADS