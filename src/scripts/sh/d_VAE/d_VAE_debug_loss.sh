#!/bin/bash
#SBATCH --job-name=d_VAE_debug
#SBATCH --qos=qos_gpu-t3
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8g
#SBATCH --output=slurm_out/d_VAE_debug-%j.out
#SBATCH --error=slurm_out/d_VAE_debug-%j.err
#SBATCH --time=20:00:00
#SBATCH -A ktz@gpu

export TMPDIR=$JOBSCRATCH
module purge
module load  pytorch-gpu/py3/1.7.1
conda activate smc-t

export PYTHONPATH=src:${PYTHONPATH}

MODEL="d_VAE"
OUTPUT_PATH="output/d_VAE/debug_loss"
NUM_LAYERS=4
D_MODEL=128
DFF=512
BS=32
EP=30
LATENT="attention"
SUBSIZE=10
SAMPLES_LOSS=1

srun python -u src/scripts/run_transformer.py -model $MODEL -latent $LATENT -num_layers $NUM_LAYERS -d_model $D_MODEL -dff $DFF -bs $BS -ep $EP -output_path $OUTPUT_PATH -subsize $SUBSIZE -debug_loss 1 -samples_loss $SAMPLES_LOSS
