#!/bin/bash
#SBATCH --job-name=wikitext_benchmark
#SBATCH --output=wikitext_%j.out
#SBATCH --error=wikitext_%j.err
#SBATCH -p gpu-large
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00

set -euo pipefail

module purge
module load python/3.11.14

cd /lustre/uc3m/gts_c3_cluster_1-12/NFattention
source .venv/bin/activate

echo "Running on $(hostname)"
which python
python -V
python -c "import torch; print(torch.__version__); print('cuda?', torch.cuda.is_available()); print('devices', torch.cuda.device_count())"
nvidia-smi

# Slurm gives us one visible GPU. Use index 0 inside the job.
export CUDA_VISIBLE_DEVICES=0

python -m experiments.language_modelling.wikitext_perplexity_benchmark.run_benchmark \
  --seeds 0 \
  --models gmm_rks \
  --gpu 0 \
  --no_cpu_fallback

