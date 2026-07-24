#!/bin/bash
#SBATCH --job-name=quora_benchmark
#SBATCH --output=quora_%j.out
#SBATCH --error=quora_%j.err
#SBATCH -p gpu-large
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4-00:00:00

module purge
module load python/3.11.14

cd /lustre/uc3m/gts_c3_cluster_1-12/NFattention
source .venv/bin/activate

echo "Running on $(hostname)"
which python
python -V
python -c "import torch; print(torch.__version__); print('cuda?', torch.cuda.is_available()); print('devices', torch.cuda.device_count())"
nvidia-smi

python experiments/text_classification/text_seq_classification_benchmark/run_quora_all_models.py --gpu=0