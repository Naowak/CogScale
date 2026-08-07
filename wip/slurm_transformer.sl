#!/bin/bash
#SBATCH --job-name=S_tf2
#SBATCH --output=output_%j.txt
#SBATCH --error=error_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=8

MODELS_GPU0=${1:-"transformer_encdec"}
MODELS_GPU1=${2:-"transformer_encdec"}

OUTPUT_FILE_GPU0="results_${MODELS_GPU0}_3_gpu0.csv"
OUTPUT_FILE_GPU1="results_${MODELS_GPU1}_3_gpu1.csv"

export TORCH_EXTENSIONS_DIR=/beegfs/ybendiou/bstream/torch_extensions
export TRITON_CACHE_DIR=/beegfs/ybendiou/bstream/triton_cache

module load compiler/cuda/12.3
module load compiler/gcc/11.2.0

# Initialize env
eval "$(conda shell.bash hook)"
conda init
conda activate /beegfs/ybendiou/bstream/venv
pip install joblib

cd /beegfs/ybendiou/bstream/baseline/

# Repartition of tasks to run (6 tasks total, generating 900 combinations)
# GPU 0 will run the first 3 tasks (contains the 155 remaining/unfinished tasks)
# GPU 1 will check the last 3 tasks (which are already fully completed)
TASKS_GPU0='simple_copy selective_copy sorting_problem'
TASKS_GPU1='associative_recall induction_heads bracket_matching'

echo "Lancement sur GPU 0 des modèles : $MODELS_GPU0"
srun --ntasks=1 python run.py \
    --tasks $TASKS_GPU0 \
    --difficulties large \
    --sizes 1000 10000 100000 \
    --seeds 10 \
    --epochs 200 \
    --device cuda:0 \
    --dtype float32 \
    --model_type $MODELS_GPU0 \
    --skip-done \
    --output $OUTPUT_FILE_GPU0 &

echo "Lancement sur GPU 1 des modèles : $MODELS_GPU1"
srun --ntasks=1 python run.py \
    --tasks $TASKS_GPU1 \
    --difficulties large \
    --sizes 1000 10000 100000 \
    --seeds 10 \
    --epochs 200 \
    --device cuda:1 \
    --dtype float32 \
    --model_type $MODELS_GPU1 \
    --skip-done \
    --output $OUTPUT_FILE_GPU1 &

wait
echo "Toutes les tâches sont terminées."