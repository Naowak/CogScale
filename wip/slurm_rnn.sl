#!/bin/bash
#SBATCH --job-name=S_2gru-lstm
#SBATCH --output=output_%j.txt
#SBATCH --error=error_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=8

MODELS_GPU0=${1:-"gru"}
MODELS_GPU1=${2:-"lstm"}

OUTPUT_FILE_GPU0="results_${MODELS_GPU0}_2.csv"
OUTPUT_FILE_GPU1="results_${MODELS_GPU1}_2.csv"

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

# Run 
if [ "$MODELS_GPU0" == "$MODELS_GPU1" ]; then
    echo "Même modèle sur les deux GPU, on répartit les tâches."
    TASKS_GPU0='first_half'
    TASKS_GPU1='second_half'
else
    # TASKS_GPU0='simple_copy selective_copy sorting_problem'
    TASKS_GPU0='associative_recall induction_heads bracket_matching'
    TASKS_GPU1='associative_recall induction_heads bracket_matching'
fi

echo "Lancement sur GPU 0 des modèles : $MODELS_GPU0"
srun --ntasks=1 python run.py \
    --tasks $TASKS_GPU0 \
    --difficulties large \
    --sizes 1000 10000 100000 \
    --seeds 10 \
    --epochs 200 \
    --device cuda:0 \
    --dtype float32 \
    --model_type $MODELS_GPU0 --output $OUTPUT_FILE_GPU0 &

echo "Lancement sur GPU 1 des modèles : $MODELS_GPU1"
srun --ntasks=1 python run.py \
    --tasks $TASKS_GPU1 \
    --difficulties large \
    --sizes 1000 10000 100000 \
    --seeds 10 \
    --epochs 200 \
    --device cuda:1 \
    --dtype float32 \
    --model_type $MODELS_GPU1 --output $OUTPUT_FILE_GPU1 &

wait
echo "Toutes les tâches sont terminées."