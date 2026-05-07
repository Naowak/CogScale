#!/bin/bash
#SBATCH --job-name=STREAM-eval
#SBATCH --output=output_%j.txt
#SBATCH --error=error_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=8

MODELS_GPU0=${1:-"lstm"}
MODELS_GPU1=${2:-"transformer_decoder"}

OUTPUT_FILE_GPU0="results_${MODELS_GPU0}.csv"
OUTPUT_FILE_GPU1="results_${MODELS_GPU1}.csv"

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
    TASKS_GPU0='all'
    TASKS_GPU1='all'
fi

echo "Lancement sur GPU 0 des modèles : $MODELS_GPU0"
srun --ntasks=1 python run.py \
    --tasks $TASKS_GPU0 \
    --difficulties small medium \
    --sizes 1000 10000 100000 \
    --seeds 10 \
    --epochs 200 \
    --device cuda:0 \
    --dtype float32 \
    --model_type $MODELS_GPU0 --output $OUTPUT_FILE_GPU0 &

echo "Lancement sur GPU 1 des modèles : $MODELS_GPU1"
srun --ntasks=1 python run.py \
    --tasks $TASKS_GPU1 \
    --difficulties small medium \
    --sizes 1000 10000 100000 \
    --seeds 10 \
    --epochs 200 \
    --device cuda:1 \
    --dtype float32 \
    --model_type $MODELS_GPU1 --output $OUTPUT_FILE_GPU1 &

wait
echo "Toutes les tâches sont terminées."