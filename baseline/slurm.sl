#!/bin/bash
#SBATCH --job-name=STREAM-eval
#SBATCH --output=output_%j.txt
#SBATCH --error=error_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=8

# 1. Récupération des paramètres passés lors du sbatch
# La syntaxe ${1:-"valeur_par_defaut"} permet d'avoir un comportement par défaut si tu ne passes rien
# Possibles values : 
# "lstm", "gru", "transformer_decoder", "transformer_encdec", "esn", "dynamical_transformer", "mamba", "xlstm"
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
echo "Lancement sur GPU 0 des modèles : $MODELS_GPU0"
srun --ntasks=1 python run.py \
    --tasks all \
    --difficulties small medium \
    --sizes 1000 10000 100000 \
    --seeds 10 \
    --epochs 200 \
    --device cuda:0 \
    --dtype float32 \
    --model_type $MODELS_GPU0 --output $OUTPUT_FILE_GPU0 &

echo "Lancement sur GPU 1 des modèles : $MODELS_GPU1"
srun --ntasks=1 python run.py \
    --tasks all \
    --difficulties small medium \
    --sizes 1000 10000 100000 \
    --seeds 10 \
    --epochs 200 \
    --device cuda:1 \
    --dtype float32 \
    --model_type $MODELS_GPU1 --output $OUTPUT_FILE_GPU1 &

wait
echo "Toutes les tâches sont terminées."