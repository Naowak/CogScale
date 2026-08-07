#!/bin/bash
#SBATCH --job-name=bstream_run
#SBATCH --output=logs/slurm_array_%A_%a.out
#SBATCH --error=logs/slurm_array_%A_%a.err
#SBATCH -C h100
#SBATCH --nodes=1                     
#SBATCH --gres=gpu:4           # Allocation complète du nœud (4 GPUs)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96     # Optimisation pour alimenter les processus
#SBATCH --time=20:00:00        # Limite maximale standard
#SBATCH --hint=nomultithread
#SBATCH --account=emr@h100     # À vérifier avec votre projet
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --array=1-4            # 3 nœuds pour traiter les 12 combinaisons

# 1. Chargement des modules
module purge
module load arch/h100
module load pytorch-gpu/py3/2.8.0
module load parallel

# 2. Activation de l'environnement virtuel (à adapter selon le chemin exact)
source $WORK/bstream/bstream_venv/bin/activate

# 3. Redirection des caches de compilation vers l'espace scratch temporaire 
# (Crucial pour éviter les conflits d'accès concurrents de Triton/Torch sur plusieurs GPUs)
export TORCH_EXTENSIONS_DIR=$JOBSCRATCH/torch_extensions
export TRITON_CACHE_DIR=$JOBSCRATCH/triton_cache
mkdir -p $TORCH_EXTENSIONS_DIR $TRITON_CACHE_DIR

# 4. Navigation vers le dossier du code
cd $WORK/bstream/baseline/ # <-- À remplacer par votre chemin exact

# 5. Génération dynamique du registre de tâches (exécutée par le 1er nœud de l'array)
REGISTRY="tasks_registry.txt"

if [ "$SLURM_ARRAY_TASK_ID" -eq 1 ]; then
    echo "Génération du registre de tâches..." >&2
    > $REGISTRY
    MODELS=("xlstm")
    TASKS=("simple_copy" "selective_copy" "sorting_problem" "induction_heads")
    
    for MODEL in "${MODELS[@]}"; do
        for TASK in "${TASKS[@]}"; do
            # Construction de la commande exacte pour run.py
            CMD="python run.py --tasks $TASK --difficulties large --sizes 1000 10000 100000 --seeds 10 --epochs 200 --dtype float32 --model_type $MODEL --output results_${MODEL}_${TASK}.csv"
            echo "$CMD" >> $REGISTRY
        done
    done
fi

# On s'assure que le nœud 2 attend que le nœud 1 ait fini de créer le fichier
sleep 10

# 6. Répartition des tâches sur les nœuds
LOCAL_TASKS="local_tasks_${SLURM_ARRAY_TASK_ID}.txt"

# Modulo 3 car nous utilisons 3 nœuds (--array=1-3)
# Le Nœud 1 prendra les lignes 1, 4, 7... Le Nœud 2 prendra les lignes 2, 5, 8...
# Le Nœud 3 prendra les lignes 3, 6, 9...
awk -v id=$SLURM_ARRAY_TASK_ID 'NR % 4 == (id - 1)' $REGISTRY > $LOCAL_TASKS

TOTAL_TASKS=$(wc -l < $LOCAL_TASKS)
echo "=== 🚀 NŒUD ARRAY $SLURMD_NODENAME (ID: $SLURM_ARRAY_TASK_ID) | $TOTAL_TASKS TÂCHES ==="

# 7. Exécution parallèle
# -j 8 permet de lancer 8 processus simultanés (2 par GPU). 
# Chaque processus est assigné à un GPU spécifique via CUDA_VISIBLE_DEVICES.
parallel -j 8 --delay 1 --joblog parallel_joblog_${SLURM_ARRAY_TASK_ID}.txt 'CUDA_VISIBLE_DEVICES=$(({%} % 4)) eval {}' < $LOCAL_TASKS

# 8. Nettoyage
rm $LOCAL_TASKS