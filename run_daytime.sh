#!/bin/bash
#SBATCH --job-name=c4fairness
#SBATCH --time=20:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=defq
#SBATCH -C cpunode|A4000
#SBATCH --array=0-7%3
#SBATCH --output=logs/%x_%A_%a.out

# CPU-only clustering (KMedoids / HDBSCAN, Gower). No GPU -> cpunode.
# 8 runs = 2 datasets x 2 algorithms x 2 sensitive-weight levels.
#   ds   = idx / 4        -> 0 deep, 1 global
#   wt   = (idx / 2) % 2  -> 0 weight=1.5, 1 weight=2
#   algo = idx % 2        -> 0 kmedoids, 1 hdbscan
# Each run uses plain --experiment, which emits every condition
# (incl. +REG-sen-err and -reg+SEN-err) as separate rows in results_summary.csv.

PROJECT="/var/scratch/$USER/Clustering_4_Fairness"

source "$HOME/.bashrc"
conda activate            # replace with `conda activate <envname>` if you made one

cd "$PROJECT"
mkdir -p logs
which python
python --version

FILES=(deep_oversampled_errors_binary_student_data \
       global_oversampled_errors_binary_student_data)
LABELS=(deep global)
WEIGHTS=(1.5 2)

ds=$((   SLURM_ARRAY_TASK_ID / 4 ))
wt=$(( ( SLURM_ARRAY_TASK_ID / 2 ) % 2 ))
algo=$((  SLURM_ARRAY_TASK_ID % 2 ))

DATA="$PROJECT/Data/experiments_datasets/${FILES[$ds]}.csv"
LABEL="${LABELS[$ds]}"
W="${WEIGHTS[$wt]}"
WTAG="${W/./}"            # 1.5 -> 15, 2 -> 2

REGULAR="num_of_prev_attempts,highest_education,studied_credits,module_presentation_length,period,date_registration,num_assessments_course,code_module"
SENSITIVE="gender,region,imd_band,disability,age_band"
CATEGORICAL="imd_band,highest_education,code_module,region,age_band"
FEATURE_WEIGHTS="disability:$W,gender:$W,region:$W,imd_band:$W,age_band:$W"

if [ "$algo" -eq 0 ]; then
    ALGO_ARGS="--algorithm kmedoids --method fasterpam --distance gower --n_min 2 --n_max 25"
    TAG="${LABEL}_kmed_gow_w${WTAG}"
else
    ALGO_ARGS="--algorithm hdbscan --distance gower --min_samples 20 --min_datapoints 675"
    TAG="${LABEL}_hdb_gow_w${WTAG}"
fi

DATE=$(date +%F)

echo "=== [$SLURM_ARRAY_TASK_ID] $TAG  ($DATA)  weights=$FEATURE_WEIGHTS ==="
OUTDIR="$PROJECT/clustering_results/$DATE/$TAG"
mkdir -p "$OUTDIR"

python -u main.py \
    --data_path "$DATA" \
    $ALGO_ARGS \
    --scoring silhouette \
    --seed 11 \
    --regular_cols "$REGULAR" \
    --sensitive_cols "$SENSITIVE" \
    --categorical_cols "$CATEGORICAL" \
    --feature_weights "$FEATURE_WEIGHTS" \
    --error_col "error" \
    --error_type binary \
    --experiment \
    --no_standardize \
    --projection pca \
    --output_dir "$OUTDIR"
