#!/bin/bash -l
#SBATCH --job-name=warp1110-reproduce
#SBATCH --partition=overflow
#SBATCH --account=wrd
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --no-requeue
#SBATCH --export=NONE

set -euo pipefail
export SLURM_EXPORT_ENV=ALL
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
task_checkout=$(realpath "${1:?Pass the restored source checkout}")
task_bundle=$(realpath "${2:?Pass the published overflow asset directory}")
task_seed="${3:-1110}"
task_run="$task_bundle/reproduced-jobs/$SLURM_JOB_ID"
task_scripts="$task_bundle/jobs/1144767"
mkdir -p "$task_run"
for task_script in benchmark_capsules.py validate_filled.py run_record.py summarize.py select_device.py worker.py finish_report.py; do
    cp "$task_scripts/$task_script" "$task_run/"
done
cd "$task_checkout"
export WARP_CACHE_PATH="$task_checkout/_build/overflow-$SLURM_JOB_ID-cache"
srun --ntasks=1 --gpus-per-task=b200:1 --gpu-bind=single:1 uv run --no-sync "$task_run/worker.py" --checkout "$task_checkout" --output "$task_run" --seed "$task_seed"
