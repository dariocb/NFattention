#!/bin/bash
# Prepare the UC3M environment and submit every rebuttal experiment.
#
# Run from any location with:
#   bash /home/dacabeza/lustre/dacabeza/NFattention/rebuttal/run_all_slurm.sh
#
# Optional environment overrides:
#   SKIP_INSTALL=1  Skip the idempotent pip installation step.
#   LISTOPS_TIME=1-12:00:00  Override the default two-day ListOps limit.

set -euo pipefail

# Step 1: Resolve the repository from this script instead of relying on the
# caller's working directory or a historical Lustre mount.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
WORKER="${SCRIPT_DIR}/rebuttal_gpu.sbatch"
EXPECTED_CLUSTER_REPO="/home/dacabeza/lustre/dacabeza/NFattention"

echo "Repository resolved to: ${REPO}"
if [[ "${REPO}" != "${EXPECTED_CLUSTER_REPO}" ]]; then
    echo "WARNING: expected ${EXPECTED_CLUSTER_REPO}, but resolved ${REPO}." >&2
    echo "The resolved path will be used. Verify that it is visible on GPU nodes." >&2
fi

if [[ ! -f "${REPO}/rebuttal/README.md" ]]; then
    echo "ERROR: could not verify the repository root at ${REPO}." >&2
    exit 2
fi
if [[ ! -f "${WORKER}" ]]; then
    echo "ERROR: missing Slurm worker ${WORKER}." >&2
    exit 2
fi

# Step 2: Load the cluster Python module before inspecting or rebuilding the
# virtual environment.
module purge
module load python/3.11.14

PYTHON_BOOTSTRAP="$(command -v python3)"
echo "Bootstrap Python: ${PYTHON_BOOTSTRAP}"
"${PYTHON_BOOTSTRAP}" -V

cd "${REPO}"

# Step 3: Detect a copied virtual environment whose activation script still
# points to the obsolete repository location. Preserve it instead of deleting
# it, then create a relocatable environment at the current repository path.
VENV_DIR="${REPO}/.venv"
ACTIVATE_FILE="${VENV_DIR}/bin/activate"
CONFIGURED_VENV=""
if [[ -f "${ACTIVATE_FILE}" ]]; then
    CONFIGURED_VENV="$(sed -n 's/^VIRTUAL_ENV=//p' "${ACTIVATE_FILE}" | head -n 1)"
    CONFIGURED_VENV="${CONFIGURED_VENV%\"}"
    CONFIGURED_VENV="${CONFIGURED_VENV#\"}"
fi

if [[ -f "${ACTIVATE_FILE}" ]] && {
    [[ -z "${CONFIGURED_VENV}" ]] ||
    [[ "$(readlink -f "${CONFIGURED_VENV}")" != "$(readlink -f "${VENV_DIR}")" ]]
}; then
    BACKUP_DIR="${REPO}/.venv_old_location_backup_$(date -u +%Y%m%dT%H%M%SZ)"
    echo "Relocated virtual environment detected."
    echo "Moving it to: ${BACKUP_DIR}"
    mv -- "${VENV_DIR}" "${BACKUP_DIR}"
fi

if [[ ! -f "${ACTIVATE_FILE}" ]]; then
    echo "Creating virtual environment at ${VENV_DIR}"
    "${PYTHON_BOOTSTRAP}" -m venv "${VENV_DIR}"
fi

# Step 4: Activate and verify the corrected environment.
source "${ACTIVATE_FILE}"
hash -r

EXPECTED_PYTHON="${VENV_DIR}/bin/python"
ACTUAL_PYTHON="$(command -v python)"
if [[ "$(readlink -f "${ACTUAL_PYTHON}")" != "$(readlink -f "${EXPECTED_PYTHON}")" ]]; then
    echo "ERROR: expected ${EXPECTED_PYTHON}, found ${ACTUAL_PYTHON}." >&2
    exit 2
fi

echo "Virtual-environment Python: ${ACTUAL_PYTHON}"
python -V

# Step 5: Install all dependencies needed by the standalone experiments.
# This operation is idempotent. Set SKIP_INSTALL=1 only after the environment
# has already been prepared successfully.
if [[ "${SKIP_INSTALL:-0}" != "1" ]]; then
    echo "Updating packaging tools."
    python -m pip install --upgrade pip setuptools wheel
    echo "Installing rebuttal requirements."
    python -m pip install -r "${REPO}/rebuttal/requirements.txt"
else
    echo "SKIP_INSTALL=1: retaining the existing installed packages."
fi

# Step 6: Verify imports on the login node. CUDA availability is checked later
# inside the allocated smoke job.
python -c "
import datasets
import matplotlib
import normflows
import numpy
import scipy
import sklearn
import torch

print('PyTorch:', torch.__version__)
print('All required imports succeeded.')
"

# Step 7: Create directories before submission. Slurm opens log files before
# the job script starts, so slurm_logs must already exist.
mkdir -p \
    "${REPO}/slurm_logs" \
    "${REPO}/rebuttal/slurm_outputs" \
    "${REPO}/rebuttal/slurm_submissions"

chmod +x "${WORKER}"

# Step 8: Verify that Slurm submission is available and submit one smoke job.
# This job also downloads/stages SST-5 and ListOps exactly once.
if ! command -v sbatch >/dev/null 2>&1; then
    echo "ERROR: sbatch is not available on this host." >&2
    exit 2
fi

echo "Submitting smoke validation and dataset staging."
SMOKE_JOB="$(sbatch --parsable \
    --job-name=fska_smoke \
    --array=0-0 \
    --export=ALL,MODE=smoke \
    "${WORKER}")"

DEPENDENCY="afterok:${SMOKE_JOB}"
echo "Smoke job: ${SMOKE_JOB}"

# Step 9: Submit experiments 01--04 as five-task seed arrays. They remain
# pending until smoke validation and dataset staging complete successfully.
echo "Submitting SST-5 seed arrays."
J01="$(sbatch --parsable \
    --job-name=fska_01_sst5 \
    --dependency="${DEPENDENCY}" \
    --array=0-4 \
    --export=ALL,MODE=01 \
    "${WORKER}")"

J02="$(sbatch --parsable \
    --job-name=fska_02_density \
    --dependency="${DEPENDENCY}" \
    --array=0-4 \
    --export=ALL,MODE=02 \
    "${WORKER}")"

J03="$(sbatch --parsable \
    --job-name=fska_03_kl \
    --dependency="${DEPENDENCY}" \
    --array=0-4 \
    --export=ALL,MODE=03 \
    "${WORKER}")"

J04="$(sbatch --parsable \
    --job-name=fska_04_elu \
    --dependency="${DEPENDENCY}" \
    --array=0-4 \
    --export=ALL,MODE=04 \
    "${WORKER}")"

# Step 10: Submit the efficiency harness as one task so every timing cell uses
# the same allocated GPU.
echo "Submitting the efficiency benchmark."
J05="$(sbatch --parsable \
    --job-name=fska_05_efficiency \
    --dependency="${DEPENDENCY}" \
    --array=0-0 \
    --export=ALL,MODE=05 \
    "${WORKER}")"

# Step 11: Submit ListOps as separate primary and ablation seed arrays.
# These jobs receive a longer default limit because each task trains several
# models for 5,000 optimizer steps.
LISTOPS_TIME="${LISTOPS_TIME:-2-00:00:00}"
echo "Submitting ListOps arrays with time limit ${LISTOPS_TIME}."
J06_PRIMARY="$(sbatch --parsable \
    --job-name=fska_06_primary \
    --dependency="${DEPENDENCY}" \
    --array=0-4 \
    --time="${LISTOPS_TIME}" \
    --export=ALL,MODE=06p \
    "${WORKER}")"

J06_ABLATION="$(sbatch --parsable \
    --job-name=fska_06_ablation \
    --dependency="${DEPENDENCY}" \
    --array=0-2 \
    --time="${LISTOPS_TIME}" \
    --export=ALL,MODE=06a \
    "${WORKER}")"

# Step 12: Save every job ID so this submission can be monitored or cancelled
# without reconstructing IDs from terminal history.
SUBMISSION_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
MANIFEST="${REPO}/rebuttal/slurm_submissions/${SUBMISSION_STAMP}.env"
{
    echo "SMOKE_JOB=${SMOKE_JOB}"
    echo "J01=${J01}"
    echo "J02=${J02}"
    echo "J03=${J03}"
    echo "J04=${J04}"
    echo "J05=${J05}"
    echo "J06_PRIMARY=${J06_PRIMARY}"
    echo "J06_ABLATION=${J06_ABLATION}"
} > "${MANIFEST}"

# Step 13: Print the final submission summary and monitoring commands.
echo
echo "All jobs submitted."
echo "Job manifest: ${MANIFEST}"
echo "  Smoke:           ${SMOKE_JOB}"
echo "  Experiment 01:   ${J01}"
echo "  Experiment 02:   ${J02}"
echo "  Experiment 03:   ${J03}"
echo "  Experiment 04:   ${J04}"
echo "  Experiment 05:   ${J05}"
echo "  ListOps primary: ${J06_PRIMARY}"
echo "  ListOps ablation:${J06_ABLATION}"
echo
echo "Monitor:"
echo "  squeue -u \"${USER}\""
echo
echo "Detailed accounting:"
echo "  sacct -j ${SMOKE_JOB},${J01},${J02},${J03},${J04},${J05},${J06_PRIMARY},${J06_ABLATION} --format=JobID,JobName,State,Elapsed,ExitCode"
echo
echo "Smoke log:"
echo "  tail -f ${REPO}/slurm_logs/rebuttal_${SMOKE_JOB}_0.out"
echo
echo "Dependent arrays start only if smoke validation and data staging succeed."
