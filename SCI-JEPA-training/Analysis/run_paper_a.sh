#!/bin/bash
#SBATCH --job-name=paper_a
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gres=gpu:l40s:1
#SBATCH --partition=p_48G
#SBATCH --output=/nfs/home/rabbyg/JEPA/Graph-JEPA/log/paper_a_%j.out
#SBATCH --error=/nfs/home/rabbyg/JEPA/Graph-JEPA/log/paper_a_%j.err
#
# Paper A runner. Works on a MacBook (local mode) and on the cluster (sbatch).
#
# LOCAL (MacBook), from this folder:
#   bash run_paper_a.sh setup            # once: venv + packages + Jupyter kernel "paper-a"
#   bash run_paper_a.sh smoke            # synthetic data, ~1 min (add --full: every arm)
#   bash run_paper_a.sh                  # 5,000 papers: build KU graph, shape audit, E17 (3 seeds)
#   N_PAPERS=20000 EXPS="E10 E11" EPOCHS=10 bash run_paper_a.sh
#   SIZE=S DEVICE=cpu bash run_paper_a.sh
#
# CLUSTER:
#   sbatch run_paper_a.sh                # D0 (60k papers), size S, 5 seeds, the ICML experiments
#   EXPS="E1 E2 E3" sbatch run_paper_a.sh
#
# Every knob below can be overridden from the environment. EXTRA passes any other
# PaperAConfig field, e.g. EXTRA="split_mode=random lambda_inv=0.3".
set -uo pipefail

if [[ -n "${SLURM_JOB_ID:-}" ]]; then MODE=cluster; else MODE=local; fi
CMD="${1:-run}"                      # setup | smoke | build | audit | run

if [[ "${MODE}" == "cluster" ]]; then
  PROJECT_DIR="${PROJECT_DIR:-/nfs/home/rabbyg/JEPA/Graph-JEPA}"
  DATA_ROOT="${DATA_ROOT:-${PROJECT_DIR}}"
  N_PAPERS="${N_PAPERS:-60000}"
  ROWS_PER_SHARD="${ROWS_PER_SHARD:-500}"
  SIZE="${SIZE:-S}"
  EPOCHS="${EPOCHS:-10}"
  BATCH="${BATCH:-128}"
  SEEDS="${SEEDS:-[0,1,2,3,4]}"
  EXPS="${EXPS:-E10 E11 E12 E4 E5 E15 E17 E18}"   # the ICML experiments (E1-E3 wait for real shapes)
  EVAL_N="${EVAL_N:-4000}"
  DEVICE="${DEVICE:-cuda}"
else
  # this script lives in SCI-JEPA-training/Analysis; the data sits next to the repo in Alex/
  PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
  DATA_ROOT="${DATA_ROOT:-$(cd "${PROJECT_DIR}/../.." && pwd)}"
  N_PAPERS="${N_PAPERS:-5000}"
  ROWS_PER_SHARD="${ROWS_PER_SHARD:-100}"
  SIZE="${SIZE:-local}"              # d=256, 4 encoder layers: ~0.4 s/step on MPS at N=8
  EPOCHS="${EPOCHS:-10}"
  BATCH="${BATCH:-64}"
  SEEDS="${SEEDS:-[0,1,2]}"
  EXPS="${EXPS:-E17}"
  EVAL_N="${EVAL_N:-1000}"
  DEVICE="${DEVICE:-auto}"           # auto: Apple GPU (mps) when available, else cpu
fi
EXTRA="${EXTRA:-}"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PAPER_A_PARQUET="${PAPER_A_PARQUET:-${DATA_ROOT}/Scientific-Summaries-Data/normalized/arxiv}"
export PAPER_A_INDEX_DIR="${PAPER_A_INDEX_DIR:-${PROJECT_DIR}/paper_a_data}"
export PAPER_A_OUT="${PAPER_A_OUT:-${PROJECT_DIR}/paper_a_results}"
export HF_HOME="${HF_HOME:-${PROJECT_DIR}/.hf_cache}"
export PYTORCH_ENABLE_MPS_FALLBACK=1     # any op MPS lacks runs on CPU instead of failing
mkdir -p "${PAPER_A_INDEX_DIR}" "${PAPER_A_OUT}"

# ---------------------------------------------------------------- environment
VENV="${PROJECT_DIR}/.venv-paper-a"
if [[ "${MODE}" == "cluster" ]]; then
  source /nfs/home/rabbyg/miniconda3/etc/profile.d/conda.sh
  conda activate kgjepa || { echo "conda activate failed"; exit 1; }
  cd "${PROJECT_DIR}" || exit 1
  PY=python
else
  cd "${PROJECT_DIR}" || exit 1
  if [[ "${CMD}" == "setup" || ! -x "${VENV}/bin/python" ]]; then
    echo "== creating ${VENV} =="
    "${PYTHON:-python3}" -m venv "${VENV}" || exit 1
    "${VENV}/bin/pip" install -q --upgrade pip
    "${VENV}/bin/pip" install -q -r requirements.txt || { echo "pip install failed"; exit 1; }
    "${VENV}/bin/python" -m ipykernel install --user --name paper-a --display-name "Python (paper-a)"
    echo "== downloading the text encoder once (gte-modernbert-base, 8192-token context, ~600 MB) =="
    HF_HOME="${HF_HOME:-${PROJECT_DIR}/.hf_cache}" "${VENV}/bin/python" -c "from sentence_transformers import SentenceTransformer as S; S('Alibaba-NLP/gte-modernbert-base')"
    [[ "${CMD}" == "setup" ]] && { echo "setup done. Open paper_a_local.ipynb with kernel 'Python (paper-a)'."; exit 0; }
  fi
  PY="${VENV}/bin/python"
fi

if [[ ! -d "${PAPER_A_PARQUET}" && "${CMD}" != "smoke" ]]; then
  echo "ERROR: parquet folder not found: ${PAPER_A_PARQUET}"
  echo "       set PAPER_A_PARQUET=/path/to/Scientific-Summaries-Data/normalized/arxiv"
  exit 1
fi

DATA="max_papers=${N_PAPERS} rows_per_shard=${ROWS_PER_SHARD} device=${DEVICE}"
TRAIN="epochs=${EPOCHS} batch_size=${BATCH} seeds=${SEEDS} eval_instances=${EVAL_N}"

echo "================================================================"
echo " Paper A | mode=${MODE} cmd=${CMD} | $(date)"
echo " data   : ${PAPER_A_PARQUET}"
echo " papers : ${N_PAPERS} (${ROWS_PER_SHARD} rows/shard) | size ${SIZE} | device ${DEVICE}"
echo " train  : ${EPOCHS} epochs, batch ${BATCH}, seeds ${SEEDS} | exps: ${EXPS}"
echo " extra  : ${EXTRA:-<none>}"
echo " output : ${PAPER_A_OUT}"
echo "================================================================"

RC=0
case "${CMD}" in
  smoke)
    "${PY}" -m train.paper_a.smoke_test "${@:2}" || RC=$? ;;
  build)
    "${PY}" -m train.paper_a.ku_graph --set ${DATA} ${EXTRA} || RC=$? ;;
  audit)
    "${PY}" -m train.paper_a.train --audit-shapes --size "${SIZE}" --set ${DATA} ${EXTRA} || RC=$? ;;
  run|setup)
    # 1. KU graph for this tier (cached; rebuilt only when data settings change)
    "${PY}" -m train.paper_a.ku_graph --set ${DATA} ${EXTRA} || { echo "KU graph build failed"; exit 1; }
    # 2. Theorem A2 audit of the shape vocabulary (no training)
    "${PY}" -m train.paper_a.train --audit-shapes --size "${SIZE}" --set ${DATA} ${EXTRA} || RC=$?
    # 3. experiments
    for E in ${EXPS}; do
      echo "################ ${E} ################"
      "${PY}" -m train.paper_a.train --exp "${E}" --size "${SIZE}" --set ${DATA} ${TRAIN} ${EXTRA} || RC=$?
    done ;;
  *)
    echo "unknown command '${CMD}' (setup | smoke | build | audit | run)"; exit 2 ;;
esac
echo "done rc=${RC}; results under ${PAPER_A_OUT}/<exp>/summary.json"
exit ${RC}
