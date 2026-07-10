#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/work/03/gw42/j40004/cj/opd-reproduction/CLightMLLM_new}"
STUDENT_MODEL="${STUDENT_MODEL:-/work/03/gw42/j40004/models/Qwen3-VL-2B-Instruct}"
TEACHER_MODEL="${TEACHER_MODEL:-/work/03/gw42/j40004/models/Qwen3-VL-8B-Instruct}"
SRC_DIR="${SRC_DIR:-/work/03/gw42/j40004/cj/opd_dumps/qwen3_vl_2b_from_8b_geo3k_layerA_3epoch_swanlab_dumpALL_3gpu_mbs12_chunk12}"
OUT_DIR="${OUT_DIR:-/work/03/gw42/j40004/cj/opd_dumps/layerC_smoke_student_rollout_teacher_rescore}"

TEACHER_GPU="${TEACHER_GPU:-1}"
STUDENT_GPU="${STUDENT_GPU:-0}"
FSDP_GPUS="${FSDP_GPUS:-0,1,2}"
FSDP_NPROC="${FSDP_NPROC:-3}"
TEACHER_PORT="${TEACHER_PORT:-29577}"
MAX_FILES="${MAX_FILES:-2}"

cd "$REPO_DIR"

export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export VLLM_USE_V1=1
export LIBRARY_PATH=/usr/local/cuda-12.4/compat:${LIBRARY_PATH:-}
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/compat:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/.singularity.d/libs
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export RAYON_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export MALLOC_ARENA_MAX=1
export UV_THREADPOOL_SIZE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$OUT_DIR" "$OUT_DIR/layerC_traces"

echo "== Layer C smoke =="
echo "repo=$REPO_DIR"
echo "src=$SRC_DIR"
echo "out=$OUT_DIR"
echo "teacher_gpu=$TEACHER_GPU student_gpu=$STUDENT_GPU fsdp_gpus=$FSDP_GPUS"
echo "max_files=$MAX_FILES"

echo "== start teacher server =="
CUDA_VISIBLE_DEVICES="$TEACHER_GPU" python tools/serve_vllm_teacher.py \
  --model "$TEACHER_MODEL" \
  --host 127.0.0.1 \
  --port "$TEACHER_PORT" \
  --topk 32 \
  --torch-dtype bfloat16 \
  --gpu-memory-utilization 0.80 \
  --max-model-len 1537 \
  --max-logprobs 32 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 1024 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --enforce-eager \
  --disable-log-stats \
  --local-files-only \
  > "$OUT_DIR/teacher_server.log" 2>&1 &
TEACHER_PID=$!

cleanup_teacher() {
  if kill -0 "$TEACHER_PID" 2>/dev/null; then
    kill "$TEACHER_PID" 2>/dev/null || true
    wait "$TEACHER_PID" 2>/dev/null || true
  fi
}
trap cleanup_teacher EXIT

python - "$TEACHER_PORT" <<'PY'
import socket
import sys
import time

port = int(sys.argv[1])
deadline = time.time() + 900
while time.time() < deadline:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=2):
            print("teacher server ready", flush=True)
            break
    except OSError:
        time.sleep(5)
else:
    raise SystemExit("teacher server did not become ready in 900 seconds")
PY

echo "== build Layer C smoke traces =="
CUDA_VISIBLE_DEVICES="$STUDENT_GPU" python tools/build_layerC_smoke_trace.py \
  "$SRC_DIR"/verl_opd_trace_dump*.pt \
  --student-model "$STUDENT_MODEL" \
  --output-dir "$OUT_DIR/layerC_traces" \
  --host 127.0.0.1 \
  --port "$TEACHER_PORT" \
  --topk 32 \
  --max-files "$MAX_FILES" \
  --student-micro-batch-size 4 \
  --teacher-micro-batch-size 1 \
  --max-response-length 512 \
  --max-model-len 1537 \
  --gpu-memory-utilization 0.35 \
  --dtype bfloat16 \
  --temperature 1.0 \
  --top-p 1.0 \
  --top-k -1 \
  --enforce-eager \
  --disable-log-stats \
  --metrics-output "$OUT_DIR/layerC_smoke_metrics.jsonl" \
  2>&1 | tee "$OUT_DIR/build_layerC_smoke.log"

cleanup_teacher
trap - EXIT

echo "== replay/train Layer C smoke traces =="
CUDA_VISIBLE_DEVICES="$FSDP_GPUS" torchrun --standalone --nproc_per_node="$FSDP_NPROC" \
  tools/replay_verl_opd_trace_fsdp.py \
  --config config/geometry3k/qwen3_vl_layerA_replay_2b_fp32student_work.yaml \
  "$OUT_DIR"/layerC_traces/verl_opd_trace_dump*.pt \
  --micro-batch-size 1 \
  --train \
  --learning-rate 1e-6 \
  --grad-clip 1.0 \
  --amp-dtype bf16 \
  --position-ids-mode none \
  --teacher-shift-offset -1 \
  --gradient-checkpointing \
  --metrics-output "$OUT_DIR/train_smoke_metrics.jsonl" \
  2>&1 | tee "$OUT_DIR/train_smoke.log"

echo "== done =="
find "$OUT_DIR" -maxdepth 2 -type f | sort
