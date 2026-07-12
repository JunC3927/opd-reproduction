# OPD Layer A/B progress and Layer C plan

This note records the current Qwen3-VL OPD reproduction status across VERL and
CLight. The main experiment is Qwen3-VL-2B student distilled from
Qwen3-VL-8B teacher on Geo3K.

## Current conclusion

We have validated two useful replay layers:

- Layer A works: CLight replays VERL dumps using VERL-provided
  `teacher_ids` and `teacher_logprobs`, then performs student forward, top-k KL
  loss, backward, and optimizer step.
- Layer B works: CLight reconstructs the VERL teacher vLLM request, recomputes
  teacher top-32 ids/logprobs with its own vLLM teacher, writes new trace files,
  and trains the student from those traces.
- Both Layer A and Layer B improved Geo3K validation accuracy by about 7 points
  over the base 2B model in the current runs.

This means the core CLight OPD replay path is no longer just runnable; it has
produced a measurable validation improvement.

## Main paths

Server paths used in the A100 experiments:

```text
repo root:
  /work/03/gw42/j40004/cj/opd-reproduction

VERL:
  /work/03/gw42/j40004/cj/opd-reproduction/verl_new

CLight:
  /work/03/gw42/j40004/cj/opd-reproduction/CLightMLLM_new

student model:
  /work/03/gw42/j40004/models/Qwen3-VL-2B-Instruct

teacher model:
  /work/03/gw42/j40004/models/Qwen3-VL-8B-Instruct

Geo3K train:
  /work/03/gw42/j40004/cj/data/geo3k/train.parquet

Geo3K test:
  /work/03/gw42/j40004/cj/data/geo3k/test.parquet
```

Important dump/model outputs:

```text
Layer A VERL trace dumps:
  /work/03/gw42/j40004/cj/opd_dumps/qwen3_vl_2b_from_8b_geo3k_layerA_3epoch_swanlab_dumpALL_3gpu_mbs12_chunk12

Layer A CLight trained model:
  /work/03/gw42/j40004/cj/opd_dumps/clight_layerA_fsdp3_full_saved/hf_model

Layer B CLight trained model:
  /work/03/gw42/j40004/cj/opd_dumps/clight_layerB_fsdp3_full_saved/hf_model
```

## Layer A

Layer A asks whether CLight can reproduce the training update if the teacher
top-k distribution is already dumped by VERL.

Input per trace file:

```text
responses
teacher_ids
teacher_logprobs
attention_mask
response_mask
position_ids
multi_modal_inputs / multi_modal_data
```

CLight does:

```text
HF/FSDP student forward
-> gather student logits at teacher top-k ids
-> forward KL top-k loss
-> backward
-> gradient clip
-> optimizer step
```

The successful full run used:

```text
TRAIN_BATCH_SIZE=24
PPO_MINI_BATCH_SIZE=12
```

Therefore each VERL global step was split into two chunk files:

```text
261 global steps * 2 chunks = 522 trace files
12 samples per trace file
```

This is intentional. Each chunk corresponds to one mini-batch update used by
CLight replay.

Layer A result:

- CLight loss decreased clearly.
- SwanLab curves were uploaded.
- The saved model improved Geo3K validation accuracy by about 7 points over
  base.

## Layer B

Layer B asks whether CLight can replace VERL's teacher top-k computation with
its own vLLM teacher call.

The student side stays unchanged from the trace:

```text
input_ids
responses
attention_mask
response_mask
position_ids
images
```

Only these fields are replaced:

```text
teacher_ids
teacher_logprobs
```

### Verified VERL teacher input path

The VERL teacher does not generate a new answer. It scores the already-generated
student trajectory:

```text
sequence_ids = prompt_ids + response_ids
```

The teacher request is effectively:

```python
client.generate(
    prompt_ids=sequence_ids,
    sampling_params={
        "max_tokens": 1,
        "prompt_logprobs": 32,
        "temperature": 1.0,
    },
    image_data=multi_modal_data.get("images"),
    mm_processor_kwargs=mm_processor_kwargs,
)
```

For Qwen3-VL, the server then converts this request into a final vLLM
`TokensPrompt`. During that conversion, consecutive image placeholder tokens are
deduplicated. vLLM expands them internally using the actual image data.

The canonical image source for reconstruction is:

```text
non_tensor_batch["multi_modal_data"]
```

`vllm_images` is useful for inspection, but it is not the canonical source.

### Layer B verification already done

We verified the chain with a one-step smoke dump:

```text
/work/03/gw42/j40004/cj/opd_dumps/layerB_teacherio_smoke_v2
```

Checks performed:

- Reconstructed teacher requests from trace matched VERL teacher-manager
  requests after accounting for async row order.
- Reconstructed teacher requests converted to final vLLM prompts matched VERL
  final prompt dumps.
- CLight vLLM teacher rescoring produced response-token top-32 set overlap
  around 0.99 against VERL teacher top-32.

Representative Layer B rescore smoke result:

```text
mean_resp_set_overlap ~= 0.990 - 0.993
mean_resp_logps_abs   ~= 0.06 - 0.09
```

This is not bitwise identical, but it is close enough that the downstream Layer B
training produced the same scale of validation improvement as Layer A.

### Important interpretation

Layer B does not require the new top-32 ids to be exactly the same as VERL's old
top-32 ids. In Layer B, the teacher target is the newly computed CLight-vLLM
teacher distribution. The student loss gathers student logits at the new
teacher-selected ids.

The old VERL teacher ids/logprobs are used only as an alignment diagnostic.

## Current CLight tools

The stable LayerC online OPD path now keeps only the tools needed for the
current reproducible workflow:

```text
replay_verl_opd_trace.py
replay_verl_opd_trace_fsdp.py
serve_vllm_teacher.py
train_online_hf_opd_fsdp.py
upload_replay_metrics_to_swanlab.py
```

For the current online LayerC runs, use:

```text
tools/train_online_hf_opd_fsdp.py
```

The `replay_verl_opd_trace*.py` files are still kept because the online trainer
reuses their optimizer, loss, dtype, FSDP, and trace helper functions.

Older one-off compare/rescore/probe scripts were removed after the trace and
Geo3K parquet paths both reproduced the expected LayerC behavior.

## Stable A100 environment settings

The A100 environment has a strict process/thread limit. Use conservative thread
settings before Ray, vLLM, or torchrun:

```bash
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
```

For vLLM on the A100 container, also use:

```bash
export VLLM_USE_V1=1
export LIBRARY_PATH=/usr/local/cuda-12.4/compat:${LIBRARY_PATH:-}
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/compat:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/.singularity.d/libs
```

## Known issues solved

- `ulimit -u=2048` caused Ray/vLLM thread exhaustion. The solution was to pin
  thread pools to one thread.
- `PPO_MINI_BATCH_SIZE=24` was unstable in the current environment. The stable
  setup was `TRAIN_BATCH_SIZE=24` and `PPO_MINI_BATCH_SIZE=12`.
- Single-card CLight replay OOMed with the 2B model in fp32 training. FSDP over
  3 A100 cards works.
- SwanLab cloud upload was unstable from the server network. Local logdirs can
  be uploaded later.
- vLLM tokenizer warning on saved CLight models should be checked by comparing
  tokenizer outputs between the base 2B model and the saved model. If token ids
  match, the warning can be treated as non-blocking for eval.

## Layer C proposal

Layer C should remove the remaining dependency on VERL-generated rollouts.

Recommended definition:

```text
Layer A:
  VERL responses + VERL teacher top-k -> CLight trains student

Layer B:
  VERL responses + CLight-vLLM teacher top-k -> CLight trains student

Layer C:
  CLight-generated responses + CLight-vLLM teacher top-k -> CLight trains student
```

Layer C is the first layer where CLight owns both:

- student rollout generation
- teacher top-k computation

There are two possible versions.

### Layer C-lite

Generate student responses once using the initial 2B model, rescore those
responses with the 8B teacher, then train CLight offline.

This tests:

```text
Geo3K raw sample -> CLight prompt/image preprocessing
-> CLight student vLLM rollout
-> CLight teacher vLLM top-k
-> CLight FSDP replay training
```

It does not test online weight sync from FSDP back to rollout vLLM.

### Layer C-full

Build the full online CLight OPD loop:

```text
sample batch
-> student vLLM rollout with current weights
-> teacher vLLM top-k
-> student FSDP update
-> sync updated student weights back to rollout vLLM
-> next batch
```

This is closest to VERL OPD, but it is also the most engineering-heavy step.

## Recommended next step

Run Layer C-lite first.

Reason:

- Layer A already validates CLight training math.
- Layer B already validates CLight teacher vLLM scoring.
- The biggest untested part is now CLight's own conversion from raw Geo3K sample
  to student rollout request and response trace.

Layer C-lite gives a clean next target without immediately adding online vLLM
weight sync complexity.

After Layer C-lite works and improves validation, move to Layer C-full.
