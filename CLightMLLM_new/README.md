# CLightMLLM

## Environment Setup

Create the Conda environment from `environment.yaml`:

```bash
conda env create -f environment.yaml
conda activate torch-2.8.0
```

## Optional: Install flash-attention

If you need flash-attention, first make sure the CUDA compiler from the active Conda environment is available:

```bash
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"

which nvcc
nvcc -V
```

Check the PyTorch and CUDA paths:

```bash
python - <<'PY'
import torch
from torch.utils.cpp_extension import CUDA_HOME

print("torch =", torch.__version__)
print("torch.version.cuda =", torch.version.cuda)
print("CUDA_HOME =", CUDA_HOME)
PY
```

Install flash-attention:

```bash
MAX_JOBS=4 pip install -v --no-cache-dir flash-attn --no-build-isolation
```

## Continual SFT Training

All training is expressed as continual SFT. A regular single-dataset SFT run is
a continual SFT run with exactly one stage:

```yaml
cl_sft:
  stages:
    - llava779k
trainer:
  save_dir: experiments/qwen3_vl/lora
```

For multi-stage continual SFT, list datasets in order. Each stage loads the
previous stage's exported HF model and writes its experiment artifacts under
`trainer.save_dir/<dataset_name>`. The HF model is exported to
`model/`, while W&B local files are saved under `wandb/`:

```yaml
method:
  name: lwf
  lwf_alpha: 1.0
  lwf_temperature: 2.0
cl_sft:
  stages:
    - llava_v1_5_mix665k_coco
    - llava_v1_5_mix665k_gqa
trainer:
  save_dir: experiments/qwen3_vl/lora/cl_sft_mix665k
```

Run one of the provided continual SFT configs:

```bash
python train.py --config config/continual_sft/qwen3_vl.yaml
```

When using LoRA across multiple stages, keep `trainer.export_hf_model_at_end`
and `trainer.merge_lora_before_export` enabled so the next stage can load a full
HF model.

## RL: GRPO Training

GRPO training uses the same stage/data/model stack. For each prompt, the
student samples multiple completions, scores them with the configured reward,
normalizes rewards within the group, and optimizes the sampled completion
tokens with a policy objective:

```yaml
method:
  name: grpo
  rollout_num_generations: 2
  rollout_max_new_tokens: 64
  grpo_reward_type: reference_match
  grpo_kl_coef: 0.0
```

Run the provided GRPO demo config:

```bash
conda activate torch-2.8.0
cd /path/to/CLightMLLM
python train.py --config config/demo2k/qwen2_vl_grpo.yaml
```

If `method.grpo_kl_coef > 0` or `method.grpo_reference_model=true`, the initial
stage model is also loaded as a frozen reference model for KL regularization.
The provided demo config follows the same experiment scale as the Qwen2-VL SFT
demo: `demo2k`, `max_epochs: 1`, `max_steps: -1`, per-device batch size 1, and
gradient accumulation 8. GRPO additionally samples multiple completions per
prompt, so it is slower than SFT at the same data scale.

## OPD Training

OPD training uses on-policy distillation. The student first samples completions
from the current policy, then a frozen teacher model is evaluated on those
student-sampled trajectories and provides token-level distillation signals:

```yaml
method:
  name: opd
  rollout_max_new_tokens: 64
  opd_teacher_model_name_or_path: /path/to/teacher_model
  opd_alpha: 1.0
  opd_temperature: 1.0
```

Run the provided OPD demo config:

```bash
conda activate torch-2.8.0
cd /path/to/CLightMLLM
python train.py --config config/demo2k/qwen2_vl_opd.yaml
```

Set `method.opd_teacher_model_name_or_path` to choose the teacher. Before
running, make sure both student and teacher model paths exist locally, and make
sure the `demo2k` dataset path in `config/dataset.json` is valid. The demo
config follows the same experiment scale as the Qwen2-VL SFT demo: `demo2k`,
`max_epochs: 1`, `max_steps: -1`, per-device batch size 1, and gradient
accumulation 8. OPD additionally runs the frozen teacher on sampled trajectories,
so it is slower than SFT at the same data scale.

## Required Local Files

The demo configs use local paths and set `model.local_files_only: true`, so
training will not download the model automatically. Before running SFT, GRPO, or
OPD demo configs, make sure these paths exist or update the YAML files:

```yaml
model:
  model_name_or_path: /ppio_net0/download/Qwen2-VL-2B-Instruct
```

For OPD, the teacher path must also exist:

```yaml
method:
  opd_teacher_model_name_or_path: /ppio_net0/download/Qwen2-VL-2B-Instruct
```

The `demo2k` dataset is configured in `config/dataset.json`:

```json
{
  "demo2k": {
    "file_name_or_path": "/ppio_net0/datasets/parquet/llava779k_demo2k"
  }
}
```

## Quick Runtime Check

Use this short check before any demo run:

```bash
python - <<'PY'
import torch
import lightning
import transformers
import datasets
import peft

print("torch:", torch.__version__, torch.cuda.is_available())
print("lightning:", lightning.__version__)
print("transformers:", transformers.__version__)
print("datasets:", datasets.__version__)
print("peft:", peft.__version__)
PY
```
