# LayerC Online OPD Worklog

这份文档用于新开对话时快速恢复上下文，也方便回顾当前 LayerC online OPD 实验的代码状态、实验假设、命令和容易踩的坑。

## 当前目标

我们想在 CLightMLLM 里复现/推进一个更接近 VERL 的 online OPD 训练流程：

- student 是 Qwen3-VL-2B。
- teacher 是 Qwen3-VL-8B，通过 vLLM 服务计算 top-k logprob。
- student 参数和 optimizer state 需要 fp32 更新。
- student forward/generate 可以用 bf16 autocast。
- 训练需要 FSDP，因为 A100 40GB 显存比较紧。
- 每 12 个样本做一次 update。
- 最终希望跑 3 epoch，并产出 `metrics.jsonl`，后续同步到 SwanLab。

核心原则是：response 必须随着 student 更新而变化，也就是 on-policy。最稳的路径是 HF/FSDP student rollout；更快但仍在调试的是 student vLLM on-policy rollout 加热更新权重。

## 已经完成的主要工作

### 1. Teacher vLLM 服务

Teacher 端使用 8B Qwen3-VL，通过 vLLM 接收多个请求并返回 top32 id/logprob。

配置方向和之前 VERL teacher 对齐：

```text
model=/work/03/gw42/j40004/models/Qwen3-VL-8B-Instruct
tokenizer=/work/03/gw42/j40004/models/Qwen3-VL-8B-Instruct
dtype=torch.bfloat16
max_seq_len=1537
tensor_parallel_size=1
pipeline_parallel_size=1
data_parallel_size=1
enforce_eager=True
device_config=cuda
seed=0
enable_prefix_caching=True
chunked_prefill_enabled=True
skip_tokenizer_init=False
tokenizer_mode=auto
load_format=auto
kv_cache_dtype=auto
quantization=None
```

Teacher 输入里的图片要从 trace/parquet 里保留的原图来，而不是随便重新构造。对 vLLM teacher 来说，我们传的是 prompt token ids 加 `multi_modal_data` 中的 PIL image。代码里也保留了连续 image token dedup 逻辑，避免 Qwen3-VL/vLLM 对连续 image placeholder 的不一致处理。

### 2. Stable 路径：HF/FSDP online rollout

脚本：

```text
CLightMLLM_new/tools/train_online_hf_opd_fsdp.py
```

稳定方向是：

1. 读取 VERL trace `.pt` 或 Geo3K parquet。
2. 用当前 FSDP student 生成 response。
3. 把 prompt + response 发给 teacher vLLM 打 top-k logprob。
4. student 用 teacher top-k 分布算 OPD loss。
5. FSDP 反传，12 个样本 update 一次。
6. 写 `metrics.jsonl`，可选保存 HF model。

之前已经看到过正常输出，例如：

```text
step=1 epoch=0 samples=12 loss=...
teacher_mass=...
student_mass=...
topk_overlap=...
grad_norm=...
resp_len_mean=...
```

这说明 HF/FSDP 路径至少能进入训练并产出 loss/metrics。

### 3. dtype 结论

当前目标状态是：

```text
FSDP 参数: torch.float32
optimizer state: torch.float32
forward logits: torch.bfloat16
loss/micro_loss: torch.float32
```

也就是 fp32 参数更新 + bf16 forward。这和 VERL 的常见策略一致：训练主权重保持 fp32，rollout 或 forward 可以 bf16。vLLM rollout 如果使用 bf16 权重副本，本质上也是拿 fp32 student 当前权重同步/转换成 bf16 后生成，通常是合理的。

需要注意：如果对齐到极致，bf16 rollout 和 fp32 forward 数值不会完全一模一样，但它是常见的效率/显存折中。

### 4. FSDP update 方式

FSDP 不是把所有梯度都集中到某一个 rank 再更新完整模型。它的直觉是：

- 每张卡只持有参数 shard。
- forward/backward 时按需 all-gather 当前层参数。
- backward 后梯度 reduce-scatter 回各自 shard。
- optimizer 在每个 rank 上更新自己持有的参数 shard。

所以 12 个样本 update 一次时，4 张 student 卡的情况大致是：

```text
global batch/update = 12
world_size = 4
local samples/rank = 3
每个 rank 算自己 3 个样本的 loss/grad
FSDP 做跨 rank 梯度同步
每个 rank 更新自己的参数 shard
```

### 5. VERL trace `.pt` 和 parquet 的关系

parquet 是原材料，trace `.pt` 是 VERL 已经处理好的 batch 中间产物。

parquet 里大概是：

```python
{
    "prompt": [
        {
            "role": "user",
            "content": "<image>\nFind the angle x."
        }
    ],
    "images": [PIL.Image 或 image bytes],
    "data_source": "geo3k",
    "extra_info": {"answer": "..."}
}
```

VERL 会做：

1. `RLHFDataset._build_messages()` 把 `<image>` 替换成结构化 image content。
2. `apply_chat_template()` 生成 Qwen3-VL chat prompt。
3. `build_multimodal_processor_inputs()` 同时处理文本和图片。
4. 得到 `input_ids`、`attention_mask`、`pixel_values`、`image_grid_thw`。
5. 再补 `images_seqlens`。
6. batch/pad 成 trace 里的 tensor 和 non-tensor metadata。

所以 trace 里能看到：

```python
non_tensor_batch["multi_modal_inputs"].keys()
# pixel_values, image_grid_thw, images_seqlens
```

其中：

```python
images_seqlens = image_grid_thw[:, 1] * image_grid_thw[:, 2]
```

意思是每张图片对应多少个视觉 patch/token。比如：

```python
image_grid_thw = [[1, 12, 26]]
images_seqlens = [312]
pixel_values.shape[0] = 312
```

### 6. Geo3K parquet 数据源

现在 `train_online_hf_opd_fsdp.py` 增加了：

```text
--data-source trace|geo3k_parquet
```

最初写过一版手写 Geo3K adapter，但后来确认这个方向不够 VERL-native，已经改成直接复用 VERL 的处理路线：

- `RLHFDataset`
- `verl.utils.chat_template.apply_chat_template`
- `verl.utils.tokenizer.build_multimodal_processor_inputs`
- `verl.utils.tokenizer.normalize_token_ids`

重要注意点：

- parquet prompt 必须像 VERL 原始数据一样包含 `<image>` 占位。
- 当前代码不会自动补 `<image>`。
- 如果 parquet 里图片数量和 `<image>` 数量对不上，会像 VERL 一样报错。
- 这比自动猜测更安全，因为可以及早暴露数据格式偏差。

可用参数：

```text
--parquet-prompt-key prompt
--parquet-image-key images
--parquet-max-rows 12
--parquet-max-prompt-length 1024
--parquet-filter-overlong-prompts / --no-parquet-filter-overlong-prompts
--parquet-mm-processor-kwargs '{"max_pixels":1048576}'
```

如果不显式设置 `--parquet-max-prompt-length`，默认会用：

```text
student_vllm_max_model_len - response_width
```

例如 `1537 - 512 = 1025`。

## 当前重要 commits

这些 commit 还需要 push 到 GitHub，然后服务器 pull：

```text
e0c847c Align Geo3K parquet source with VERL preprocessing
a7581a3 Add Geo3K parquet source for online OPD
b367777 Start student vLLM before FSDP distributed init
b3a5ee1 Disable student vLLM V1 multiprocessing by default
cde3d97 Align dedicated student vLLM device for IPC sync
508fc9e Allow dedicated GPU for student vLLM rollout
08834c2 Isolate student vLLM init from torchrun env
137b08d Synchronize rank0 vLLM init before FSDP rollout
```

本地曾经 push 失败，原因是 Windows GitHub credential：

```text
schannel: AcquireCredentialsHandle failed: SEC_E_NO_CREDENTIALS
```

在有 GitHub 凭据的 PowerShell 里执行：

```powershell
cd D:\opd_again\opd_full_github_repo
git push origin main
```

服务器上再：

```bash
cd /work/03/gw42/j40004/cj/opd-reproduction
git pull origin main
```

如果服务器提示本地改动会被覆盖，要先确认那些改动是不是需要保留。之前遇到过：

```text
error: Your local changes to the following files would be overwritten by merge:
        CLightMLLM_new/tools/train_online_hf_opd_fsdp.py
```

最稳做法是先备份或 stash，再 pull。

## 推荐运行方式

### Teacher 服务

一张卡给 teacher，例如 GPU 0 或单独一张空卡：

```bash
CUDA_VISIBLE_DEVICES=0 python CLightMLLM_new/tools/serve_vllm_teacher.py \
  --model /work/03/gw42/j40004/models/Qwen3-VL-8B-Instruct \
  --host 127.0.0.1 \
  --port 29577 \
  --topk 32 \
  --no-trust-remote-code \
  --torch-dtype bfloat16 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.80 \
  --max-model-len 1537 \
  --max-logprobs 32 \
  --load-format auto \
  --seed 0 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --enforce-eager \
  --local-files-only
```

### Stable HF/FSDP rollout from VERL trace

4 张卡给 student：

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nproc_per_node=4 \
  CLightMLLM_new/tools/train_online_hf_opd_fsdp.py \
  --config CLightMLLM_new/config/geometry3k/qwen3_vl_layerA_replay_2b_fp32student_work.yaml \
  --data-source trace \
  /work/03/gw42/j40004/cj/opd_dumps/qwen3_vl_2b_from_8b_geo3k_layerA_3epoch_swanlab_dumpALL_3gpu_mbs12_chunk12/verl_opd_trace_dump*.pt \
  --teacher-host 127.0.0.1 \
  --teacher-port 29577 \
  --samples-per-update 12 \
  --response-width 512 \
  --epochs 1 \
  --micro-batch-size 1 \
  --learning-rate 1e-6 \
  --grad-clip 1.0 \
  --generate-amp-dtype bf16 \
  --train-amp-dtype bf16 \
  --gradient-checkpointing \
  --debug-dtypes \
  --rollout-backend manual_cache \
  --metrics-output /work/03/gw42/j40004/cj/opd_dumps/layerC_online_hf_fsdp/metrics.jsonl \
  --save-model-dir /work/03/gw42/j40004/cj/opd_dumps/layerC_online_hf_fsdp/hf_model
```

如果只 smoke test，建议加：

```text
--max-updates 1
```

### HF/FSDP rollout from Geo3K parquet

parquet 路径用于绕过 trace dump，直接从 Geo3K 原始数据开始：

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nproc_per_node=4 \
  CLightMLLM_new/tools/train_online_hf_opd_fsdp.py \
  --config CLightMLLM_new/config/geometry3k/qwen3_vl_layerA_replay_2b_fp32student_work.yaml \
  --data-source geo3k_parquet \
  /work/03/gw42/j40004/data/geo3k/train.parquet \
  --parquet-max-rows 12 \
  --teacher-host 127.0.0.1 \
  --teacher-port 29577 \
  --samples-per-update 12 \
  --response-width 512 \
  --epochs 1 \
  --max-updates 1 \
  --micro-batch-size 1 \
  --learning-rate 1e-6 \
  --grad-clip 1.0 \
  --generate-amp-dtype bf16 \
  --train-amp-dtype bf16 \
  --gradient-checkpointing \
  --debug-dtypes \
  --rollout-backend manual_cache \
  --metrics-output /work/03/gw42/j40004/cj/opd_dumps/layerC_online_geo3k_parquet_smoke/metrics.jsonl \
  --save-model-dir /work/03/gw42/j40004/cj/opd_dumps/layerC_online_geo3k_parquet_smoke/hf_model
```

如果 parquet 真实路径不同，替换最后的 parquet 文件路径即可。

## Student vLLM on-policy rollout

我们也探索了 student vLLM on-policy：

- 先启动 vLLM student。
- 从 HF/FSDP student 同步权重到 vLLM。
- 用 vLLM 一次 rollout 当前 12 个样本。
- 再用 teacher vLLM 打分。
- 最后 HF/FSDP student 做 OPD update。

已经验证过一个关键点：vLLM 可以通过 `llm.apply_model` 走权重更新路径。bucketed IPC 版本也曾跑出：

```text
weight_update_path = llm.apply_model_ipc
RESULT=OK
```

但是集成到 torchrun/FSDP 训练时，仍然遇到过 vLLM EngineCore 和 torch distributed/c10d/NCCL 环境互相干扰、卡住或 timeout 的问题。我们后来做过一些修复方向：

- vLLM student 尽量在 FSDP distributed init 前启动。
- 清理 torchrun 环境变量，避免 vLLM 子进程继承 `RANK/WORLD_SIZE/MASTER_*`。
- 支持 `--student-vllm-device cuda:4` 这种 dedicated GPU。
- 对 A100 40GB，rank0 同卡放 FSDP shard + vLLM student 很紧张，不推荐。

当前结论：

- HF/FSDP `manual_cache` 是优先稳定路径。
- student vLLM IPC 是实验路径，后续还需要继续修。
- 如果有第 5 张空卡给 student vLLM，会更合理。
- 如果只有 4 张 student 卡，rank0 同时放 vLLM 容易 OOM 或卡住。

## 常见报错和处理

### 1. vLLM / c10d IPv6 warning

常见：

```text
The client socket cannot be initialized to connect to [::ffff:...]
Address family not supported by protocol
```

这类 warning 不一定致命。如果后面继续跑出 step/loss，通常可以先忽略。真正致命的是长时间卡住或出现 timeout。

### 2. NCCL collective timeout

例如：

```text
Watchdog caught collective operation timeout
OpType=ALLGATHER
```

可能原因：

- 某个 rank 死掉或卡住。
- vLLM EngineCore 继承了 torchrun distributed 环境。
- 不同 rank 进入 collective 的顺序不一致。
- 生成/teacher 请求某个 rank 卡住，其他 rank 等待。

遇到这个时，通常需要 kill 掉残留进程再重跑。

### 3. 残留 vLLM/torchrun 进程占显存

检查：

```bash
pgrep -af "torchrun|train_online_hf_opd_fsdp|VLLM|vllm|EngineCore|python"
nvidia-smi
```

如果确认是旧实验残留，可以 kill 对应 PID。注意不要杀 JupyterLab 和正在用的 teacher，除非你确定要重启它们。

### 4. teacher 服务被关

如果训练正在跑时 teacher 服务被关，训练大概率会因为请求失败或 timeout 中断。重新开 teacher 不一定能救回当前训练，通常需要重新启动训练。

### 5. `model.generate()` 和 FSDP full params

FSDP 下直接 `model.generate()` 需要特殊处理 full params，因为 generate 过程中模型会频繁访问完整权重和 cache。我们目前稳定路线不是直接依赖裸 `model.generate()`，而是用脚本里实现的 `manual_cache` rollout，避免一些 FSDP generate 的坑。

### 6. generation_config 的 temperature/top_p/top_k warning

之前看到过：

```text
generation_config default values have been modified ...
temperature: 0.7, top_p: 0.8, top_k: 20
```

这是 transformers 从模型 generation config 里读到默认采样参数的提示。我们后续通过显式 rollout sampling 参数来控制，关键是确保命令里的：

```text
--rollout-temperature
--rollout-top-p
--rollout-top-k
--rollout-do-sample / --no-rollout-do-sample
```

和实验目标一致。

## 当前工作树注意

后来已经清理了一批一次性 debug/check/compare 脚本，以及两个未跟踪的临时运行脚本。清理后仍需单独注意的本地改动是：

```text
M  CLightMLLM_new/tools/probe_vllm_update_weight.py
```

这个 probe 文件是 vLLM 热更新探索相关，不是稳定训练主路径；提交或恢复前需要单独确认。

## 新对话快速上下文

如果新开对话，可以直接贴这段：

```text
我在做 Qwen3-VL LayerC online OPD。student=Qwen3-VL-2B，teacher=Qwen3-VL-8B vLLM top32 logprob。student 要 FSDP，fp32 参数和 optimizer，bf16 forward，每 12 个样本 update 一次。当前稳定路径是 CLightMLLM_new/tools/train_online_hf_opd_fsdp.py 的 rollout_backend=manual_cache，从 VERL trace 或 Geo3K parquet 读取 prompt/image，teacher vLLM 打分，写 metrics.jsonl。parquet 路径已经改成 VERL-native：RLHFDataset -> apply_chat_template -> build_multimodal_processor_inputs。student vLLM on-policy 热更新已验证 probe 可行，但集成 torchrun/FSDP 仍容易 c10d/NCCL 卡住，暂时不是稳定路径。
```

## 下一步建议

1. 先 push 当前本地 commits。
2. 服务器 pull 后，优先跑 trace + `manual_cache` + `--max-updates 1` smoke。
3. 再跑 parquet + `manual_cache` + `--parquet-max-rows 12` smoke。
4. 如果两条都能出 metrics，再扩大到 1 epoch。
5. student vLLM IPC 等服务器恢复后再单独调，最好给 dedicated GPU。
