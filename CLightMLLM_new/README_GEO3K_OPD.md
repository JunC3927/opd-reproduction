# Geo3K OPD 运行说明

本文档说明如何在 `CLightMLLM_new` 中跑通 Geo3K 的 OPD 训练。

当前已验证的主线是：

```text
teacher: Qwen3-VL-8B-Instruct, 独立 vLLM server
student: Qwen3-VL-2B-Instruct, Lightning + FSDP 训练
data: Geo3K parquet, 使用 VERLPromptConverter 对齐 VERL prompt/image 格式
loss: teacher top-k logprobs 上的 OPD forward KL
```

## 1. 目录与环境

进入项目目录：

```bash
cd /work/03/gw42/j40004/cj/opd-reproduction/CLightMLLM_new
```

激活环境：

```bash
conda activate clightmllm
```

确认 Python 和 CUDA：

```bash
which python
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
PY
```

## 2. 需要准备的本地文件

### Student 模型

默认配置使用：

```text
/work/03/gw42/j40004/models/Qwen3-VL-2B-Instruct
```

对应配置字段：

```yaml
model:
    model_name_or_path: /work/03/gw42/j40004/models/Qwen3-VL-2B-Instruct
```

### Teacher 模型

teacher server 默认使用：

```text
/work/03/gw42/j40004/models/Qwen3-VL-8B-Instruct
```

### Geo3K 数据

默认配置使用：

```text
/work/03/gw42/j40004/cj/data/geo3k/train.parquet
```

它在 `config/dataset.json` 中注册为：

```json
"geo3k_j40004_verl_prompt": {
  "load_from": "parquet",
  "file_name_or_path": "/work/03/gw42/j40004/cj/data/geo3k/train.parquet",
  "formatting": "verl_prompt",
  "columns": {
    "messages": "prompt",
    "images": "images"
  }
}
```

训练配置里使用：

```yaml
cl_sft:
    stages:
        - geo3k_j40004_verl_prompt
```

## 3. 推荐配置文件

当前 Geo3K OPD 配置在：

```text
config/opd/
```

主要有三份：

```text
config/opd/qwen3_vl_geo3k_hf.yaml
config/opd/qwen3_vl_geo3k_vllm_student_server.yaml
config/opd/qwen3_vl_geo3k_vllm_student_server_5epoch.yaml
```

推荐优先使用：

```text
config/opd/qwen3_vl_geo3k_vllm_student_server_5epoch.yaml
```

这条路径使用独立 student vLLM server 做 rollout，并支持 optimizer step 后热同步 student 权重。

如果想先跑更稳但更慢的版本，可以使用：

```text
config/opd/qwen3_vl_geo3k_hf.yaml
```

## 4. 启动 teacher vLLM server

另开一个终端，启动 teacher server：

```bash
cd /work/03/gw42/j40004/cj/opd-reproduction/CLightMLLM_new
conda activate clightmllm

CUDA_VISIBLE_DEVICES=<teacher_gpu> python tools/serve_vllm_teacher.py \
  --model /work/03/gw42/j40004/models/Qwen3-VL-8B-Instruct \
  --host 127.0.0.1 \
  --port 29577 \
  --torch-dtype bfloat16 \
  --gpu-memory-utilization 0.8 \
  --max-model-len 6000 \
  --max-num-batched-tokens 8192 \
  --topk 32
```

检查 teacher server 是否在监听：

```bash
python - <<'PY'
from src.method.rpc import rpc_call
print(rpc_call("127.0.0.1", 29577, {"op": "ping"}, 30.0))
PY
```

## 5. 路径 A：HF rollout 训练

HF rollout 不需要 student vLLM server。

启动训练：

```bash
cd /work/03/gw42/j40004/cj/opd-reproduction/CLightMLLM_new
conda activate clightmllm

CUDA_VISIBLE_DEVICES=4,5,6,7 python train.py \
  --config config/opd/qwen3_vl_geo3k_hf.yaml
```

特点：

```text
优点：路径简单，稳定
缺点：rollout 较慢
```

## 6. 路径 B：student vLLM server rollout 训练

这个版本需要先启动 student server。

### 6.1 启动 student vLLM server

另开一个终端：

```bash
cd /work/03/gw42/j40004/cj/opd-reproduction/CLightMLLM_new
conda activate clightmllm

CUDA_VISIBLE_DEVICES=<student_gpu> python tools/serve_vllm_student.py \
  --model /work/03/gw42/j40004/models/Qwen3-VL-2B-Instruct \
  --host 127.0.0.1 \
  --port 29588 \
  --torch-dtype bfloat16 \
  --gpu-memory-utilization 0.30 \
  --max-model-len 1536 \
  --enforce-eager \
  --local-files-only \
  --image-min-pixels 1024 \
  --image-max-pixels 262144 \
  --ipc-bucket-size-mb 2048 \
  --sync-dtype none
```

检查 student server：

```bash
python - <<'PY'
from src.method.rpc import rpc_call
print(rpc_call("127.0.0.1", 29588, {"op": "ping"}, 30.0))
PY
```

### 6.2 启动 4 卡训练

```bash
cd /work/03/gw42/j40004/cj/opd-reproduction/CLightMLLM_new
conda activate clightmllm

CUDA_VISIBLE_DEVICES=4,5,6,7 python train.py \
  --config config/opd/qwen3_vl_geo3k_vllm_student_server_5epoch.yaml
```

特点：

```text
优点：rollout 更快，支持 on-policy 热权重同步
缺点：需要额外维护 student vLLM server
```

## 7. 6 卡训练注意事项

如果想用 6 卡训练，不需要改代码，但需要改 YAML：

```yaml
trainer:
    devices: 6
```

如果想保持原来的 global batch size 为 12：

```yaml
loader:
    per_device_train_batch_size: 2
```

如果 `per_device_train_batch_size` 仍然是 3，那么 6 卡 global batch 会变成 18，实验就不再严格对齐 4 卡设置。

## 8. 输出位置

训练输出由 YAML 中的 `trainer.save_dir` 控制。

HF rollout 默认：

```text
/work/03/gw42/j40004/cj/opd_dumps/layerC_lightning_opd_geo3k
```

student vLLM server 5 epoch 默认：

```text
/work/03/gw42/j40004/cj/opd_dumps/layerC_lightning_opd_geo3k_vllm_student_server_5epoch
```

训练结束后，如果：

```yaml
trainer:
    export_hf_model_at_end: true
```

会导出 Hugging Face safetensors 格式模型到：

```text
<save_dir>/model
```

指标日志：

```text
<save_dir>/metrics.jsonl
```

## 9. 已验证结果参考

当前已验证过的结果大致如下：

```text
HF rollout:
  Geo3K 评测约 40.33%
  训练耗时约 7 小时 47 分 34 秒

student vLLM server rollout:
  Geo3K 评测约 39.67%
  训练耗时约 5 小时 10 分 40 秒

base 约 30%
```

HF rollout 和 student vLLM server rollout 的训练曲线基本对齐。

## 10. 常见问题

### teacher server 和 student server 不要混淆

teacher server：

```text
tools/serve_vllm_teacher.py
默认端口 29577
Qwen3-VL-8B-Instruct
负责给 top-k logprobs
```

student server：

```text
tools/serve_vllm_student.py
默认端口 29588
Qwen3-VL-2B-Instruct
负责 student rollout
```

### student server 代码更新后要重启

如果修改了 `tools/serve_vllm_student.py` 或 `src/method/vllm_student.py`，训练服务器需要：

```bash
git pull
```

然后重启 student vLLM server。

已经启动的 server 不会自动加载新代码。

### vLLM EngineCore 显存残留

如果 student vLLM server 异常退出后 GPU 显存不释放，优先杀父进程：

```bash
ps -ef | grep serve_vllm_student.py
kill <parent_pid>
```

如果还有 `VLLM::EngineCore` 残留，再处理对应进程。

如果某台 A100 NVLink 机器出现单卡 ERR 或残留显存，单卡 reset 可能不被允许，可能需要整组 GPU reset 或换卡/换节点。

## 11. 最小运行清单

最少需要开三个终端：

```text
Terminal 1: teacher vLLM server
Terminal 2: student vLLM server
Terminal 3: Lightning training
```

如果使用 HF rollout，则只需要：

```text
Terminal 1: teacher vLLM server
Terminal 2: Lightning training
```
