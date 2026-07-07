# verl OPD 核心链路笔记

这份笔记只关注 Qwen3-VL OPD 里最核心的四件事：

1. student 怎么用 vLLM rollout。
2. teacher 怎么用 vLLM 计算 top-k logprob。
3. `forward_kl_topk` loss 怎么算。
4. student 怎么用 FSDP/HF forward 做 backward 和 optimizer update。

结论先放前面：verl 的 OPD 数学核心不复杂，复杂的是 Ray worker、异步 vLLM server、FSDP 权重同步、padding/no-padding 转换这些工程外壳。真正需要复刻的核心流程大约是：

```text
dataset prompt/image
  -> student vLLM generate response
  -> sequence = prompt + response
  -> teacher vLLM prompt_logprobs(topk=32) over sequence
  -> actor FSDP/HF forward on sequence
  -> student logprob gathered at teacher top-k ids
  -> forward KL token loss
  -> mask response tokens
  -> token-mean loss
  -> backward, clip grad, optimizer step
  -> sync updated actor weights back to student vLLM
```

## 1. 总体入口

训练主循环在：

- `D:\opd_again\verl_new\verl\trainer\ppo\ray_trainer.py`

关键阶段：

- `generate_sequences`: 让 student rollout server 生成回答。
- `_update_actor`: 把 rollout 后的 batch 交给 actor worker 更新。
- `checkpoint_manager.update_weights`: actor 更新后把新权重同步回 student vLLM。

在 `_update_actor` 里，verl 会把 batch 从 padded tensor 转成 no-padding tensordict，再设置训练所需的 non-tensor metadata：

```python
batch_td = batch.to_tensordict()
batch_td = left_right_2_no_padding(batch_td)
tu.assign_non_tensor(
    batch_td,
    calculate_entropy=calculate_entropy,
    distillation_use_topk=distillation_use_topk,
    global_batch_size=ppo_mini_batch_size,
    mini_batch_size=ppo_mini_batch_size,
    epochs=ppo_epochs,
    seed=seed,
    dataloader_kwargs={"shuffle": shuffle},
    compute_loss=True,
)
actor_output = self.actor_rollout_wg.update_actor(batch_td)
```

所以在 OPD 里，真正进入 actor update 的 batch 已经包含：

- `prompts`: `[B, 1024]`
- `responses`: `[B, 512]`
- `input_ids`: `[B, 1536]`
- `attention_mask`: `[B, 1536]`
- `position_ids`: `[B, 4, 1536]` for Qwen3-VL
- `response_mask`: `[B, 512]`
- `teacher_ids`: `[B, 1536, 32]`
- `teacher_logprobs`: `[B, 1536, 32]`

## 2. Student vLLM Rollout

### 2.1 vLLM server 怎么启动

核心文件：

- `D:\opd_again\verl_new\verl\workers\rollout\vllm_rollout\vllm_async_server.py`

`launch_server` 里组装 vLLM engine 参数。和我们实验最相关的是这些：

```python
args = {
    "dtype": self.config.dtype,
    "load_format": self.config.load_format,
    "distributed_executor_backend": "mp",
    "trust_remote_code": self.model_config.trust_remote_code,
    "max_model_len": self.config.max_model_len,
    "max_num_seqs": self.config.max_num_seqs,
    "enable_chunked_prefill": self.config.enable_chunked_prefill,
    "max_num_batched_tokens": self.config.max_num_batched_tokens,
    "enable_prefix_caching": self.config.enable_prefix_caching,
    "enable_sleep_mode": self.config.enable_sleep_mode,
    "logprobs_mode": self.config.logprobs_mode,
    "enforce_eager": self.config.enforce_eager,
    "gpu_memory_utilization": self.config.gpu_memory_utilization,
    "tensor_parallel_size": self.config.tensor_model_parallel_size,
    "seed": self.replica_rank + (self.config.get("seed") or 0),
}
```

默认 rollout 配置在：

- `D:\opd_again\verl_new\verl\trainer\config\rollout\rollout.yaml`

重要默认值：

```yaml
temperature: 1.0
top_k: -1
top_p: 1
dtype: bfloat16
enforce_eager: False
max_num_batched_tokens: 8192
max_num_seqs: 1024
enable_chunked_prefill: True
enable_prefix_caching: True
logprobs_mode: processed_logprobs
load_format: dummy
do_sample: True
n: 1
```

你的原始命令又覆盖了：

```text
actor_rollout_ref.rollout.name=vllm
actor_rollout_ref.rollout.tensor_model_parallel_size=1
actor_rollout_ref.rollout.gpu_memory_utilization=0.4
actor_rollout_ref.rollout.n=1
actor_rollout_ref.rollout.max_model_len=1537
actor_rollout_ref.rollout.free_cache_engine=False
actor_rollout_ref.rollout.enable_sleep_mode=False
```

注意：`load_format: dummy` 不代表使用随机权重训练。它是 vLLM engine 启动时的加载策略，之后会通过 actor/FSDP 权重同步把真实 student 权重送进 vLLM。

### 2.2 单条请求怎么 generate

核心函数：

- `vllm_async_server.py::generate`

关键流程：

```python
max_possible_tokens = self.config.max_model_len - len(prompt_ids)
max_tokens = min(
    self.config.response_length,
    self.config.prompt_length + self.config.response_length - len(prompt_ids),
)
sampling_params = SamplingParams(max_tokens=max_tokens, **sampling_params)
prompt_ids = qwen2_5_vl_dedup_image_tokens(prompt_ids, self.model_config.processor)

prompt = TokensPrompt(
    prompt_token_ids=prompt_ids,
    multi_modal_data={"image": image_data},
)

generator = self.engine.generate(
    prompt=prompt,
    sampling_params=sampling_params,
    request_id=request_id,
)
```

这里有一个多模态关键点：`qwen2_5_vl_dedup_image_tokens` 会把连续的 `<|image_pad|>` 压缩成一个：

```text
<|vision_start|><|image_pad|><|image_pad|>...<|image_pad|><|vision_end|>
=> <|vision_start|><|image_pad|><|vision_end|>
```

原因是 vLLM 会根据 `multi_modal_data["image"]` 自己展开图像占位。这个逻辑在：

- `D:\opd_again\verl_new\verl\workers\rollout\utils.py::qwen2_5_vl_dedup_image_tokens`

### 2.3 rollout 结果怎么 padding 回训练 batch

核心文件：

- `D:\opd_again\verl_new\verl\experimental\agent_loop\agent_loop.py`

`generate_sequences` 先构造采样参数：

```python
sampling_params = dict(
    temperature=config.temperature,
    top_p=config.top_p,
    top_k=config.top_k,
    repetition_penalty=1.0,
    logprobs=config.calculate_log_probs,
)
```

然后每个 sample 调 `_run_agent_loop`。单轮任务最终在 `_agent_loop_postprocess` 里 padding：

```python
prompt_output = pad(prompt_ids, max_length=prompt_length, padding_side="left")
response_output = pad(response_ids, max_length=response_length, padding_side="right")
response_mask = response_mask_output["input_ids"] * response_output["attention_mask"]
attention_mask = cat(prompt_attention_mask, response_attention_mask)
input_ids = cat(prompt_input_ids, response_input_ids)
```

所以 verl 的训练输入是固定宽度：

```text
prompt: left padded to 1024
response: right padded to 512
input_ids: 1024 + 512 = 1536
```

这也解释了我们之前看到的：

```text
prompts: (24, 1024)
responses: (24, 512)
input_ids: (24, 1536)
attention_mask: (24, 1536)
```

## 3. Teacher vLLM Logprob

Teacher 核心文件：

- `D:\opd_again\verl_new\verl\experimental\teacher_loop\teacher_manager.py`

### 3.1 Teacher 不生成答案，只打分整段 sequence

在 agent loop 里：

```python
teacher_ids, teacher_logprobs = await self.teacher_server_manager.compute_teacher_logprobs_single(
    sequence_ids=prompt_ids + response_ids,
    multi_modal_data=output.multi_modal_data,
    mm_processor_kwargs=output.mm_processor_kwargs,
    routing_key=routing_key,
)
```

也就是说 teacher 收到的是：

```text
sequence_ids = prompt_ids + student_response_ids
```

不是只给 prompt，也不是让 teacher 自己 generate response。

### 3.2 Teacher sampling params

`_get_teacher_sampling_params` 返回：

```python
{
    "max_tokens": 1,
    "temperature": teacher_model_config.inference.temperature,
    "prompt_logprobs": topk,
}
```

你的实验里 `topk=32`，所以 teacher vLLM 返回每个 prompt position 的 top-32 token ids 和 logprobs。

这里的 `max_tokens=1` 只是为了满足 vLLM generate 接口，它真正要的是 `prompt_logprobs`。teacher 输出的一个新 token不参与 OPD loss。

### 3.3 prompt_logprobs 怎么变成 teacher_ids / teacher_logprobs

vLLM 返回的 `RequestOutput.prompt_logprobs` 在：

- `D:\opd_again\verl_new\verl\workers\rollout\vllm_rollout\utils.py::extract_prompt_logprobs`

转换成两个列表：

```python
result_dict["prompt_ids"] = prompt_ids_ls
result_dict["prompt_logprobs"] = prompt_logprobs_ls
```

其中：

```text
prompt_ids_ls[position][rank]      = teacher top-k token id
prompt_logprobs_ls[position][rank] = teacher top-k logprob
```

然后 teacher manager 转成 tensor：

```python
teacher_ids = torch.tensor(teacher_output.extra_fields["prompt_ids"], dtype=torch.int32)
teacher_logprobs = torch.tensor(teacher_output.extra_fields["prompt_logprobs"])
```

单条样本 unpadded 时形状是：

```text
teacher_ids:      [sequence_len, topk]
teacher_logprobs: [sequence_len, topk]
```

padding 回 batch 后是：

```text
teacher_ids:      [batch, 1536, topk]
teacher_logprobs: [batch, 1536, topk]
```

### 3.4 为什么 response 通常从 1023 取 teacher top-k

HF causal LM 的 logits 对齐关系是：

```text
logits[:, t, :] 预测 input_ids[:, t + 1]
```

verl 的 `teacher_ids[:, pos, :]` 表示 teacher 对 `sequence[pos]` 这个 token 的 prompt logprob/top-k。

如果 response token 从 full sequence 的 token index `1024` 开始，那么 student 用来预测它的是：

```text
shifted_logits index = 1024 - 1 = 1023
```

所以 replay 里取 teacher/topk 对齐 student shifted logits 时，常见写法是：

```python
response_start = prompt_width - 1
active_teacher_ids = teacher_ids[:, response_start : response_start + response_len, :]
student_logits = outputs.logits[:, :-1, :]
active_student_logits = student_logits[:, response_start : response_start + response_len, :]
```

这不是说 response token 本身从 1023 开始，而是“预测第一个 response token 的 logits”在 shifted 序列里位于 1023。

## 4. forward_kl_topk Loss

核心文件：

- `D:\opd_again\verl_new\verl\trainer\distillation\fsdp\losses.py`
- `D:\opd_again\verl_new\verl\trainer\distillation\losses.py`

FSDP top-k loss 的核心公式：

```python
student_log_probs = F.log_softmax(student_logits, dim=-1)
student_topk_log_probs = torch.gather(student_log_probs, dim=-1, index=teacher_topk_ids)
student_mass = student_topk_log_probs.exp().sum(dim=-1)
teacher_mass = teacher_topk_log_probs.exp().sum(dim=-1)

distillation_losses = sum(
    exp(teacher_topk_log_probs) * (teacher_topk_log_probs - student_topk_log_probs),
    dim=-1,
)
```

也就是：

```text
loss[token] = sum over teacher top-k:
    p_teacher(token_k) * (log p_teacher(token_k) - log p_student(token_k))
```

注意这里不是比较 student 自己 top-k 和 teacher top-k 的完整集合分布，而是：

```text
在 teacher top-k ids 上 gather student logprob。
```

这正是我们 replay 里复现的逻辑。

### 4.1 clamp 和负值

verl 里还有：

```python
student_topk_log_probs = clamp_min(log_prob_min_clamp)
teacher_topk_log_probs = clamp_min(log_prob_min_clamp)
distillation_losses = distillation_losses.clamp_min(0.0)
distillation_losses = clamp(loss_max_clamp)
```

你的配置里：

```text
log_prob_min_clamp = -10
loss_max_clamp = 10
```

### 4.2 token-mean 聚合

最终聚合在 `distillation_loss`：

```python
distillation_loss = agg_loss(
    loss_mat=distillation_losses,
    loss_mask=response_mask,
    loss_agg_mode=loss_agg_mode,
    **config.global_batch_info,
)
```

actor 默认：

```python
loss_agg_mode = "token-mean"
```

所以它是按有效 response token 做归一化，而不是简单 batch mean。

## 5. Student FSDP/HF Forward 和 Update

核心文件：

- `D:\opd_again\verl_new\verl\workers\engine_workers.py`
- `D:\opd_again\verl_new\verl\workers\engine\fsdp\transformer_impl.py`

Ray worker 入口：

```python
def update_actor(self, data):
    output = self.actor.train_mini_batch(data=data)
```

FSDP engine 的 forward/backward：

```python
batch_num_tokens = data["loss_mask"].sum()
torch.distributed.all_reduce(batch_num_tokens)
tu.assign_non_tensor(data, batch_num_tokens=batch_num_tokens.item())

micro_batches, indices = prepare_micro_batches(...)

for micro_batch in micro_batches:
    loss, meta_info = self.forward_step(...)
    loss.backward()
```

实际 model forward：

```python
with autocast_ctx:
    raw_output = self.module(
        **model_inputs,
        use_cache=False,
    )
    model_output = self.prepare_model_outputs(
        output=raw_output,
        output_args=output_args,
        micro_batch=micro_batch,
        logits_processor_func=loss_function,
    )
    loss, metrics = loss_function(
        model_output=model_output,
        data=micro_batch,
        dp_group=self.get_data_parallel_group(),
    )
```

也就是说 student 更新阶段不是 vLLM forward，而是 FSDP 包住的 HF model forward。

### 5.1 参数 dtype 和计算 dtype

verl rollout vLLM 配置是：

```yaml
rollout.dtype: bfloat16
```

FSDP actor 的 forward 里有 autocast：

```python
autocast_dtype = self._autocast_dtype
autocast_ctx = nullcontext() if autocast_dtype == torch.float32 else torch.autocast(..., dtype=autocast_dtype)
```

所以常见状态是：

```text
参数存储/优化器状态: FSDP/optimizer 管理，可能包含 fp32 master / mixed precision 策略
forward 计算: bf16 autocast
vLLM rollout/teacher: bf16 engine
```

准确 dtype 要看 actor FSDP mixed precision 配置，不能只从模型文件大小推断。

### 5.2 梯度裁剪与 optimizer step

FSDP optimizer step：

```python
grad_norm = self.module.clip_grad_norm_(clip_grad)
optimizer.step()
lr_scheduler.step()
```

你的 verl 里 `actor/grad_norm` 是裁剪前 norm，`clip_grad=1.0` 后再 step。

### 5.3 update weights 回 vLLM

actor 更新后，Ray trainer 会调用：

```python
self.checkpoint_manager.update_weights(self.global_steps)
```

worker 内部：

```python
per_tensor_param, peft_config = self.actor.engine.get_per_tensor_param(...)
await self.rollout.update_weights(per_tensor_param, peft_config=peft_config, global_steps=global_steps)
```

所以 verl 是：

```text
每个 train step 更新 actor FSDP 权重
再把新权重同步到 student vLLM
下一轮 rollout 使用更新后的 student
```

不是一直用初始模型 rollout。

## 6. 哪些代码值得搬，哪些不值得搬

### 值得搬到 CLight 的核心

1. vLLM rollout 请求构造：
   - `TokensPrompt(prompt_token_ids=dedup_prompt_ids, multi_modal_data={"image": images})`
   - `SamplingParams(temperature=1.0, top_p=1.0, top_k=-1, max_tokens=512, stop_token_ids=[eos])`

2. teacher scoring：
   - `sequence_ids = prompt_ids + response_ids`
   - `SamplingParams(max_tokens=1, temperature=1.0, prompt_logprobs=32)`
   - 取 `prompt_logprobs` 变成 `[seq_len, topk]` 的 `teacher_ids/logprobs`

3. alignment：
   - response token 本身在 full sequence 的 `prompt_width`
   - student shifted logits 从 `prompt_width - 1`
   - teacher prompt logprobs 同样从 `prompt_width - 1`

4. loss：
   - `student_log_probs.gather(teacher_ids)`
   - `sum p_teacher * (logp_teacher - logp_student)`
   - response mask token-mean

5. update：
   - bf16 forward autocast
   - grad clip 1.0
   - AdamW
   - 每步后刷新 student rollout 权重

### 不建议直接搬的工程外壳

1. Ray worker group 和 resource pool。
2. async OpenAI-compatible server 包装。
3. 多节点 DP/TP RPC。
4. checkpoint engine 的 async 权重传输。
5. LoRA / QAT / MTP / MoE / routing replay。
6. Prometheus/profiler/trace 系统。

这些是 verl 为通用大规模训练准备的，不是 OPD 必须条件。

## 7. CLight 复现时的最小目标

最小可复现版本可以分三层：

### Layer A: replay 已验证

输入：verl dump 的 `responses + teacher_ids + teacher_logprobs + masks + pixels`

CLight 自己做：

```text
HF forward -> top-k KL loss -> backward -> optimizer step
```

我们已经验证：

```text
loss / grad_norm / update magnitude 和 verl 很接近
```

### Layer B: teacher vLLM scoring

输入：CLight/HF 或 vLLM 生成的 response

CLight 自己调用 teacher vLLM：

```text
prompt + response -> prompt_logprobs topk=32
```

再走 Layer A 的 loss/update。

### Layer C: student vLLM online rollout

每步：

```text
student vLLM rollout
teacher vLLM score
student HF/FSDP update
sync updated student weights to vLLM
```

这是最接近 verl 的完整 OPD。难点不在 loss，而在：

```text
如何稳定地把更新后的 student 权重同步给 vLLM。
```

## 8. 一句话定位

如果只想盯 OPD 核心，可以按这个顺序读代码：

1. `agent_loop.py::generate_sequences`
2. `vllm_async_server.py::generate`
3. `teacher_manager.py::compute_teacher_logprobs_single`
4. `utils.py::extract_prompt_logprobs`
5. `distillation/fsdp/losses.py::compute_forward_kl_topk`
6. `distillation/losses.py::distillation_loss`
7. `ray_trainer.py::_update_actor`
8. `engine/fsdp/transformer_impl.py::forward_backward_batch`
9. `engine/fsdp/transformer_impl.py::optimizer_step`
10. `engine_workers.py::update_weights`

