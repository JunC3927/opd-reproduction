# OPD Reproduction Workspace

This repository is a self-contained workspace for the current OPD alignment work.

## Layout

- `verl_new/`: the modified verl codebase.
- `CLightMLLM_new/`: the modified CLight codebase.
- `download_geometry3k.py`: small helper script for the Geometry3K data.
- `weight_delta_*.json`: small reference summaries kept for comparison.

## Current OPD Focus

The current implementation is set up for the Layer A and Layer B alignment path:

- Layer A replays verl dumps in CLight.
- Layer B reuses verl's teacher-facing inputs and lets CLight call the vLLM teacher again.

The verl side now dumps the vLLM-facing multimodal fields needed by Layer B:

- `multi_modal_data`
- `vllm_images`
- `mm_processor_kwargs`

The CLight side forwards those fields into its local or remote vLLM teacher scorer so that teacher scoring can match verl's prompt-token and image input path as closely as possible.

## Important Files

- `CLightMLLM_new/docs/verl_opd_core.md`
- `verl_new/verl/experimental/agent_loop/agent_loop.py`
- `CLightMLLM_new/src/method/opd.py`
- `CLightMLLM_new/src/method/vllm_teacher.py`
- `CLightMLLM_new/src/method/vllm_teacher_client.py`
- `CLightMLLM_new/tools/serve_vllm_teacher.py`
- `CLightMLLM_new/tools/rescore_verl_trace_with_vllm_teacher.py`
- `CLightMLLM_new/tools/inspect_verl_opd_trace.py`

## Server Usage Sketch

After cloning this repository on the server, use the two project directories directly:

```bash
cd verl_new
# run verl dump / Layer A data generation here

cd ../CLightMLLM_new
# run CLight replay or teacher rescore tools here
```

Large runtime outputs, model weights, trace dumps, and logs are intentionally ignored by Git.
