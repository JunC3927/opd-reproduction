# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0

from verl.utils.import_utils import deprecated


def default_compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
    **kwargs,
):
    source = str(data_source)

    if source == "openai/gsm8k":
        from . import gsm8k

        res = gsm8k.compute_score(solution_str, ground_truth)

    elif source in ["lighteval/MATH", "DigitalLearningGmbH/MATH-lighteval", "HuggingFaceH4/MATH-500"]:
        from . import math_reward

        res = math_reward.compute_score(solution_str, ground_truth)

    elif source in ["math_dapo", "math", "math_dapo_reasoning"] or source.startswith("aime"):
        from . import math_dapo

        res = math_dapo.compute_score(solution_str, ground_truth)

    elif source in [
        "numina_aops_forum",
        "numina_synthetic_math",
        "numina_amc_aime",
        "numina_synthetic_amc",
        "numina_cn_k12",
        "numina_olympiads",
    ]:
        from . import prime_math

        res = prime_math.compute_score(solution_str, ground_truth)

    elif source in ["codecontests", "apps", "codeforces", "taco"]:
        if sandbox_fusion_url:
            from . import sandbox_fusion

            res = sandbox_fusion.compute_score(
                sandbox_fusion_url,
                concurrent_semaphore,
                memory_limit_mb,
                solution_str,
                ground_truth,
                continuous=True,
            )
        else:
            from . import prime_code

            res = prime_code.compute_score(solution_str, ground_truth, continuous=True)

    elif source in ["hiyouga/geometry3k", "geo3k"]:
        from . import geo3k

        res = geo3k.compute_score(solution_str, ground_truth, extra_info=extra_info)

    elif source in [
        "searchR1_nq",
        "searchR1_triviaqa",
        "searchR1_popqa",
        "searchR1_hotpotqa",
        "searchR1_2wikimultihopqa",
        "searchR1_musique",
        "searchR1_bamboogle",
    ]:
        from . import search_r1_like_qa_em

        res = search_r1_like_qa_em.compute_score(solution_str, ground_truth)

    else:
        raise NotImplementedError(f"Reward function is not implemented for data_source={source!r}")

    if isinstance(res, dict):
        return res
    elif isinstance(res, int | float | bool):
        return float(res)
    else:
        return float(res[0])


@deprecated("verl.utils.reward_score.default_compute_score")
def _default_compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
):
    return default_compute_score(
        data_source,
        solution_str,
        ground_truth,
        extra_info,
        sandbox_fusion_url,
        concurrent_semaphore,
        memory_limit_mb,
    )


def get_default_compute_score(reward_name: str | None):
    return default_compute_score


__all__ = ["default_compute_score"]