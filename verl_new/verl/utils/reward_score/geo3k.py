import re


def _normalize(x):
    x = str(x).strip().lower()
    x = x.replace("\\boxed", "")
    x = x.replace("{", "").replace("}", "")
    x = x.replace("$", "")
    x = re.sub(r"\s+", "", x)
    return x


def _extract_answer(solution_str):
    s = str(solution_str)

    boxed = re.findall(r"\\boxed\{([^{}]+)\}", s)
    if boxed:
        return boxed[-1]

    nums = re.findall(r"[-+]?\d*\.?\d+(?:/\d+)?", s)
    if nums:
        return nums[-1]

    return s.strip()[-50:]


def compute_score(solution_str, ground_truth, extra_info=None, **kwargs):
    pred = _extract_answer(solution_str)
    gt = ground_truth

    return 1.0 if _normalize(pred) == _normalize(gt) else 0.0