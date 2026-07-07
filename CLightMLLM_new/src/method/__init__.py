from .base import BaseLearner
from .grpo import GRPOLearner
from .lwf import LwFLearner
from .opd import OPDLearner


def create_learner(
    method_args,
    *,
    model,
    optimizer_args,
    tokenizer=None,
    teacher_model=None,
    reference_model=None,
    student_model_path=None,
    torch_dtype="bfloat16",
):
    if method_args.name == "base":
        return BaseLearner(model=model, optimizer_args=optimizer_args)
    if method_args.name == "lwf":
        return LwFLearner(
            model=model,
            optimizer_args=optimizer_args,
            teacher_model=teacher_model,
            alpha=method_args.lwf_alpha,
            temperature=method_args.lwf_temperature,
        )
    if method_args.name == "grpo":
        return GRPOLearner(
            model=model,
            optimizer_args=optimizer_args,
            tokenizer=tokenizer,
            method_args=method_args,
            reference_model=reference_model,
        )
    if method_args.name == "opd":
        return OPDLearner(
            model=model,
            optimizer_args=optimizer_args,
            tokenizer=tokenizer,
            method_args=method_args,
            teacher_model=teacher_model,
            student_model_path=student_model_path,
            torch_dtype=torch_dtype,
        )
    raise ValueError(f"Unsupported method: {method_args.name}")


__all__ = ["BaseLearner", "GRPOLearner", "LwFLearner", "OPDLearner", "create_learner"]
