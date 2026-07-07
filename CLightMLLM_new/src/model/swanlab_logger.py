import inspect
from argparse import Namespace
from typing import Any

import torch
from lightning.pytorch.loggers.logger import Logger, rank_zero_experiment
from lightning.pytorch.utilities.rank_zero import rank_zero_only


def _to_plain_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        value = value.detach()
        if value.numel() == 1:
            return value.item()
        return value.cpu()
    return value


def _namespace_to_dict(params: Any) -> dict[str, Any]:
    if isinstance(params, Namespace):
        return vars(params)
    if isinstance(params, dict):
        return params
    return dict(params) if params is not None else {}


def _supported_kwargs(function: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return {key: value for key, value in kwargs.items() if value is not None}

    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
        return {key: value for key, value in kwargs.items() if value is not None}

    return {
        key: value
        for key, value in kwargs.items()
        if value is not None and key in signature.parameters
    }


class SwanLabLogger(Logger):
    def __init__(
        self,
        *,
        project: str,
        experiment_name: str | None = None,
        workspace: str | None = None,
        mode: str | None = None,
        save_dir: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self._project = project
        self._experiment_name = experiment_name
        self._workspace = workspace
        self._mode = mode
        self._save_dir = save_dir
        self._config = config or {}
        self._experiment = None

    @property
    def name(self) -> str:
        return self._experiment_name or self._project

    @property
    def version(self) -> str | None:
        return None

    @property
    @rank_zero_experiment
    def experiment(self) -> Any:
        if self._experiment is not None:
            return self._experiment

        try:
            import swanlab
        except ImportError as exc:
            raise ImportError(
                "trainer.swanlab_project is set, but the `swanlab` package is not installed. "
                "Install it on the training server with `pip install swanlab`, or remove "
                "trainer.swanlab_project from the config."
            ) from exc

        init_kwargs = _supported_kwargs(
            swanlab.init,
            {
                "project": self._project,
                "experiment_name": self._experiment_name,
                "workspace": self._workspace,
                "mode": self._mode,
                "logdir": self._save_dir,
                "save_dir": self._save_dir,
                "config": self._config,
            },
        )
        self._experiment = swanlab.init(**init_kwargs)
        return self._experiment

    @rank_zero_only
    def log_hyperparams(self, params: Namespace | dict[str, Any], *args: Any, **kwargs: Any) -> None:
        _ = self.experiment
        try:
            import swanlab
        except ImportError:
            import swanlab

        config = _namespace_to_dict(params)
        if not config:
            return

        if hasattr(swanlab, "config") and hasattr(swanlab.config, "update"):
            swanlab.config.update(config)

    @rank_zero_only
    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        _ = self.experiment
        try:
            import swanlab
        except ImportError:
            import swanlab

        clean_metrics = {key: _to_plain_value(value) for key, value in metrics.items()}
        if not clean_metrics:
            return

        log_kwargs = _supported_kwargs(swanlab.log, {"step": step})
        swanlab.log(clean_metrics, **log_kwargs)

    @rank_zero_only
    def finalize(self, status: str) -> None:
        try:
            import swanlab
        except ImportError:
            return
        finish = getattr(swanlab, "finish", None)
        if callable(finish):
            finish()
