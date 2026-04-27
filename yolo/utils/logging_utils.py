"""
Module for initializing logging tools used in machine learning and data processing.
Supports integration with Weights & Biases (wandb), Loguru, TensorBoard, and other
logging frameworks as needed.

This setup ensures consistent logging across various platforms, facilitating
effective monitoring and debugging.

Example:
    from tools.logger import custom_logger
    custom_logger()
"""

import logging
from collections import deque
from logging import FileHandler
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import wandb
from lightning import LightningModule, Trainer, seed_everything
from lightning.pytorch.callbacks import Callback, RichModelSummary, RichProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import CustomProgress
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from lightning.pytorch.utilities import rank_zero_only
from omegaconf import ListConfig, OmegaConf
from rich import get_console, reconfigure
from rich.console import Console, Group
from rich.logging import RichHandler
from rich.table import Table
from rich.text import Text
from torch import Tensor
from torch.nn import ModuleList
from typing_extensions import override

from yolo.config.config import Config, YOLOLayer
from yolo.model.yolo import YOLO
from yolo.utils.logger import logger
from yolo.utils.model_utils import EMA, GradientAccumulation
from yolo.utils.solver_utils import make_ap_table


# TODO: should be moved to correct position
def set_seed(seed):
    seed_everything(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class YOLOCustomProgress(CustomProgress):
    def get_renderable(self):
        pass


class YOLORichProgressBar(RichProgressBar):
    @override
    @rank_zero_only
    def _init_progress(self, trainer: "Trainer") -> None:
        pass

    @override
    def _get_train_description(self, current_epoch: int) -> str:
        pass

    @override
    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        pass

    @override
    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch: Any, batch_idx: int):
        pass

    @override
    @rank_zero_only
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        pass

    @override
    @rank_zero_only
    def on_train_end(self, trainer: "Trainer", pl_module: "LightningModule") -> None:
        pass

    @override
    @rank_zero_only
    def on_validation_end(self, trainer: "Trainer", pl_module: "LightningModule") -> None:
        pass

    @override
    def refresh(self) -> None:
        pass

    @property
    def validation_description(self) -> str:
        pass


class YOLORichModelSummary(RichModelSummary):
    @staticmethod
    @override
    def summarize(
        summary_data: List[Tuple[str, List[str]]],
        total_parameters: int,
        trainable_parameters: int,
        model_size: float,
        total_training_modes: Dict[str, int],
        **summarize_kwargs: Any,
    ) -> None:
        pass


class ImageLogger(Callback):
    def on_validation_batch_end(self, trainer: Trainer, pl_module, outputs, batch, batch_idx) -> None:
        pass


def setup_logger(logger_name, quiet=False):
    class EmojiFormatter(logging.Formatter):
        def format(self, record, emoji=":high_voltage:"):
            pass

    rich_handler = RichHandler(markup=True)
    rich_handler.setFormatter(EmojiFormatter("%(message)s"))
    rich_logger = logging.getLogger(logger_name)
    if rich_logger:
        rich_logger.handlers.clear()
        rich_logger.addHandler(rich_handler)
        if quiet:
            rich_logger.setLevel(logging.ERROR)

    coco_logger = logging.getLogger("faster_coco_eval.core.cocoeval")
    coco_logger.setLevel(logging.ERROR)


def setup(cfg: Config):
    quiet = hasattr(cfg, "quiet")
    setup_logger("lightning.fabric", quiet=quiet)
    setup_logger("lightning.pytorch", quiet=quiet)

    def custom_wandb_log(string="", level=int, newline=True, repeat=True, prefix=True, silent=False):
        pass

    wandb.errors.term._log = custom_wandb_log

    save_path = validate_log_directory(cfg, cfg.name)

    progress, loggers = [], []

    if cfg.task.task == "train" and hasattr(cfg.task.data, "equivalent_batch_size"):
        progress.append(GradientAccumulation(data_cfg=cfg.task.data, scheduler_cfg=cfg.task.scheduler))

    if hasattr(cfg.task, "ema") and cfg.task.ema.enable:
        progress.append(EMA(cfg.task.ema.decay))
    if quiet:
        logger.setLevel(logging.ERROR)
        return progress, loggers, save_path

    progress.append(YOLORichProgressBar())
    progress.append(YOLORichModelSummary())
    progress.append(ImageLogger())
    if cfg.use_tensorboard:
        loggers.append(TensorBoardLogger(log_graph="all", save_dir=save_path))
    if cfg.use_wandb:
        wandb_cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        loggers.append(WandbLogger(project="YOLO", name=cfg.name, save_dir=save_path, id=None, config=wandb_cfg))

    return progress, loggers, save_path


def log_model_structure(model: Union[ModuleList, YOLOLayer, YOLO]):
    pass


@rank_zero_only
def validate_log_directory(cfg: Config, exp_name: str) -> Path:
    base_path = Path(cfg.out_path, cfg.task.task)
    save_path = base_path / exp_name

    if not cfg.exist_ok:
        index = 1
        old_exp_name = exp_name
        while save_path.is_dir():
            exp_name = f"{old_exp_name}{index}"
            save_path = base_path / exp_name
            index += 1
        if index > 1:
            logger.opt(colors=True).warning(
                f"🔀 Experiment directory exists! Changed <red>{old_exp_name}</> to <green>{exp_name}</>"
            )

    save_path.mkdir(parents=True, exist_ok=True)
    if not getattr(cfg, "quiet", False):
        logger.info(f"📄 Created log folder: [blue b u]{save_path}[/]")
    logger.addHandler(FileHandler(save_path / "output.log"))
    return save_path


def log_bbox(
    bboxes: Tensor, class_list: Optional[List[str]] = None, image_size: Tuple[int, int] = (640, 640)
) -> List[dict]:
    """
    Convert bounding boxes tensor to a list of dictionaries for logging, normalized by the image size.

    Args:
        bboxes (Tensor): Bounding boxes with shape (N, 5) or (N, 6), where each box is [class_id, x_min, y_min, x_max, y_max, (confidence)].
        class_list (Optional[List[str]]): List of class names. Defaults to None.
        image_size (Tuple[int, int]): The size of the image, used for normalization. Defaults to (640, 640).

    Returns:
        List[dict]: List of dictionaries containing normalized bounding box information.
    """
    pass
