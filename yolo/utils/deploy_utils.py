from pathlib import Path

import torch
from torch import Tensor

from yolo.config.config import Config
from yolo.model.yolo import create_model
from yolo.utils.logger import logger


class FastModelLoader:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.compiler = cfg.task.fast_inference
        self.class_num = cfg.dataset.class_num

        self._validate_compiler()
        if cfg.weight == True:
            cfg.weight = Path("weights") / f"{cfg.model.name}.pt"
        self.model_path = f"{Path(cfg.weight).stem}.{self.compiler}"

    def _validate_compiler(self):
        pass

    def load_model(self, device):
        pass

    def _load_onnx_model(self, device):
        pass

    def _create_onnx_model(self, providers):
        pass

    def _load_trt_model(self):
        pass

    def _create_trt_model(self):
        pass
