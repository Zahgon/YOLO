from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from omegaconf import ListConfig, OmegaConf
from torch import nn

from yolo.config.config import ModelConfig, YOLOLayer
from yolo.tools.dataset_preparation import prepare_weight
from yolo.utils.logger import logger
from yolo.utils.module_utils import get_layer_map


class YOLO(nn.Module):
    """
    A preliminary YOLO (You Only Look Once) model class still under development.

    Parameters:
        model_cfg: Configuration for the YOLO model. Expected to define the layers,
                   parameters, and any other relevant configuration details.
    """

    def __init__(self, model_cfg: ModelConfig, class_num: int = 80):
        super(YOLO, self).__init__()
        self.num_classes = class_num
        self.layer_map = get_layer_map()  # Get the map Dict[str: Module]
        self.model: List[YOLOLayer] = nn.ModuleList()
        self.reg_max = getattr(model_cfg.anchor, "reg_max", 16)
        self.build_model(model_cfg.model)

    def build_model(self, model_arch: Dict[str, List[Dict[str, Dict[str, Dict]]]]):
        pass

    def forward(self, x, external: Optional[Dict] = None, shortcut: Optional[str] = None):
        pass

    def get_out_channels(self, layer_type: str, layer_args: dict, output_dim: list, source: Union[int, list]):
        pass

    def get_source_idx(self, source: Union[ListConfig, str, int], layer_idx: int):
        pass

    def create_layer(self, layer_type: str, source: Union[int, list], layer_info: Dict, **kwargs) -> YOLOLayer:
        pass

    def save_load_weights(self, weights: Union[Path, OrderedDict]):
        """
        Update the model's weights with the provided weights.

        args:
            weights: A OrderedDict containing the new weights.
        """
        if isinstance(weights, Path):
            weights = torch.load(weights, map_location=torch.device("cpu"), weights_only=False)
        if "state_dict" in weights:
            weights = {name.removeprefix("model.model."): key for name, key in weights["state_dict"].items()}
        model_state_dict = self.model.state_dict()

        # TODO1: autoload old version weight
        # TODO2: weight transform if num_class difference

        error_dict = {"Mismatch": set(), "Not Found": set()}
        for model_key, model_weight in model_state_dict.items():
            if model_key not in weights:
                error_dict["Not Found"].add(tuple(model_key.split(".")[:-2]))
                continue
            if model_weight.shape != weights[model_key].shape:
                error_dict["Mismatch"].add(tuple(model_key.split(".")[:-2]))
                continue
            model_state_dict[model_key] = weights[model_key]

        for error_name, error_set in error_dict.items():
            error_dict = dict()
            for layer_idx, *layer_name in error_set:
                if layer_idx not in error_dict:
                    error_dict[layer_idx] = [".".join(layer_name)]
                else:
                    error_dict[layer_idx].append(".".join(layer_name))
            for layer_idx, layer_name in error_dict.items():
                layer_name.sort()
                logger.warning(f":warning: Weight {error_name} for Layer {layer_idx}: {', '.join(layer_name)}")

        self.model.load_state_dict(model_state_dict)


def create_model(model_cfg: ModelConfig, weight_path: Union[bool, Path] = True, class_num: int = 80) -> YOLO:
    """Constructs and returns a model from a Dictionary configuration file.

    Args:
        config_file (dict): The configuration file of the model.

    Returns:
        YOLO: An instance of the model defined by the given configuration.
    """
    OmegaConf.set_struct(model_cfg, False)
    model = YOLO(model_cfg, class_num)
    if weight_path:
        if weight_path == True:
            weight_path = Path("weights") / f"{model_cfg.name}.pt"
        elif isinstance(weight_path, str):
            weight_path = Path(weight_path)

        if not weight_path.exists():
            logger.info(f"🌐 Weight {weight_path} not found, try downloading")
            prepare_weight(weight_path=weight_path)
        if weight_path.exists():
            model.save_load_weights(weight_path)
            logger.info(":white_check_mark: Success load model & weight")
    else:
        logger.info(":white_check_mark: Success load model")
    return model
