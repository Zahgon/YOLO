convert_dict = {
    "19.cv1": "19.conv",
    "16.cv1": "16.conv",
    ".7.cv1": ".7.conv",
    ".5.cv1": ".5.conv",
    ".3.cv1": ".3.conv",
    ".28.": ".29.",
    ".25.": ".26.",
    ".22.": ".23.",
    "cv": "conv",
    ".m.": ".bottleneck.",
}

HEAD_NUM = "29"


def convert_weight(old_state_dict, new_state_dict, model_size: int = 38):
    new_weight_set = set(new_state_dict.keys())
    for weight_name, weight_value in old_state_dict.items():
        if HEAD_NUM in weight_name:
            _, _, conv_name, conv_id, *post_fix = weight_name.split(".")
            head_id = 30 if conv_name in ["cv2", "cv3"] else 22
            head_type = "anchor_conv" if conv_name in ["cv2", "cv4"] else "class_conv"
            weight_name = ".".join(["model", str(head_id), "heads", conv_id, head_type, *post_fix])
        else:
            for old_name, new_name in convert_dict.items():
                if old_name in weight_name:
                    weight_name = weight_name.replace(old_name, new_name)
        if weight_name in new_weight_set:
            assert new_state_dict[weight_name].shape == weight_value.shape, f"shape miss match {weight_name}"
            new_state_dict[weight_name] = weight_value
            new_weight_set.remove(weight_name)

    return new_state_dict


head_converter = {
    "head_conv": "m",
    "implicit_a": "ia",
    "implicit_m": "im",
}

SPP_converter = {
    "pre_conv.0": "cv1",
    "pre_conv.1": "cv3",
    "pre_conv.2": "cv4",
    "post_conv.0": "cv5",
    "post_conv.1": "cv6",
    "short_conv": "cv2",
    "merge_conv": "cv7",
}

REP_converter = {"conv1": "rbr_dense", "conv2": "rbr_1x1", "conv": "0", "bn": "1"}


def convert_weight_v7(old_state_dict, new_state_dict):
    pass


replace_dict = {"cv": "conv", ".m.": ".bottleneck."}


def convert_weight_seg(old_state_dict, new_state_dict):
    pass


import sys
from pathlib import Path

import hydra
import torch

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from yolo.config.config import Config
from yolo.tools.solver import BaseModel


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: Config):
    old_weight_path = getattr(cfg, "old_weight", "v9t.pt")
    new_weight_path = getattr(cfg, "new_weight", "ait.pt")
    print(f"Changing {old_weight_path} -> {new_weight_path}")
    cfg.weight = None
    model = BaseModel(cfg)
    old_weight = torch.load(old_weight_path, weights_only=False)
    new_weight = convert_weight(old_weight, model.model.state_dict())
    model.model.load_state_dict(new_weight)
    torch.save(model.model.model.state_dict(), new_weight_path)
    cfg.weight = new_weight_path
    BaseModel(cfg)


if __name__ == "__main__":
    main()
