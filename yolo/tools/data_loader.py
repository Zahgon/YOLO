from pathlib import Path
from queue import Empty, Queue
from statistics import mean
from threading import Event, Thread
from typing import Generator, List, Tuple, Union

import numpy as np
import torch
from PIL import Image
from rich.progress import track
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from yolo.config.config import DataConfig, DatasetConfig
from yolo.tools.data_augmentation import *
from yolo.tools.data_augmentation import AugmentationComposer
from yolo.tools.dataset_preparation import prepare_dataset
from yolo.utils.dataset_utils import (
    create_image_metadata,
    locate_label_paths,
    scale_segmentation,
    tensorlize,
)
from yolo.utils.logger import logger


class YoloDataset(Dataset):
    def __init__(self, data_cfg: DataConfig, dataset_cfg: DatasetConfig, phase: str = "train2017"):
        augment_cfg = data_cfg.data_augment
        self.image_size = data_cfg.image_size
        phase_name = dataset_cfg.get(phase, phase)
        self.batch_size = data_cfg.batch_size
        self.dynamic_shape = getattr(data_cfg, "dynamic_shape", False)
        self.base_size = mean(self.image_size)

        transforms = [eval(aug)(prob) for aug, prob in augment_cfg.items()]
        self.transform = AugmentationComposer(transforms, self.image_size, self.base_size)
        self.transform.get_more_data = self.get_more_data
        self.img_paths, self.bboxes, self.ratios = tensorlize(self.load_data(Path(dataset_cfg.path), phase_name))

    def load_data(self, dataset_path: Path, phase_name: str):
        """
        Loads data from a cache or generates a new cache for a specific dataset phase.

        Parameters:
            dataset_path (Path): The root path to the dataset directory.
            phase_name (str): The specific phase of the dataset (e.g., 'train', 'test') to load or generate data for.

        Returns:
            dict: The loaded data from the cache for the specified phase.
        """
        pass

    def filter_data(self, dataset_path: Path, phase_name: str, sort_image: bool = False) -> list:
        """
        Filters and collects dataset information by pairing images with their corresponding labels.

        Parameters:
            images_path (Path): Path to the directory containing image files.
            labels_path (str): Path to the directory containing label files.
            sort_image (bool): If True, sorts the dataset by the width-to-height ratio of images in descending order.

        Returns:
            list: A list of tuples, each containing the path to an image file and its associated segmentation as a tensor.
        """
        pass

    def load_valid_labels(self, label_path: str, seg_data_one_img: list) -> Union[Tensor, None]:
        """
        Loads valid COCO style segmentation data (values between [0, 1]) and converts it to bounding box coordinates
        by finding the minimum and maximum x and y values.

        Parameters:
            label_path (str): The filepath to the label file containing annotation data.
            seg_data_one_img (list): The actual list of annotations (in segmentation format)

        Returns:
            Tensor or None: A tensor of all valid bounding boxes if any are found; otherwise, None.
        """
        pass

    def get_data(self, idx):
        pass

    def get_more_data(self, num: int = 1):
        pass

    def _update_image_size(self, idx: int) -> None:
        """Update image size based on dynamic shape and batch settings."""
        pass

    def __getitem__(self, idx) -> Tuple[Image.Image, Tensor, Tensor, List[str]]:
        img, bboxes, img_path = self.get_data(idx)

        if self.dynamic_shape:
            self._update_image_size(idx)

        img, bboxes, rev_tensor = self.transform(img, bboxes)
        bboxes[:, [1, 3]] *= self.image_size[0]
        bboxes[:, [2, 4]] *= self.image_size[1]
        return img, bboxes, rev_tensor, img_path

    def __len__(self) -> int:
        return len(self.bboxes)


def collate_fn(batch: List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, List[Tensor]]:
    """
    A collate function to handle batching of images and their corresponding targets.

    Args:
        batch (list of tuples): Each tuple contains:
            - image (Tensor): The image tensor.
            - labels (Tensor): The tensor of labels for the image.

    Returns:
        Tuple[Tensor, List[Tensor]]: A tuple containing:
            - A tensor of batched images.
            - A list of tensors, each corresponding to bboxes for each image in the batch.
    """
    pass


def create_dataloader(data_cfg: DataConfig, dataset_cfg: DatasetConfig, task: str = "train"):
    if task == "inference":
        return StreamDataLoader(data_cfg)

    if getattr(dataset_cfg, "auto_download", False):
        prepare_dataset(dataset_cfg, task)
    dataset = YoloDataset(data_cfg, dataset_cfg, task)

    return DataLoader(
        dataset,
        batch_size=data_cfg.batch_size,
        num_workers=data_cfg.cpu_num,
        pin_memory=data_cfg.pin_memory,
        collate_fn=collate_fn,
    )


class StreamDataLoader:
    def __init__(self, data_cfg: DataConfig):
        self.source = data_cfg.source
        self.running = True
        self.is_stream = isinstance(self.source, int) or str(self.source).lower().startswith("rtmp://")

        self.transform = AugmentationComposer([], data_cfg.image_size)
        self.stop_event = Event()

        if self.is_stream:
            import cv2

            self.cap = cv2.VideoCapture(self.source)
        else:
            self.source = Path(self.source)
            self.queue = Queue()
            self.thread = Thread(target=self.load_source)
            self.thread.start()

    def load_source(self):
        pass

    def load_image_folder(self, folder):
        pass

    def process_image(self, image_path):
        pass

    def load_video_file(self, video_path):
        pass

    def process_frame(self, frame):
        pass

    def __iter__(self) -> Generator[Tensor, None, None]:
        return self

    def __next__(self) -> Tensor:
        if self.is_stream:
            ret, frame = self.cap.read()
            if not ret:
                self.stop()
                raise StopIteration
            self.process_frame(frame)
            return self.current_frame
        else:
            try:
                frame = self.queue.get(timeout=1)
                return frame
            except Empty:
                raise StopIteration

    def stop(self):
        pass

    def __len__(self):
        return self.queue.qsize() if not self.is_stream else 0
