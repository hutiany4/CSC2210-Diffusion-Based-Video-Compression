from typing import Tuple, List
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
import numpy as np
from src.models.base_model import BaseModel

class FrameInterpolationModel(BaseModel):
    def create_model(self):
        pass

    def train(self, 
              weight_decay: float,
              learning_rate: float,
              num_epochs: int,
              device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        pass

    def inference(self, dataset: Dataset) -> Tuple[List[Tensor], List]:
        new_images = []
        new_frames = []
        for i in range(len(dataset)):
            data = dataset[i]
            interpolated_image = 0.5 * data[0][0] + 0.5 * data[0][1]
            
            new_images.append(interpolated_image)
            new_frames.append((data[2][0] + data[2][1]) // 2)

        return new_images, new_frames
