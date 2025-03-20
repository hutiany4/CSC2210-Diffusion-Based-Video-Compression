from typing import Union
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
import numpy as np
from src.models.base_model import BaseModel
import cv2 as cv

class ImageCompressionModel(BaseModel):
    def inference(self, dataset: Dataset) -> Tensor:
        output = []
        for i in range(len(dataset)):
            output.append(dataset[i])
        
        return torch.stack(output, dim=0)


class ImageReconstructionModel(BaseModel):
    def inference(self, dataset: Dataset) -> Tensor:
        output = []
        for i in range(len(dataset)):
            output.append(dataset[i])
        
        return torch.stack(output, dim=0)

class AuxModel(BaseModel):

    def inference(self, dataset: Dataset) -> Union[Tensor, None]:
        output = []
        for data in dataset:
            output.append(cv.Canny(data.numpy(), 220, 255))
        
        return np.array(output)