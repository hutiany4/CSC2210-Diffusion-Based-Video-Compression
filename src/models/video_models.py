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
        result = torch.empty([1, dataset[0][0][0].shape[0], dataset[0][0][0].shape[1], dataset[0][0][0].shape[2]]).to(torch.uint8)
        result[0] = dataset[0][0][0]
        for i in range(len(dataset)):
            data = dataset[i]
            interpolation_result = torch.empty([8, data[0][0].shape[0], data[0][0].shape[1], data[0][0].shape[2]]).to(torch.uint8)
            interpolation_result[3] = 0.5 * data[0][0] + 0.5 * data[0][1]
            interpolation_result[1] = 0.5 * data[0][0] + 0.5 * interpolation_result[3]
            interpolation_result[5] = 0.5 * interpolation_result[3] + 0.5 * data[0][1]
            interpolation_result[0] = 0.5 * data[0][0] + 0.5 * interpolation_result[1]
            interpolation_result[2] = 0.5 * interpolation_result[1] + 0.5 * interpolation_result[3]
            interpolation_result[4] = 0.5 * interpolation_result[3] + 0.5 * interpolation_result[5]
            interpolation_result[6] = 0.5 * interpolation_result[5] + 0.5 * data[0][1]
            interpolation_result[7] = data[0][1]

            result = torch.cat((result, interpolation_result), dim=0)


        return result
