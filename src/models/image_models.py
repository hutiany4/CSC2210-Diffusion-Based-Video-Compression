from typing import Union
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
import numpy as np
from src.models.base_model import BaseModel

class ImageCompressionModel(BaseModel):
    def create_model(self):
        pass

    def train(self, 
              weight_decay: float,
              learning_rate: float,
              num_epochs: int,
              device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        pass

    def inference(self, dataset: Dataset) -> Tensor:
        # output = np.array([])
        # with torch.no_grad():
        #     self.model.eval()
        #     for data in dataloader:
        #          # Tensors to gpu
        #         if torch.cuda.is_available():
        #             data, target = data.cuda(), target.cuda()

        #         # Forward pass
        #         cur_output = self.model(data)

        #         output.append(torch.Tensor.numpy(cur_output, force=True))

        output = []
        for i in range(len(dataset)):
            output.append(dataset[i])
        
        return torch.stack(output, dim=0)


class ImageReconstructionModel(BaseModel):
    def create_model(self):
        pass

    def train(self, 
              weight_decay: float,
              learning_rate: float,
              num_epochs: int,
              device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        pass

    def inference(self, dataset: Dataset) -> Tensor:
        # output = np.array([])
        # with torch.no_grad():
        #     self.model.eval()
        #     for data in dataloader:
        #          # Tensors to gpu
        #         if torch.cuda.is_available():
        #             data, target = data.cuda(), target.cuda()

        #         # Forward pass
        #         cur_output = self.model(data)

        #         output.append(torch.Tensor.numpy(cur_output, force=True))
        
        output = []
        for i in range(len(dataset)):
            output.append(dataset[i])
        
        return torch.stack(output, dim=0)

class AuxModel(BaseModel):
    def create_model(self):
        pass

    def train(self, 
              weight_decay: float,
              learning_rate: float,
              num_epochs: int,
              device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        pass

    def inference(self, dataset: Dataset) -> Union[Tensor, None]:
        # output = np.array([])
        # with torch.no_grad():
        #     self.model.eval()
        #     for data in dataloader:
        #          # Tensors to gpu
        #         if torch.cuda.is_available():
        #             data, target = data.cuda(), target.cuda()

        #         # Forward pass
        #         cur_output = self.model(data)

        #         output.append(torch.Tensor.numpy(cur_output, force=True))
        
        return None