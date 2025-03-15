import torch
import numpy as np
from models.base_model import BaseModel

class FrameInterpolationModel(BaseModel):
    def create_model(self):
        pass

    def train(self, 
              weight_decay: float,
              learning_rate: float,
              num_epochs: int,
              device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        pass

    def inference(self, dataloader):
        pass