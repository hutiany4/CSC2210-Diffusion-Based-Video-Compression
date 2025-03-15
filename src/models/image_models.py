import torch
import numpy as np
from models.base_model import BaseModel

class ImageCompressionModel(BaseModel):
    def create_model(self):
        pass

    def train(self, 
              weight_decay: float,
              learning_rate: float,
              num_epochs: int,
              device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        pass

    def inference(self, dataloader):
        output = np.array([])
        with torch.no_grad():
            self.model.eval()
            for data in dataloader:
                 # Tensors to gpu
                if torch.cuda.is_available():
                    data, target = data.cuda(), target.cuda()

                # Forward pass
                cur_output = self.model(data)

                output.append(torch.Tensor.numpy(cur_output, force=True))
        
        return output


class ImageReconstructionModel(BaseModel):
    def create_model(self):
        pass

    def train(self, 
              weight_decay: float,
              learning_rate: float,
              num_epochs: int,
              device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        pass

    def inference(self, dataloader):
        output = np.array([])
        with torch.no_grad():
            self.model.eval()
            for data in dataloader:
                 # Tensors to gpu
                if torch.cuda.is_available():
                    data, target = data.cuda(), target.cuda()

                # Forward pass
                cur_output = self.model(data)

                output.append(torch.Tensor.numpy(cur_output, force=True))
        
        return output

class AuxModel(BaseModel):
    def create_model(self):
        pass

    def train(self, 
              weight_decay: float,
              learning_rate: float,
              num_epochs: int,
              device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        pass

    def inference(self, dataloader):
        output = np.array([])
        with torch.no_grad():
            self.model.eval()
            for data in dataloader:
                 # Tensors to gpu
                if torch.cuda.is_available():
                    data, target = data.cuda(), target.cuda()

                # Forward pass
                cur_output = self.model(data)

                output.append(torch.Tensor.numpy(cur_output, force=True))
        
        return output