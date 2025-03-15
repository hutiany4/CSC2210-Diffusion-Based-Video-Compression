from typing import Dict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

class BaseModel():
    """
    A class used to store an arbitrary model
    ...

    Attributes
    ----------
    model : nn.Module
        the module to use
    dataloaders : Dict[str, DataLoader]
        A dictionary containing training, validation, and test dataloaders
    """

    model: nn.Module
    dataloader: Dict[str, DataLoader]

    def __init__(self):
        self.model = None
        self.dataloaders = {"train": None, "val": None, "test": None}

    def create_model(self):
        pass

    def set_dataloaders(self, train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader) -> None:
        self.dataloaders["train"] = train_loader
        self.dataloaders["val"] = val_loader
        self.dataloaders["test"] = test_loader

    def train(self, 
              weight_decay: float,
              learning_rate: float,
              num_epochs: int,
              device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        pass

    def inference(self):
        pass

    def save(self, path: str, filename: str):
        torch.save(self.model.state_dict(), path+"/"+filename)

    def load(self, path: str, filename: str):
        self.model.load_state_dict(torch.load(path+"/"+filename, weights_only=True))
