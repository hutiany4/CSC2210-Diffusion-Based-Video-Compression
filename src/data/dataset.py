from typing import List
import torchvision
import torch
from torch import Tensor
import numpy as np
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
import gc, os
import cv2 as cv

class ImageDataset(Dataset):

    __images: Tensor

    def __init__(self, video: Tensor):
        self.__images = video

    def __len__(self):
        return len(self.__images)

    def __getitem__(self, index: int):
        return self.__images[index]

class CompressedDataset(Dataset):
    __dir: str
    __frames: np.ndarray

    def __init__(self, path: str, frames: np.ndarray):
        self.__dir = path
        self.__frames = frames


    def __len__(self):
        return len(self.__frames) - 1

    def __getitem__(self, index: int):
        if index >= self.__len__():
            return None
        keyframe1 = cv.cvtColor(cv.imread(f"{self.__dir}/keyframes/{self.__frames[index]}.png"), cv.COLOR_BGR2RGB)
        keyframe2 = cv.cvtColor(cv.imread(f"{self.__dir}/keyframes/{self.__frames[index+1]}.png"), cv.COLOR_BGR2RGB)

        aux = []

        for i in range(self.__frames[index]+1, self.__frames[index+1]):
            aux.append(cv.imread(f"{self.__dir}/auxiliary/{i}.png", cv.IMREAD_GRAYSCALE))
        
        assert(len(aux) == 7)
        
        return Tensor(np.array([keyframe1, keyframe2])).to(torch.uint8), Tensor(np.array(aux)).to(torch.uint8)
    

if __name__ == "__main__":
    pass
