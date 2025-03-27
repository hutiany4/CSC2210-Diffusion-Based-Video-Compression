from typing import List
import torchvision
import einops
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

        frame_tensor = []
        
        ## Add first frame tensor into the list
        frame1_tensor = torch.from_numpy(np.array(keyframe1)).to(torch.uint8).permute(2,0,1)
        frame_tensor.append(frame1_tensor)

        for i in range(int(self.__frames[index])+1, int(self.__frames[index+1])):
            aux_img = cv.imread(f"{self.__dir}/auxiliary/{i}.png", cv.IMREAD_GRAYSCALE)
            aux_img = np.expand_dims(aux_img, axis=-1)
            control = torch.from_numpy(aux_img).to(torch.uint8).cuda().repeat(1, 1, 3) / 255.0
            control = torch.stack([control for _ in range(1)], dim=0)
            control = einops.rearrange(control, 'b h w c -> b c h w').clone()
            print("The canny image has the dimension:")
            print(control.shape)
            frame_tensor.append(control)
        
        ## Add last frame tensor into the list
        frame2_tensor = torch.from_numpy(np.array(keyframe2)).to(torch.uint8).permute(2,0,1)
        frame_tensor.append(frame2_tensor)
        
        return frame_tensor
    

if __name__ == "__main__":
    pass
