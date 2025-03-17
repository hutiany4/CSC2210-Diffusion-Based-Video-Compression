import torchvision
import torch
from torch import Tensor
import numpy as np
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
import gc

class ImageDataset(Dataset):

    __images: Tensor

    def __init__(self, video: Tensor):
        self.__images = video

    def __len__(self):
        return len(self.__images)

    def __getitem__(self, index: int):
        return self.__images[index]

class CompressedDataset(Dataset):
    __images: Tensor
    __aux: Tensor = None
    __frames: Tensor

    def __init__(self, images: Tensor, aux: Tensor, frames: Tensor):
        self.__images = images
        self.__frames = frames
        if aux is not None:
            self.__aux = aux


    def __len__(self):
        return len(self.__images) - 1

    def __getitem__(self, index: int):
        if self.__aux is not None:
            aux_index = (self.__frames[index].item() + self.__frames[index+1].item()) // 2
            return self.__images[index:index+2,:,:], self.__aux[aux_index], self.__frames[index:index+2]
        else:
            return self.__images[index:index+2,:,:], None, self.__frames[index:index+2]
    
    def update(self, new_data):
        cur_images = torch.empty([self.__images.shape[0]+ len(new_data[0]), self.__images.shape[1], self.__images.shape[2], self.__images.shape[3]], dtype=torch.uint8)
        cur_frames = torch.empty([self.__images.shape[0]+ len(new_data[0])], dtype=torch.uint8)

        for i in range(len(self.__images)):
            cur_images[i*2] = self.__images[i]
            cur_frames[i*2] = self.__frames[i]
        
        del self.__images
        del self.__frames
        gc.collect()

        for i in range(len(new_data[0])):
            cur_images[i*2+1] = new_data[0][i]
            cur_frames[i*2+1] = new_data[1][i]
        
        del new_data
        gc.collect()

        self.__images = cur_images
        self.__frames = cur_frames
    
    def get_images(self):
        return self.__images
    
    def delete(self):
        del self.__images
        del self.__frames
        del self.__aux
    

def __test_image_dataset(path, filename):
    video = torchvision.io.read_video(path+"/"+filename)[0]
    data = ImageDataset(video)
    print(len(data))
    plt.imshow(data[0])
    plt.show()
    

if __name__ == "__main__":
    __test_image_dataset("./input", "basketball.mp4")


