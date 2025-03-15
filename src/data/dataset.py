import torchvision
import torch
from torch import Tensor
import numpy as np
from torch.utils.data import Dataset
from matplotlib import pyplot as plt

class ImageDataset(Dataset):

    __images: Tensor

    def __init__(self, path, filename):
        video = torchvision.io.read_video(path+"/"+filename)
        self.__images = video[0]

    def __len__(self):
        return len(self.__images)

    def __getitem__(self, index):
        return self.__images[index]

class CompressedDataset(Dataset):
    __images: Tensor
    __aux: Tensor = None
    __frames: Tensor

    def __init__(self, images, aux, frames):
        image_tuples = np.zeros([images.shape[0] - 1, 2, images.shape[1], images.shape[2]])
        frame_tuples = np.zeros([images.shape[0] - 1, 2])
        relate_aux = np.array([])

        for i in (len(images) - 1):
            image_tuples[i] = images[i:i+2,:,:]
            frame_tuples[i] = frames[i:i+2]
            if aux:
                relate_aux = np.append(relate_aux, aux[(frames[i] + frames[i+1]) // 2])
        
        self.__frames = torch.from_numpy(frame_tuples)
        self.__images = torch.from_numpy(image_tuples)
        if aux:
            self.__aux = torch.from_numpy(relate_aux)


    def __len__(self):
        return len(self.__images)

    def __getitem__(self, index):
        if self.__aux:
            return self.__images[index], self.__aux[index], self.__frames[index]
        else:
            return self.__images[index], None, self.__frames[index]
    

def __test_image_dataset(path, filename):
    data = ImageDataset(path, filename)
    print(len(data))
    plt.imshow(data[0])
    plt.show()
    

if __name__ == "__main__":
    __test_image_dataset("./input", "basketball.mp4")


