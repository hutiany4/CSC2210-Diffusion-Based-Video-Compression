from src.models.image_models import ImageReconstructionModel
from src.models.video_models import FrameInterpolationModel
from src.data.dataset import ImageDataset, CompressedDataset
from torch.utils.data import DataLoader
import numpy as np
import torch
import gc
import torchvision

path = "./input"
filename = "basketball"

image_model = ImageReconstructionModel()
interpolate_model = FrameInterpolationModel()

frames = torch.from_numpy(np.load(f"{path}/{filename}/frames.npy")).to(torch.uint8)

compressed_dataset = CompressedDataset(f"{path}/{filename}", frames)

print(f"key frames length: {len(compressed_dataset)}")


print(f"generating middel frames")
result = interpolate_model.inference(compressed_dataset)


print("generation complete!")
print(f"final frames length: {len(compressed_dataset)}")
print("writing to video file")

torchvision.io.write_video("./output/"+filename+".mp4", result, 24, options = {"crf": "0"})

print("writing complete!")