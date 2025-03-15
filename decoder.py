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
frame_rate = 8

images = torch.from_numpy(np.load(path + "/" + filename + "_images.npy"))
frames = torch.from_numpy(np.load(path + "/" + filename + "_frames.npy"))
aux = torch.from_numpy(np.load(path + "/" + filename + "_aux.npy"))

image_model = ImageReconstructionModel()
interpolate_model = FrameInterpolationModel()

uncompressed_images = []

for img in images:
    uncompressed_images.append(image_model.inference(img))

images = torch.stack(uncompressed_images, dim=0)

print(f"image length: {len(images)}")
print(f"aux length: {len(aux) if aux is not None else 0}")
print(f"frames length: {len(frames)}")

compressed_dataset = CompressedDataset(images, aux, frames)

while frame_rate > 1:
    compressed_dataset.update(interpolate_model.inference(compressed_dataset))

    frame_rate = frame_rate // 2
    
torchvision.io.write_video("./output/basketball.mp4", compressed_dataset.get_images(), 12)
