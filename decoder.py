from src.models.image_models import ImageReconstructionModel
from src.models.video_models import FrameInterpolationModel
from src.data.dataset import ImageDataset, CompressedDataset
from torch.utils.data import DataLoader
import numpy as np
import torch
import gc
import torchvision

path = "./input"
filename = "sing"
frame_rate = 16

images = torch.from_numpy(np.load(path + "/" + filename + "_images.npy")).to(torch.uint8)
frames = torch.from_numpy(np.load(path + "/" + filename + "_frames.npy")).to(torch.uint8)
aux = torch.from_numpy(np.load(path + "/" + filename + "_aux.npy")).to(torch.uint8)

image_model = ImageReconstructionModel()
interpolate_model = FrameInterpolationModel()

uncompressed_images = []

for img in images:
    uncompressed_images.append(image_model.inference(img))

images = torch.stack(uncompressed_images, dim=0).to(torch.uint8)

print(f"key frames length: {len(images)}")
print(f"aux length: {len(aux) if aux is not None else 0}")

compressed_dataset = CompressedDataset(images, aux, frames)

while frame_rate > 1:
    print(f"generating {len(compressed_dataset)} middel frames")
    compressed_dataset.update(interpolate_model.inference(compressed_dataset))

    frame_rate = frame_rate // 2

print("generation complete!")
print(f"final frames length: {len(compressed_dataset)}")
print("writing to video file")

torchvision.io.write_video("./output/"+filename+".mp4", compressed_dataset.get_images(), 24, options = {"crf": "0"})

print("writing complete!")