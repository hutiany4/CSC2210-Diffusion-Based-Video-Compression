from src.models.image_models import ImageCompressionModel, AuxModel
from src.data.dataset import ImageDataset, CompressedDataset
from torch.utils.data import DataLoader
from pathlib import Path
import os
import numpy as np
import torchvision
import torch
from torch import Tensor
import cv2 as cv

path = "./input"
filename = "basketball"
frame_rate = 8


compression_model = ImageCompressionModel()
aux_model = AuxModel()
video = torchvision.io.read_video(path+"/"+filename+".mp4", pts_unit="sec")[0].to(torch.uint8)

print(f"video length: {len(video)}")

key_frames = []
frames = []

for i in range(len(video)):
    if i % frame_rate == 0:
        key_frames.append(video[i])
        frames.append(i)

key_frames = torch.stack(key_frames, dim=0)

print(f"key frame length: {len(key_frames)}")

key_frame_data = ImageDataset(key_frames)
video_data = ImageDataset(video)

images = compression_model.inference(key_frame_data)
aux = aux_model.inference(video_data)

print(f"image length: {len(images)}")
print(f"aux length: {len(aux) if aux is not None else 0}")
print(f"frames length: {len(frames)}")

Path(f"./output/{filename}/keyframes").mkdir(parents=True, exist_ok=True)
Path(f"./output/{filename}/auxiliary").mkdir(parents=True, exist_ok=True)

for i in range(len(images)):
    cv.imwrite(f"./output/{filename}/keyframes/{i}.png", cv.cvtColor(images[i].numpy(), cv.COLOR_RGB2BGR))

np.save(f"./output/{filename}/frames", frames)

if aux is not None:
    for i in range(len(aux)):
        cv.imwrite(f"./output/{filename}/auxiliary/{i}.png", aux[i])