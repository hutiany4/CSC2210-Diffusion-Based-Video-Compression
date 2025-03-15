from src.models.image_models import ImageCompressionModel, AuxModel
from src.data.dataset import ImageDataset, CompressedDataset
from torch.utils.data import DataLoader
import numpy as np
import torchvision
import torch
from torch import Tensor

path = "./input"
filename = "basketball.mp4"
frame_rate = 16


compression_model = ImageCompressionModel()
aux_model = AuxModel()
video = torchvision.io.read_video(path+"/"+filename, pts_unit="sec")[0]

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
frames = torch.Tensor(frames)

if aux is None:
    aux = torch.Tensor([1] * len(video))

print(f"image length: {len(images)}")
print(f"aux length: {len(aux) if aux is not None else 0}")
print(f"frames length: {len(frames)}")

np.save("./input/basketball_images", Tensor.numpy(images, force=True))
np.save("./input/basketball_frames", Tensor.numpy(frames, force=True))
if aux is not None:
    np.save("./input/basketball_aux", Tensor.numpy(aux, force=True))
else:
    np.save("./input/basketball_aux", np.array([]))