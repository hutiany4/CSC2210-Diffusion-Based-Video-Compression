from models.image_models import ImageReconstructionModel
from models.video_models import FrameInterpolationModel
from data.dataset import CompressedDataset
from torch.utils.data import DataLoader
import numpy as np
import torch

def decoder(images, aux, frames, frame_rate):
    image_model = ImageReconstructionModel()
    interpolate_model = FrameInterpolationModel()

    uncompressed_images = []

    for img in images:
        uncompressed_images.append(image_model(img))
    
    uncompressed_images = torch.stack(uncompressed_images, dim=0)

    while frame_rate > 1:
        compressed_dataset = CompressedDataset(images, aux, frames)
        loader = DataLoader(compressed_dataset)
        updated_images = []
        updated_frames = []
        for data in loader:
            updated_images.append(data[0][0])
            updated_frames.append(data[2][0])

            interpolated_image = interpolate_model.inference(data[0], data[1])
            updated_images.append(interpolated_image)
            updated_frames.append((data[2][0] + data[2][1]) // 2)
        
        updated_images.append(compressed_dataset[-1][0][1])
        updated_frames.append((compressed_dataset[-1][2][1] + compressed_dataset[-1][2][1]) // 2)

        images = torch.stack(updated_images, dim=1)
        frames = torch.Tensor(updated_frames)

        frame_rate = frame_rate // 2
        
    return images
        
