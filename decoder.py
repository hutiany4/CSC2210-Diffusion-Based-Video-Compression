from src.models.image_models import ImageReconstructionModel
from src.models.video_models import FrameInterpolationModel
from src.data.dataset import ImageDataset, CompressedDataset
from torch.utils.data import DataLoader
import numpy as np
import torch
import gc
import torchvision
from pathlib import Path
import cv2 as cv

path = "./output"
filename = "basketball"

## Image prompt that is used to describing the input images
prompt = 'A man is playing basketball, photograph, ultra HD'
n_prompt = 'blurred, watermark, lowres, low quality, cartoon, unreal'

## Prompt used by CLIP model to calculate the score of the genereated images
## the image with the highest score will be chosen to be the final generated image
qc_prompt = 'A man is playing basketball, photograph, ultra HD'
qc_neg_prompt = 'blurred, watermark, lowres, low quality, cartoon, unreal'

image_model = ImageReconstructionModel()
interpolate_model = FrameInterpolationModel()

frames = torch.from_numpy(np.load(f"{path}/{filename}/frames.npy")).to(torch.uint16)

compressed_dataset = CompressedDataset(f"{path}/{filename}", frames)

print(f"key frames length: {len(compressed_dataset)}")


print(f"generating middel frames")
result = interpolate_model.inference(compressed_dataset, interpolate_type = "diffusion", prompt=prompt, n_prompt=n_prompt, qc_prompt=qc_prompt, qc_neg_prompt=qc_neg_prompt)


print("generation complete!")
print(f"final frames length: {len(result)}")
print("writing to video file")

Path(f"./output/{filename}_decoded").mkdir(parents=True, exist_ok=True)
for i in range(len(result)):
    cv.imwrite(f"./output/{filename}_decoded/{i}.png", result[i].numpy())

torchvision.io.write_video("./output/"+filename+".mp4", result, 24, options = {"crf": "0"})

print("writing complete!")