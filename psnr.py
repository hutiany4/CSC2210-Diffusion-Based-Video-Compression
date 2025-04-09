import torchvision
import torch
import torch.nn.functional as F
import cv2 as cv
import numpy as np

path = "./input"
filename = "basketball"

# h264 = torchvision.io.read_video(f"./output/{filename}/h264.avi", pts_unit="sec")[0].numpy()
h265 = torchvision.io.read_video(f"./output/{filename}/h265_{filename}.mp4", pts_unit="sec")[0].numpy()
diffusion = torchvision.io.read_video(f"./output/{filename}.mp4", pts_unit="sec")[0].numpy()

# h264_psnr = 0.0
h265_psnr = 0.0
diffusion_psnr = 0.0

for i in range(len(diffusion)):
    image = cv.imread(f"./output/{filename}/raw/{i}.png")
    # frame_h264 = cv.cvtColor(h264[i], cv.COLOR_BGR2RGB)
    frame_h265 = cv.cvtColor(h265[i], cv.COLOR_BGR2RGB)
    frame_diffusion = cv.cvtColor(diffusion[i], cv.COLOR_BGR2RGB)
    # h264_psnr += cv.PSNR(image, frame_h264)
    h265_psnr += cv.PSNR(image, frame_h265)
    diffusion_psnr += cv.PSNR(image, frame_diffusion)

# print(h264_psnr / len(h264))
print(h265_psnr / len(diffusion))
print(diffusion_psnr / len(diffusion))