import torchvision
import torch
import torch.nn.functional as F
import cv2 as cv
import numpy as np

path = "./input"
filename = "basketball"

video = torchvision.io.read_video(path+"/"+filename+".mp4", pts_unit="sec")[0]
video = video.permute(0, 3, 1, 2)
resized_video = F.interpolate(video, size=(960, 1280), mode='bilinear', align_corners=False)
resized_video = resized_video.round().clamp(0, 255).to(torch.uint8)
video = resized_video.permute(0, 2, 3, 1)
video = video.numpy()


out_264 = cv.VideoWriter(f"./output/{filename}/h264.avi", cv.VideoWriter_fourcc('D','I','V','X'), 24.0, (1280, 960))
out_264.set(cv.VIDEOWRITER_PROP_QUALITY, 0.5)
out_265 = cv.VideoWriter(f"./output/{filename}/h265.mp4", cv.VideoWriter_fourcc('H', 'E', 'V', 'C'), 24.0, (1280, 960))
out_265.set(cv.VIDEOWRITER_PROP_QUALITY, 0.5)

for i, frame in enumerate(video):
    frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
    cv.imwrite(f"./output/{filename}/raw/{i}.png", frame)
    out_264.write(frame)
    out_265.write(frame)

out_264.release()
out_265.release()