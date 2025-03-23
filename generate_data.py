import torchvision
import torch
import cv2 as cv
import os, sys, json, argparse, re
from pathlib import Path
from tqdm import tqdm

parser = argparse.ArgumentParser(description ='Generate training dataset')
parser.add_argument('--file', nargs=1, type=str, help='filename of the video to process')
parser.add_argument('--prompt', nargs=1, type=str, help='text prompt related to this video')

arg = parser.parse_args()

video_file = arg.file[0]
prompt = arg.prompt[0]

input_path = f"./input/{video_file}"
output_path = "./output/data"

video = torchvision.io.read_video(input_path, pts_unit="sec")[0].to(torch.uint8).numpy()

# Path(f"{output_path}/images1").mkdir(parents=True, exist_ok=True)
# Path(f"{output_path}/images2").mkdir(parents=True, exist_ok=True)
Path(f"{output_path}/conditioning_images").mkdir(parents=True, exist_ok=True)
Path(f"{output_path}/images").mkdir(parents=True, exist_ok=True)

exist_images = os.listdir("./output/data/images")

index = 0
if len(exist_images) != 0:
    last_file = max(os.listdir("./output/data/images"), key=lambda x: int(re.search(r'\d+', x).group()))
    index = int(Path(last_file).stem) + 1

info = []
canny = []
for i in range(len(video)):
    video[i] = cv.cvtColor(video[i], cv.COLOR_RGB2BGR)
    canny.append(cv.Canny(video[i], 200, 255))
for i in tqdm(range(0, len(video), 2)):
    # cv.imwrite(f"{output_path}/images1/{index}.png", video[i])
    # cv.imwrite(f"{output_path}/images2/{index}.png", video[i+rate])
    cv.imwrite(f"{output_path}/conditioning_images/{index}.png", canny[i])
    cv.imwrite(f"{output_path}/images/{index}.png", video[i])

    info.append({"text": prompt,
                    "images": f"images/{index}.png",
                #  "images1": f"images1/{index}.png",
                #  "images2": f"images2/{index}.png",
                    "conditioning_images": f"conditioning_images/{index}.png"})
    
    index += 1

with open(f"{output_path}/train.jsonl", "a") as out:
    for line in info:
        out.write(json.dumps(line) + '\n')