import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from typing import Tuple, List
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
import numpy as np
from src.models.base_model import BaseModel
import cm

class FrameInterpolationModel(BaseModel):
    def create_model(self):
        pass

    def train(self, 
              weight_decay: float,
              learning_rate: float,
              num_epochs: int,
              device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        pass

    def inference(self, dataset: Dataset, interpolate_type="naive", prompt=None, n_prompt=None, qc_prompt=None, qc_neg_prompt=None) -> Tuple[List[Tensor], List]:
        result = torch.empty([1, dataset[0][0].shape[1], dataset[0][0].shape[2], dataset[0][0].shape[0]]).to(torch.uint8)
        result[0] = dataset[0][0].permute(1,2,0)

        if interpolate_type == "naive":
            for i in range(len(dataset)):
                data = dataset[i]
                interpolation_result = torch.empty([8, data[0].shape[1], data[0].shape[2], data[0].shape[0]]).to(torch.uint8)
                keyframe1 = data[0].permute(1,2,0)
                keyframe2 = data[0].permute(1,2,0)
                interpolation_result[3] = 0.5 * keyframe1 + 0.5 * keyframe2
                interpolation_result[1] = 0.5 * keyframe1 + 0.5 * interpolation_result[3]
                interpolation_result[5] = 0.5 * interpolation_result[3] + 0.5 * keyframe2
                interpolation_result[0] = 0.5 * keyframe1 + 0.5 * interpolation_result[1]
                interpolation_result[2] = 0.5 * interpolation_result[1] + 0.5 * interpolation_result[3]
                interpolation_result[4] = 0.5 * interpolation_result[3] + 0.5 * interpolation_result[5]
                interpolation_result[6] = 0.5 * interpolation_result[5] + 0.5 * keyframe2
                interpolation_result[7] = keyframe2

                result = torch.cat((result, interpolation_result), dim=0)
        else:
            CM = cm.ContextManager(version='1.5')

            for i in range(len(dataset)):
                data = dataset[i]
                interpolation_result = torch.empty([8, result[0].shape[0], result[0].shape[1], result[0].shape[2]]).to(torch.uint8)
                CM.interpolate_qc(data[0], data[8], n_choices=4, controls=data, control_type='canny', qc_prompts=(qc_prompt, qc_neg_prompt), cond_path='data/road2500.pt', prompt=prompt, n_prompt=n_prompt, optimize_cond=500, ddim_steps=100, num_frames=9, guide_scale=7.5, schedule_type='linear', interpolate_result=interpolation_result, index=i)
                interpolation_result[7] = data[8].permute(1,2,0)

                result = torch.cat((result, interpolation_result), dim=0)
                print("After torch.cat the result becomes: ")
                print(result.shape)

        return result
