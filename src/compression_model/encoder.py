from models.image_models import ImageCompressionModel, AuxModel
from data.dataset import ImageDataset
from torch.utils.data import DataLoader
import numpy as np

def encoder(path, filename, frame_rate):
    compression_model = ImageCompressionModel()
    aux_model = AuxModel()
    video_data = ImageDataset(path, filename)
    loader = DataLoader(video_data)

    images = []
    frames = []
    aux = []

    for i, data in enumerate(loader):
        if i % frame_rate == 0:
            images.append(compression_model.inference(data))
            frames.append(i)
        aux.append(aux_model.inference(data))

    return np.array(images), np.array(frames), np.array(aux)