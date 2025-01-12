import os
import numpy as np
import torch
from torch.utils.data import Dataset
from denoising_diffusion_pytorch import Unet1D, GaussianDiffusion1D, Trainer1D, Dataset1D

model = Unet1D(
    channels=192,
    dim=192,
    dim_mults=(1, 2, 4, 8),
    self_condition=False
)
model.load_state_dict(torch.load('diffusion.pth'))
model.cuda()

diffusion = GaussianDiffusion1D(
    model,
    seq_length = 512,
    timesteps = 1000,
    objective = 'pred_v',
    auto_normalize = False
)
diffusion.cuda()

sampled_seq = diffusion.sample(batch_size = 4)
import pdb;pdb.set_trace()
sampled_seq.shape