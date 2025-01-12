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

diffusion = GaussianDiffusion1D(
    model,
    seq_length = 512,
    timesteps = 1000,
    objective = 'pred_v',
    auto_normalize = False
)

class EmbeddingDataset(Dataset):

    def __init__(self, emb_dir, n_sample=512):
        emb_fs = os.listdir(emb_dir)
        self.emb_fs = [os.path.join(emb_dir, f) for f in emb_fs]
        self.emb_fs.sort()
        self.n_sample = n_sample

    def __len__(self):
        return len(self.emb_fs)

    def __getitem__(self, idx):
        d = np.load(open(self.emb_fs[idx], "rb"))
        n, c, t = d.shape
        high = max(0, t - self.n_sample)
        s = 0 if high == 0 else np.random.randint(high)
        o = np.zeros((c, self.n_sample))
        d = d[0, :, s : s + self.n_sample]
        o[:, : d.shape[1]] = d
        return o.astype(np.float16)

dataset = EmbeddingDataset('outputs/emb')

trainer = Trainer1D(
    diffusion,
    dataset = dataset,
    train_batch_size = 256,
    train_lr = 8e-5,
    train_num_steps = 120000,         # total training steps
    gradient_accumulate_every = 1,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                       # turn on mixed precision
)
trainer.train()
torch.save(model.state_dict(), 'diffusion.pth')
