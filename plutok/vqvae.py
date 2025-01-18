import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.optim as optim
from sklearn.cluster import MiniBatchKMeans
import numpy as np
from vqtorch.nn import VectorQuant
import collections
import os
from tqdm import tqdm
        

class ResLayer(nn.Module):
    def __init__(self, h_dim):
        super(ResLayer, self).__init__()
        self.conv = nn.Conv1d(h_dim, h_dim, 3, 1, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x + self.relu(self.conv(x))
        return x


class VQVAE(nn.Module):
    def __init__(self, h_dim=384, n_embeddings=512, beta=0.25):
        super(VQVAE, self).__init__()
        self.h_dim = h_dim
        self.encoder = nn.Sequential(
            ResLayer(h_dim),
            nn.Conv1d(h_dim, 2 * h_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            ResLayer(2 * h_dim),
            ResLayer(2 * h_dim),
            ResLayer(2 * h_dim),
            ResLayer(2 * h_dim),
            nn.Conv1d(2 * h_dim, 2 * h_dim, kernel_size=3, stride=1, padding=1),
        )
        self.decoder = nn.Sequential(
            ResLayer(2 * h_dim),
            nn.ConvTranspose1d(2 * h_dim, h_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            ResLayer(h_dim),
            ResLayer(h_dim),
            ResLayer(h_dim),
            ResLayer(h_dim),
            nn.Conv1d(h_dim, h_dim, kernel_size=3, stride=1, padding=1),
        )
        self.centroids = None

    def get_device(self):
        # Check if the module has parameters or buffers
        try:
            return next(self.parameters()).device
        except StopIteration:
            try:
                return next(self.buffers()).device
            except StopIteration:
                return None  # No parameters or buffers

    def forward(self, x, use_vq=False):
        if not use_vq:
            z_e = self.encoder(x)
            x_hat = self.decoder(z_e + torch.randn_like(z_e))
            return x_hat
        else:
            ids = self.predict(x)
            x_hat = self.decode(ids, noise_std=0.5)
            return x_hat

    def predict(self, x):
        z_e = self.encoder(x)
        bs, c, l = z_e.shape
        z_e = z_e.permute(0, 2, 1).reshape(-1, 1, c)
        cnt = self.centroids.unsqueeze(0)
        diff = torch.sum((z_e - cnt)**2, dim=2)
        ids = torch.argmin(diff, dim=1)
        ids = ids.reshape(bs, l)
        return ids

    def decode(self, q, noise_std):
        bs, l = q.shape
        q = q.flatten()
        z_q = self.centroids[q]
        z_q = z_q.reshape(bs, l, -1).permute(0, 2, 1) # (b, c, l)
        x_hat = self.decoder(z_q + torch.randn_like(z_q) * noise_std)
        return x_hat

    def reconstruct(self, x):
        z_e = self.encoder(x)
        bs, c, t = z_e.shape
        z_e = z_e.permute(0, 2, 1).reshape(-1, c)
        z_q, vq_dict = self.vq_layer(z_e)
        z_q = z_q.permute(1, 0).unsqueeze(0)
        x_hat = self.decoder(z_q)
        return x_hat


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
        return o


def train_vqvae(
    emb_dir,
    batch_size=64,
    n_sample=1024,
    n_cluster=256,
    epochs=10,
    n_worker=16,
    save_dir="outputs/vqvae",
    device="cuda",
):
    dataset = EmbeddingDataset(emb_dir)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=n_worker
    )
    model = VQVAE(n_embeddings=n_cluster).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, amsgrad=True)

    model.train()
    for epoch in range(epochs//2):
        results = collections.defaultdict(list)
        for x in tqdm(dataloader, desc="epoch {}".format(epoch)):
            x = x.to(device).float()
            optimizer.zero_grad()
            x_hat = model(x)
            recon_loss = torch.mean((x_hat - x) ** 2) / 0.2
            loss = recon_loss
            loss.backward()
            optimizer.step()
            results["recon_errors"].append(recon_loss.cpu().detach().numpy())
            results["loss_vals"].append(loss.cpu().detach().numpy())
        for k, v in results.items():
            print("{}: {:.2f}".format(k, np.mean(v)))
    
    #save_path = os.path.join(save_dir, "model.pth")
    #torch.save(model.state_dict(), save_path)
    #print("Saved vqvae model to " + save_path)

    kmeans = MiniBatchKMeans(
        n_clusters=n_cluster, max_iter=1000, batch_size=batch_size * n_sample
    )

    model = model.eval()
    with torch.no_grad():
        for epoch in range(max(1, epochs//20)):
            for x in tqdm(dataloader, desc="kmeans epoch {}".format(epoch)):
                x = x.to(device).float()
                optimizer.zero_grad()
                z_e = model.encoder(x)
                bs, c, l = z_e.shape
                d = z_e.permute(0, 2, 1).reshape(-1, c).detach().cpu().numpy()
                kmeans.partial_fit(d)

    x_eval = x.clone()
    model.centroids = torch.tensor(kmeans.cluster_centers_.astype(np.float32)).to(device)
    ids = model.predict(x_eval)
    for noise_std in [0.1, 0.3, 0.5, 0.7, 0.9]:
        x_hat = model.decode(ids, noise_std)
        diff = torch.mean((x_hat - x_eval) ** 2) / 0.2
        print('at noise {:.2f}, est recon err: {:.2f}'.format(noise_std, diff.detach().cpu().numpy()))

    model.train()
    for param in model.encoder.parameters():
        param.requires_grad = False
    model.centroids.requires_grad = False
    for epoch in range(epochs//2):
        results = collections.defaultdict(list)
        for x in tqdm(dataloader, desc="epoch {}".format(epoch)):
            x = x.to(device).float()
            optimizer.zero_grad()
            x_hat = model(x, use_vq=True)
            recon_loss = torch.mean((x_hat - x) ** 2) / 0.2
            loss = recon_loss
            loss.backward()
            optimizer.step()
            results["recon_errors"].append(recon_loss.cpu().detach().numpy())
            results["loss_vals"].append(loss.cpu().detach().numpy())
        for k, v in results.items():
            print("{}: {:.2f}".format(k, np.mean(v)))

    model = model.eval()
    ids = model.predict(x_eval)
    for noise_std in [0.1, 0.3, 0.5, 0.7, 0.9]:
        x_hat = model.decode(ids, noise_std)
        diff = torch.mean((x_hat - x_eval) ** 2) / 0.2
        print('at noise {:.2f}, est recon err: {:.2f}'.format(noise_std, diff.detach().cpu().numpy()))
    
    save_dir = os.path.join(save_dir, "centroid_ids")
    os.makedirs(save_dir, exist_ok=True)
    for f in tqdm(dataset.emb_fs, desc="pred tok"):
        d = torch.tensor(np.load(open(f, "rb"))).float().to(model.get_device())
        ids = model.predict(d).flatten().detach().cpu().numpy().squeeze()
        text = ["#" + str(i) for i in ids]
        save_name = os.path.splitext(f.split("/")[-1])[0] + ".txt"
        save_path = os.path.join(save_dir, save_name)
        open(save_path, "w").write("".join(text))
