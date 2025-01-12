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

class _VectorQuantizer(nn.Module):
    """
    Discretization bottleneck part of the VQ-VAE.

    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    """

    def __init__(self, n_e, e_dim, beta):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def get_device(self):
        # Check if the module has parameters or buffers
        try:
            return next(self.parameters()).device
        except StopIteration:
            try:
                return next(self.buffers()).device
            except StopIteration:
                return None  # No parameters or buffers

    def forward(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete
        one-hot vector that is the index of the closest embedding vector e_j

        z (continuous) -> z_q (discrete)

        z.shape = (batch, channel, height, width)

        quantization pipeline:

            1. get encoder input (B,C,T)
            2. flatten input to (B*T,C)

        """
        # reshape z -> (batch, time, channel) and flatten
        z = z.permute(0, 2, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(z_flattened, self.embedding.weight.t())
        )

        # find closest encodings
        device = self.get_device()
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.n_e).to(device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        # compute loss for embedding
        loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean(
            (z_q - z.detach()) ** 2
        )

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape back to match original input shape
        z_q = z_q.permute(0, 2, 1).contiguous()

        return loss, z_q, perplexity, min_encodings, min_encoding_indices

    def predict(self, z):
        # reshape z -> (batch, time, channel) and flatten
        z = z.permute(0, 2, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(z_flattened, self.embedding.weight.t())
        )

        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        return min_encoding_indices


class _VQVAE(nn.Module):
    def __init__(self, h_dim=192, n_embeddings=128, beta=0.25):
        super(VQVAE, self).__init__()
        # encode image into continuous latent space
        self.encoder = nn.Sequential(
            nn.Conv1d(h_dim, h_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(h_dim, h_dim, kernel_size=3, stride=1, padding=1),
        )
        self.vector_quantization = VectorQuantizer(n_embeddings, h_dim, beta)
        # decode the discrete latent representation
        self.decoder = nn.Sequential(
            nn.Conv1d(h_dim, h_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(h_dim, h_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(h_dim, h_dim, kernel_size=3, stride=1, padding=1),
        )

    def get_device(self):
        # Check if the module has parameters or buffers
        try:
            return next(self.parameters()).device
        except StopIteration:
            try:
                return next(self.buffers()).device
            except StopIteration:
                return None  # No parameters or buffers

    def forward(self, x):
        z_e = self.encoder(x)
        embedding_loss, z_q, perplexity, _, _ = self.vector_quantization(z_e)
        # x_hat = self.decoder(z_q)
        x_hat = self.decoder(z_e)

        return embedding_loss, x_hat, perplexity

    def predict(self, x):
        z_e = self.encoder(x)
        ids = self.vector_quantization.predict(z_e)
        return ids


class ResLayer(nn.Module):
    def __init__(self, h_dim):
        super(ResLayer, self).__init__()
        self.conv = nn.Conv1d(h_dim, h_dim, 3, 1, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x + self.relu(self.conv(x))
        return x


class VQVAE(nn.Module):
    def __init__(self, h_dim=192, n_embeddings=128, beta=0.25):
        super(VQVAE, self).__init__()
        # encode image into continuous latent space
        self.encoder = nn.Sequential(
            ResLayer(h_dim),
            nn.Conv1d(h_dim, 2 * h_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            ResLayer(2 * h_dim),
            ResLayer(2 * h_dim),
            ResLayer(2 * h_dim),
            ResLayer(2 * h_dim),
            nn.Conv1d(2 * h_dim, 4 * h_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            ResLayer(4 * h_dim),
            ResLayer(4 * h_dim),
            ResLayer(4 * h_dim),
            ResLayer(4 * h_dim),
            nn.Conv1d(4 * h_dim, 4 * h_dim, kernel_size=3, stride=1, padding=1),
        )
        #self.vector_quantization = VectorQuantizer(n_embeddings, 4 * h_dim, beta)
        # decode the discrete latent representation
        self.decoder = nn.Sequential(
            ResLayer(4 * h_dim),
            ResLayer(4 * h_dim),
            nn.ConvTranspose1d(4 * h_dim, h_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            ResLayer(h_dim),
            ResLayer(h_dim),
            ResLayer(h_dim),
            ResLayer(h_dim),
            ResLayer(h_dim),
            nn.Conv1d(h_dim, h_dim, kernel_size=3, stride=1, padding=1),
        )

        self.vq_layer = VectorQuant(
                feature_size=4 * h_dim,     # feature dimension corresponding to the vectors
                num_codes=n_embeddings,      # number of codebook vectors
                beta=0.98,           # (default: 0.9) commitment trade-off
                kmeans_init=False,    # (default: False) whether to use kmeans++ init
                norm=None,           # (default: None) normalization for the input vectors
                cb_norm=None,        # (default: None) normalization for codebook vectors
                affine_lr=10.0,      # (default: 0.0) lr scale for affine parameters
                sync_nu=0.2,         # (default: 0.0) codebook synchronization contribution
                replace_freq=20,     # (default: None) frequency to replace dead codes
                dim=-1,              # (default: -1) dimension to be quantized
                )

    def get_device(self):
        # Check if the module has parameters or buffers
        try:
            return next(self.parameters()).device
        except StopIteration:
            try:
                return next(self.buffers()).device
            except StopIteration:
                return None  # No parameters or buffers

    def forward(self, x, stage):
        z_e = self.encoder(x)
        # embedding_loss, z_q, perplexity, _ = self.vector_quantization(z_e)
        bs, c, t = z_e.shape
        z_e = z_e.permute(0, 2, 1).reshape(-1, c)
        z_q, vq_dict = self.vq_layer(z_e)
        z_q = z_q.reshape(bs, t, c).permute(0, 2, 1)
        x_hat = self.decoder(z_q)
        return vq_dict['loss'], x_hat, vq_dict['perplexity']

    def predict(self, x):
        z_e = self.encoder(x)
        bs, c, t = z_e.shape
        z_e = z_e.permute(0, 2, 1).reshape(-1, c)
        z_q, vq_dict = self.vq_layer(z_e)
        return vq_dict['q']

    def decode(self, q):
        assert len(q.shape) == 1
        z_q = self.vq_layer.codebook(q)
        z_q = z_q.unsqueeze(0).permute(0, 2, 1)
        x_hat = self.decoder(z_q)
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
    for epoch in range(epochs):
        results = collections.defaultdict(list)
        for x in tqdm(dataloader, desc="epoch {}".format(epoch)):
            x = x.to(device).float()
            optimizer.zero_grad()
            embedding_loss, x_hat, perplexity = model(x, stage='2')
            recon_loss = torch.mean((x_hat - x) ** 2) / 0.2
            loss = recon_loss + embedding_loss
            loss.backward()
            optimizer.step()

            results["embedding_loss"].append(embedding_loss.cpu().detach().numpy())
            results["recon_errors"].append(recon_loss.cpu().detach().numpy())
            results["perplexities"].append(perplexity.cpu().detach().numpy())
            results["loss_vals"].append(loss.cpu().detach().numpy())
        for k, v in results.items():
            print("{}: {:.2f}".format(k, np.mean(v)))
    
    save_path = os.path.join(save_dir, "model.pth")
    torch.save(model.state_dict(), save_path)
    print("Saved vqvae model to " + save_path)

    save_dir = os.path.join(save_dir, "centroid_ids")
    os.makedirs(save_dir, exist_ok=True)
    model = model.eval()
    for f in tqdm(dataset.emb_fs, desc="pred tok"):
        d = torch.tensor(np.load(open(f, "rb"))).float().to(model.get_device())
        ids = model.predict(d).detach().cpu().numpy().squeeze()
        text = ["#" + str(i) for i in ids]
        save_name = os.path.splitext(f.split("/")[-1])[0] + ".txt"
        save_path = os.path.join(save_dir, save_name)
        open(save_path, "w").write("".join(text))
