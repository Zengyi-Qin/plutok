import numpy as np
import os
from sklearn.cluster import MiniBatchKMeans
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
import json


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
        s = np.random.randint(t, size=self.n_sample)
        d = d[0].T  # (t, c)
        return d[s]


def train_kmeans(
    emb_dir,
    batch_size=256,
    n_sample=128,
    n_cluster=256,
    epochs=10,
    n_worker=16,
    save_dir="outputs/kmeans",
):
    dataset = EmbeddingDataset(emb_dir)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=n_worker
    )
    kmeans = MiniBatchKMeans(
        n_clusters=n_cluster, max_iter=1000, batch_size=batch_size * n_sample
    )
    for epoch in range(epochs):
        for d in tqdm(dataloader, desc="epoch {}".format(epoch)):
            d = d.numpy()
            bs, n, c = d.shape
            d = d.reshape(-1, c)
            kmeans.partial_fit(d)

    centroids = kmeans.cluster_centers_
    res = {"centroids": centroids.tolist()}
    save_path = os.path.join(save_dir, "centroids.json")
    json.dump(res, open(save_path, "w"), indent=2)
    print("Saved centroids to " + save_path)

    save_dir = os.path.join(save_dir, "centroid_ids")
    os.makedirs(save_dir, exist_ok=True)
    for f in tqdm(dataset.emb_fs, desc="pred tok"):
        d = np.load(open(f, "rb"))[0].T
        ids = kmeans.predict(d)
        text = ["#" + str(i) for i in ids]
        save_name = os.path.splitext(f.split("/")[-1])[0] + ".txt"
        save_path = os.path.join(save_dir, save_name)
        open(save_path, "w").write("".join(text))
