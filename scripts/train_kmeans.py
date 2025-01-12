import os
from plutok.kmeans import train_kmeans
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb_dir")
    parser.add_argument("--n_cluster", type=int, default=256)
    parser.add_argument("--save_dir")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    train_kmeans(
        args.emb_dir,
        batch_size=256,
        n_sample=128,
        n_cluster=args.n_cluster,
        epochs=10,
        n_worker=16,
        save_dir=args.save_dir,
    )
