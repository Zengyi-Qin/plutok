import os
import torch
from plutok.vqvae import train_vqvae
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb_dir")
    parser.add_argument("--n_cluster", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--save_dir", default="./outputs/vqvae")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    torch.set_float32_matmul_precision('high')
    train_vqvae(
        args.emb_dir,
        batch_size=64,
        n_sample=512,
        n_cluster=args.n_cluster,
        epochs=args.epochs,
        n_worker=26,
        save_dir=args.save_dir,
    )
