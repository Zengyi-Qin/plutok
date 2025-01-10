import os
from plutok.bpe import train_bpe
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--text_dir')
    parser.add_argument('--vocab_size', type=int)
    parser.add_argument('--save_dir')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    train_bpe(args.text_dir, args.vocab_size, args.save_dir)
