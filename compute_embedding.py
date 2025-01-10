import torch
from plutok.extract_embedding import extract_batch
from multiprocessing import Pool
import argparse
import os


def list_audio_files(directory):
    # Define the common audio file extensions
    audio_extensions = {".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a", ".wma", ".aiff"}
    audio_files = []

    # Walk through the directory and subdirectories
    for root, _, files in os.walk(directory):
        for file in files:
            # Check if the file has a valid audio extension
            if os.path.splitext(file)[1].lower() in audio_extensions:
                # Append the full path of the file to the list
                audio_files.append(os.path.join(root, file))
    
    return audio_files


def wrapper_extract_batch(args):
    # Unpack the arguments tuple and call extract_batch
    ckpt, audio_file, save_dir, device = args
    extract_batch(ckpt, audio_file, save_dir, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt')
    parser.add_argument('--audio_dir')
    parser.add_argument('--save_dir')
    parser.add_argument('--nproc', type=int)
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    print('Listing audio files')
    audio_files = list_audio_files(args.audio_dir)
    ckpt = args.ckpt

    gpus = torch.cuda.device_count()
    print('Using {} GPUs'.format(gpus))
    input_args = []
    batch_sz = 4096
    i = 0
    while i * batch_sz < len(audio_files):
        device = 'cpu' if gpus == 0 else 'cuda:{}'.format(i%gpus)
        input_args.append((ckpt, audio_files[i*batch_sz:(i+1)*batch_sz], args.save_dir, device))
        i += 1

    with Pool(args.nproc) as p:
        p.map(wrapper_extract_batch, input_args)