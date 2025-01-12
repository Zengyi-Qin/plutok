import numpy as np
import torch
from plutok.openvoice import se_extractor
from plutok.openvoice.api import ToneColorConverter
import os


def extract(ov, audio_path, save_dir):
    se = se_extractor.get_se(audio_path, ov)
    emb = ov.convert(audio_src=audio_path, src_se=se)
    n, c, t = emb.shape
    assert n == 1
    if t < 100:
        return
    save_path = (
        os.path.splitext(audio_path)[0].replace("/", "_").replace(".", "") + ".npy"
    )
    save_path = os.path.join(save_dir, save_path)
    np.save(open(save_path, "wb"), emb)


def extract_batch(ckpt, audio_files, save_dir, device):
    ckpt_converter = f"{ckpt}/converter"
    ov = ToneColorConverter(f"{ckpt_converter}/config.json", device=device)
    ov.load_ckpt(f"{ckpt_converter}/checkpoint.pth")
    for audio_file in audio_files:
        extract(ov, audio_file, save_dir)
