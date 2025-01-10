import numpy as np
import torch
from plutok.openvoice import se_extractor
from plutok.openvoice.api import ToneColorConverter
from transformers import PreTrainedTokenizerFast
import json


class Encode(object):
    def __init__(self, ids, se):
        self.ids = ids
        self.se = se


class Plutok(object):

    def __init__(self, ov_ckpt, centroids, tokenizer_file, device):
        ckpt_converter = f'{ov_ckpt}/converter'
        ov = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
        ov.load_ckpt(f'{ckpt_converter}/checkpoint.pth')
        self.device = device
        self.ov = ov
        self.centroids = np.array(json.load(open(centroids))['centroids'])
        self.bpe = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file)

    def encode(self, wav_array):
        se = se_extractor.get_se(wav_array, self.ov)
        emb = self.ov.convert(audio_src=wav_array, src_se=se)[0].T # (t, c)
        emb = emb[:, np.newaxis] # (t, 1, c)
        ctr = self.centroids[np.newaxis] # (1, 256, c)
        diff = np.linalg.norm(ctr - emb, axis=2) # (t, 256)
        ids = np.argmin(diff, axis=1)
        ids = ''.join(['#'+str(i) for i in ids])
        bpe_ids = self.bpe(ids)['input_ids']
        enc = Encode(bpe_ids, se)
        return enc

    def decode(self, enc):
        text = self.bpe.decode(enc.ids)
        text = text.replace(' ', '')
        ids = [int(i) for i in text.split('#') if i.isdigit()]
        feat = self.centroids[ids].T 
        feat = feat[np.newaxis]
        feat = feat + np.random.normal(size=feat.shape) * 0.7
        feat = torch.tensor(feat).float().to(self.device)
        wav = self.ov.model.decode(feat, enc.se).detach().cpu().numpy()
        return wav

        


    