# Plutok: Pushing the Token Rate Limit of Multi-lingual Audio Tokenizers

<div style="display: flex; justify-content: center; align-items: center; height: 100vh;">
    <img width="754" alt="image" src="https://github.com/user-attachments/assets/46f780aa-8d82-4f23-aae7-fdc40f3750dc" />
</div>

## Introduction
Plutok is an audio tokenizer that compresses multilingual speech into discrete codes at **27 toks/s**, enabling efficient audio-language autoregressive modeling. With a **frame rate 32% lower than previous SOTA** audio tokenizers, it allows for **48% longer audio sequences within the same context window**. This makes Plutok particularly effective for processing extended audio content while maintaining high-quality compression.

**Technical report:** [link](https://twisty-oval-d44.notion.site/Plutok-Pushing-the-Token-Rate-Limit-of-Multi-lingual-Audio-Tokenizers-179c58cbbfe280ec96c1d750609b3da2)

## Pretrained Checkpoints
You can download pretrained checkpoints [here](https://github.com/Zengyi-Qin/plutok/releases/download/v0.1/pretrained.zip).

## Training
Compute semantic embedding of speech audio:
```bash
python scripts/compute_embedding.py --ckpt ckpt/ --audio_dir data --save_dir outputs --nproc 32
```
The `ckpt` is the checkpoint dir of OpenVoice. The `data` contains the training audios. We used around 120000 pieces of multi-lingual speech audios with average length of 10 secs. And then train vector quantized VAE with:
```bash
python scripts/train_vqvae.py --emb_dir outputs/emb/ --save_dir outputs/vqvae
```
Then we train the BPE tokenizer:
```
python scripts/train_bpe.py --text_dir outputs/vqvae/centroid_ids/ --save_dir outputs/tokenizer --vocab_size 12000
```

## Inference
```python
ov_ckpt = "./ckpt"
vqvae_ckpt = "./outputs/vqvae/model.pth"
tokenizer_file = "./outputs/tokenizer/tokenizer.json"
device = "cuda"
audio_path = "input.mp3" 
output_path = "output.mp3"
plutok_tokenizer = Plutok(ov_ckpt, centroids, tokenizer_file, device)
wav, _ = librosa.load(audio_path, sr=22050)

enc = plutok_tokenizer.encode(wav)
hz = len(enc.ids) / (len(wav.squeeze()) / 22050)

print("token hz: {:.2f}".format(hz))
wav_dec = plutok_tokenizer.decode(enc).squeeze()
soundfile.write(output_path, wav_dec, 22050)
```
