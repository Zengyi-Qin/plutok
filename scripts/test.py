import numpy as np
import librosa
import soundfile
from plutok.tokenizer import Plutok


def calculate_entropy(int_list):
    # Count occurrences of each unique integer
    unique, counts = np.unique(int_list, return_counts=True)
    probabilities = counts / counts.sum()  # Normalize to get probabilities
    
    # Calculate Shannon entropy
    entropy = -np.sum(probabilities * np.log2(probabilities)) * len(int_list)
    return entropy


if __name__ == "__main__":
    ov_ckpt = "./ckpt"
    centroids = "./outputs/vqvae/model.pth"
    tokenizer_file = "./outputs/tokenizer/tokenizer.json"
    device = "cuda"
    audio_path = "resource/okunohosomichi_01-09_basho_16000.mp3" 
    output_path = "./output.mp3"
    plutok_tokenizer = Plutok(ov_ckpt, centroids, tokenizer_file, device)
    wav, _ = librosa.load(audio_path, sr=22050)

    enc = plutok_tokenizer.encode(wav)
    hz = len(enc.ids) / (len(wav.squeeze()) / 22050)
    entropy = calculate_entropy(enc.ids)
    entropy_per_sec = entropy / (len(wav.squeeze()) / 22050)
    print("token hz: {:.2f}, information entropy: {:.2f} / sec".format(hz, entropy_per_sec))
    wav_dec = plutok_tokenizer.decode(enc).squeeze()
    soundfile.write(output_path, wav_dec, 22050)
