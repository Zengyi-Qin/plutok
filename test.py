import librosa
import soundfile
from plutok.tokenizer import Plutok


if __name__ == "__main__":
    ov_ckpt = "./ckpt"
    #centroids = "./outputs/kmeans/centroids.json"
    centroids = "./outputs/vqvae/model.pth"
    tokenizer_file = "./outputs/tokenizer/tokenizer.json"
    device = "cuda"
    audio_path = "resource/example_reference_22.05khz.mp3"  # "tellitall_00_stenhouse_64kb_cut.mp3"
    output_path = "./output.mp3"
    plutok_tokenizer = Plutok(ov_ckpt, centroids, tokenizer_file, device)
    wav, _ = librosa.load(audio_path, sr=22050)
    wav_dec = plutok_tokenizer.reconstruct(wav).squeeze()
    soundfile.write(output_path, wav_dec, 22050)
    # enc = plutok_tokenizer.encode(wav)
    # hz = len(enc.ids) / (len(wav.squeeze()) / 22050)
    # print("token hz: {:.2f}".format(hz))
    # wav_dec = plutok_tokenizer.decode(enc).squeeze()
    # soundfile.write(output_path, wav_dec, 22050)
