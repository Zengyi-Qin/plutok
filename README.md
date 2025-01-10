# Plutok: 18Hz Tokenizer for Audio LLM

## Training
```bash
python compute_embedding.py --ckpt ckpt/ --audio_dir data --save_dir outputs --nproc 32

python compute_kmeans.py --emb_dir outputs/emb/ --save_dir outputs/kmeans

python compute_bpe.py --text_dir outputs/kmeans/centroid_ids/ --save_dir outputs/tokenizer --vocab_size 24000
```