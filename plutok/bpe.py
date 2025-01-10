import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer


def dataset(text_dir):
    for f in os.listdir(text_dir):
        string = open(os.path.join(text_dir, f), 'r').read()
        yield string

def train_bpe(text_dir, vocab_size, save_dir):
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"], vocab_size=vocab_size)
    tokenizer.train_from_iterator(dataset(text_dir), trainer=trainer)
    
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'tokenizer.json')
    tokenizer.save(save_path)
