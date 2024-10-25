# generate_text.py

import numpy as np
import pickle
from src.tokenizer import Tokenizer
from src.model import Transformer
from src.inference import generate_text

def main():
    # Tokenizer'ı yükleme
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    
    # Model parametreleri
    vocab_size = tokenizer.vocab_size
    embed_size = 128
    num_layers = 2
    heads = 4
    forward_expansion = 4
    max_length = 500  # Positional encoding için max_length artırıldı
    seq_length = 50
    
    # Modeli oluşturma ve parametreleri yükleme
    model = Transformer(
        vocab_size=vocab_size,
        embed_size=embed_size,
        num_layers=num_layers,
        heads=heads,
        forward_expansion=forward_expansion,
        max_length=max_length
    )
    model.load_model('model_weights.npy')
    
    # Metin oluşturma
    seed = "Merhaba, nasılsın?"
    generated = generate_text(model, tokenizer, seed, seq_length=seq_length, max_length=100)
    print(f"Seed Text: '{seed}'\n")
    print(f"Generated Text: '{generated}'")
    
if __name__ == "__main__":
    main()