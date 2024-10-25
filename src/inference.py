# src/inference.py

import numpy as np

def generate_text(model, tokenizer, seed_text, seq_length, max_length):
    tokens = tokenizer.encode(seed_text)
    for _ in range(max_length):
        input_seq = np.array(tokens[-seq_length:]).reshape(1, -1)
        logits = model.forward(input_seq)
        probabilities = np.exp(logits[0, -1]) / np.sum(np.exp(logits[0, -1]))
        next_token = np.random.choice(len(probabilities), p=probabilities)
        tokens.append(next_token)
    return tokenizer.decode(tokens)