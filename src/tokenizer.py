# src/tokenizer.py

class Tokenizer:
    def __init__(self, vocab_size=30000):
        self.vocab_size = vocab_size
        self.word_to_id = {}
        self.id_to_word = {}
    
    def build_vocab(self, text):
        from collections import Counter
        words = text.split()
        word_counts = Counter(words)
        most_common = word_counts.most_common(self.vocab_size - 2)  # PAD ve UNK için yer ayırıyoruz
        self.word_to_id = {word: idx+2 for idx, (word, _) in enumerate(most_common)}
        self.word_to_id['<PAD>'] = 0
        self.word_to_id['<UNK>'] = 1
        self.id_to_word = {idx: word for word, idx in self.word_to_id.items()}
    
    def encode(self, text):
        return [self.word_to_id.get(word, self.word_to_id['<UNK>']) for word in text.split()]
    
    def decode(self, tokens):
        return ' '.join([self.id_to_word.get(token, '<UNK>') for token in tokens])