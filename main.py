import os
import pickle
import numpy as np
from src.tokenizer import Tokenizer
from src.model import Transformer, CrossEntropyLoss, Adam
from src.utils import load_and_preprocess
from tqdm import tqdm
import gc

class BatchGenerator:
    def __init__(self, tokens, batch_size, seq_length):
        self.tokens = tokens
        self.batch_size = batch_size
        self.seq_length = seq_length
        
    def __iter__(self):
        tokens = np.array(self.tokens)
        num_batches = len(tokens) // (self.batch_size * self.seq_length)
        tokens = tokens[:num_batches * self.batch_size * self.seq_length]
        tokens = tokens.reshape((self.batch_size, -1))
        
        for i in range(0, tokens.shape[1] - self.seq_length, self.seq_length):
            x = tokens[:, i:i+self.seq_length]
            y = tokens[:, i+1:i+self.seq_length+1]
            yield x, y

def main():
    # Veri dizinini belirtin
    data_directory = 'data'
    
    # Tüm .txt dosyalarının yollarını al
    file_paths = [os.path.join(data_directory, file) 
                 for file in os.listdir(data_directory) 
                 if file.endswith('.txt')]
    
    print("Dosyalar yükleniyor ve ön işleniyor...")
    clean_text = load_and_preprocess(file_paths)
    
    print(f"İşlenen veri boyutu: {len(clean_text)} karakter")
    print(f"Bellek kullanımı: {get_memory_usage():.2f} MB")
    
    tokenizer = Tokenizer(vocab_size=30000)
    print("Vocabulary oluşturuluyor...")
    tokenizer.build_vocab(clean_text)
    
    print("Metin tokenize ediliyor...")
    tokens = tokenizer.encode(clean_text)
    
    # Bellek optimizasyonu için clean_text'i sil
    del clean_text
    gc.collect()
    
    print(f"Tokenization sonrası bellek: {get_memory_usage():.2f} MB")
    
    # Tokenizer'ı kaydet
    with open('tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    
    # <UNK> oranını kontrol et
    unk_count = tokens.count(tokenizer.word_to_id['<UNK>'])
    total_tokens = len(tokens)
    print(f"<UNK> oranı: {unk_count / total_tokens * 100:.2f}%")
    
    # Model parametreleri
    model = Transformer(
    vocab_size=30000,
    embed_size=2048,
    num_layers=80,
    heads=32,
    forward_expansion=4,
    max_length=500
)

    
    # Eğitim parametreleri
    batch_size = 64
    seq_length = 40
    epochs = 10
    
    # Loss ve optimizer
    loss_fn = CrossEntropyLoss()
    parameters = model.get_parameters()
    optimizer = Adam(parameters=parameters, lr=0.002)
    
    # Eğitim döngüsü
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs} başlıyor...")
        epoch_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        batch_generator = BatchGenerator(tokens, batch_size, seq_length)
        
        for x, y in tqdm(batch_generator, desc=f"Epoch {epoch+1}"):
            # Forward pass
            logits = model.forward(x)
            loss = loss_fn.forward(logits, y)
            epoch_loss += loss
            
            # Doğruluk hesaplama
            predictions = np.argmax(logits, axis=-1)
            correct_predictions += np.sum(predictions == y)
            total_predictions += y.size
            
            # Backward pass
            grad_logits = loss_fn.backward()
            model.backward(grad_logits)
            
            # Parametreleri güncelle
            grads = model.get_gradients()
            optimizer.step(grads)
            
            # Gradyanları sıfırla
            model.zero_grad()
            
            # Düzenli bellek temizliği
            if total_predictions % (batch_size * seq_length * 100) == 0:
                gc.collect()
        
        # Epoch istatistiklerini yazdır
        avg_loss = epoch_loss / (len(tokens) // (batch_size * seq_length))
        accuracy = correct_predictions / total_predictions * 100
        print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        print(f"Bellek kullanımı: {get_memory_usage():.2f} MB")
        
        # Her epoch sonunda modeli kaydet
        model.save_model(f'model_weights_epoch_{epoch+1}.npy')
    
    print("Model eğitimi tamamlandı ve kaydedildi.")

def get_memory_usage():
    """Mevcut işlemin bellek kullanımını MB cinsinden döndürür"""
    import psutil
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

if __name__ == "__main__":
    main()

