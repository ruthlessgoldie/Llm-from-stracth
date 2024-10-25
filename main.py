import os
import pickle
import numpy as np
from src.tokenizer import Tokenizer
from src.model import Transformer, CrossEntropyLoss, Adam
from src.utils import load_and_preprocess
from tqdm import tqdm

def get_batches(tokens, batch_size, seq_length):
    tokens = np.array(tokens)
    num_batches = len(tokens) // (batch_size * seq_length)
    tokens = tokens[:num_batches * batch_size * seq_length]
    tokens = tokens.reshape((batch_size, -1))
    for i in range(0, tokens.shape[1] - seq_length, seq_length):
        x = tokens[:, i:i+seq_length]
        y = tokens[:, i+1:i+seq_length+1]
        yield x, y

def main():
    # Veri dizinini belirtin
    data_directory = 'data'
    
    # Tüm .txt dosyalarının yollarını al
    file_paths = [os.path.join(data_directory, file) for file in os.listdir(data_directory) if file.endswith('.txt')]
    
    clean_text = load_and_preprocess(file_paths)
    
    tokenizer = Tokenizer(vocab_size=30000)  # Kelime dağarcığını artırdık
    tokenizer.build_vocab(clean_text)
    tokens = tokenizer.encode(clean_text)
    
    # Tokenizer'ı kaydetme
    with open('tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    
    # Tokenizer'daki <UNK> oranını kontrol et
    unk_count = tokens.count(tokenizer.word_to_id['<UNK>'])
    total_tokens = len(tokens)
    print(f"<UNK> oranı: {unk_count / total_tokens * 100:.2f}%")
    
    # Model parametreleri
    vocab_size = 30000
    embed_size = 128
    num_layers = 2
    heads = 4
    forward_expansion = 4
    max_length = 500  # Positional encoding için max_length artırıldı
    
    # Transformer modelini başlatma
    model = Transformer(
        vocab_size=vocab_size,
        embed_size=embed_size,
        num_layers=num_layers,
        heads=heads,
        forward_expansion=forward_expansion,
        max_length=max_length
    )
    
    # Eğitim parametreleri
    batch_size = 32
    seq_length = 50
    epochs = 10
    learning_rate = 0.001
    
    # Kayıp fonksiyonu ve optimizer
    loss_fn = CrossEntropyLoss()
    
    # Modeldeki tüm öğrenilebilir parametreleri alıyoruz
    parameters = model.get_parameters()
    optimizer = Adam(parameters=parameters, lr=learning_rate)
    
    # Eğitim döngüsü
    for epoch in range(epochs):
        batch_generator = get_batches(tokens, batch_size, seq_length)
        epoch_loss = 0
        correct_predictions = 0
        total_predictions = 0
        for x, y in tqdm(batch_generator, desc=f"Epoch {epoch+1}/{epochs}"):
            logits = model.forward(x)  # (batch_size, seq_len, vocab_size)
            loss = loss_fn.forward(logits, y)  # Scalar
            epoch_loss += loss
                
            # Doğruluk hesaplama
            predictions = np.argmax(logits, axis=-1)
            correct_predictions += np.sum(predictions == y)
            total_predictions += y.size
                
            # Geri yayılım
            grad_logits = loss_fn.backward()  # (batch_size, seq_len, vocab_size)
            model.backward(grad_logits)
                
            # Parametreleri güncelle
            grads = model.get_gradients()
            optimizer.step(grads)
            
            # Modelin gradyanlarını sıfırla
            model.zero_grad()
                
        avg_loss = epoch_loss / (len(tokens) // (batch_size * seq_length))
        accuracy = correct_predictions / total_predictions * 100
        print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    
    # Modeli kaydetme
    model.save_model('model_weights.npy')
    
    print("Model eğitimi tamamlandı ve kaydedildi.")
    
if __name__ == "__main__":
    main()
