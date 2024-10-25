import numpy as np

class Embedding:
    def __init__(self, vocab_size, embed_size):
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        # Embedding matrisini küçük rastgele değerlerle başlatıyoruz
        self.weights = np.random.randn(vocab_size, embed_size) * 0.01
        self.grad_weights = np.zeros_like(self.weights)
        
    def forward(self, x):
        self.x = x
        return self.weights[x]
    
    def backward(self, grad_output):
        # Gradyanları güncelle
        np.add.at(self.grad_weights, self.x, grad_output)
    
    def zero_grad(self):
        self.grad_weights.fill(0)

class PositionalEncoding:
    def __init__(self, embed_size, max_len=5000):
        self.embed_size = embed_size
        self.max_len = max_len
        self.pe = self.create_positional_encoding()
        
    def create_positional_encoding(self):
        pe = np.zeros((self.max_len, self.embed_size))
        for pos in range(self.max_len):
            for i in range(0, self.embed_size, 2):
                pe[pos, i] = np.sin(pos / (10000 ** ((2 * i)/self.embed_size)))
                if i + 1 < self.embed_size:
                    pe[pos, i + 1] = np.cos(pos / (10000 ** ((2 * (i +1))/self.embed_size)))
        return pe
        
    def forward(self, x):
        seq_len = x.shape[1]
        return x + self.pe[:seq_len]
        
    def backward(self, grad_output):
        # Positional encoding öğrenilebilir olmadığı için direkt gradyanı döndürüyoruz
        return grad_output

class MultiHeadSelfAttention:
    def __init__(self, embed_size, heads):
        assert embed_size % heads == 0, "Embedding boyutu head sayısına tam bölünmeli"
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        # Ağırlıkları başlat
        self.W_q = np.random.randn(embed_size, embed_size) * (1./np.sqrt(embed_size))
        self.W_k = np.random.randn(embed_size, embed_size) * (1./np.sqrt(embed_size))
        self.W_v = np.random.randn(embed_size, embed_size) * (1./np.sqrt(embed_size))
        self.W_o = np.random.randn(embed_size, embed_size) * (1./np.sqrt(embed_size))
        
        # Gradyanlar
        self.grad_W_q = np.zeros_like(self.W_q)
        self.grad_W_k = np.zeros_like(self.W_k)
        self.grad_W_v = np.zeros_like(self.W_v)
        self.grad_W_o = np.zeros_like(self.W_o)
    
    def set_parameters(self, parameters):
        self.W_q = parameters[0]
        self.W_k = parameters[1]
        self.W_v = parameters[2]
        self.W_o = parameters[3]

    def forward(self, x):
        batch_size, seq_len, embed_size = x.shape
        
        # Linear projeksiyonlar
        self.x = x
        self.Q = x @ self.W_q
        self.K = x @ self.W_k
        self.V = x @ self.W_v
        
        # Head'lere böl
        self.Q = self.Q.reshape(batch_size, seq_len, self.heads, self.head_dim).transpose(0,2,1,3)
        self.K = self.K.reshape(batch_size, seq_len, self.heads, self.head_dim).transpose(0,2,1,3)
        self.V = self.V.reshape(batch_size, seq_len, self.heads, self.head_dim).transpose(0,2,1,3)
        
        # Scaled Dot-Product Attention
        scores = self.Q @ self.K.transpose(0,1,3,2) / np.sqrt(self.head_dim)  # (batch_size, heads, seq_len, seq_len)
        self.scores = scores
        self.weights = self.softmax(scores)
        self.attention = self.weights @ self.V  # (batch_size, heads, seq_len, head_dim)
        
        # Head'leri birleştir
        attention = self.attention.transpose(0,2,1,3).reshape(batch_size, seq_len, embed_size)
        
        # Son lineer layer
        out = attention @ self.W_o
        self.out = out
        return out
    
    def softmax(self, x):
        x = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def backward(self, grad_output):
        batch_size, seq_len, embed_size = grad_output.shape

        # W_o gradyanı ve attention gradyanı
        attention = self.attention.transpose(0,2,1,3).reshape(batch_size, seq_len, embed_size)
        d_attention = grad_output @ self.W_o.T
        self.grad_W_o += attention.reshape(-1, embed_size).T @ grad_output.reshape(-1, embed_size)

        # d_attention'u head'lere böl
        d_attention = d_attention.reshape(batch_size, seq_len, self.heads, self.head_dim).transpose(0,2,1,3)

        # weights ve V gradyanları
        d_weights = d_attention @ self.V.transpose(0,1,3,2)
        d_V = self.weights.transpose(0,1,3,2) @ d_attention

        # V gradyanı ve W_v gradyanı
        d_V = d_V.transpose(0,2,1,3).reshape(batch_size, seq_len, embed_size)
        self.grad_W_v += self.x.reshape(-1, embed_size).T @ d_V.reshape(-1, embed_size)
        d_x_v = d_V @ self.W_v.T

        # Softmax gradyanı
        d_scores = self.softmax_backward(d_weights)

        # Q ve K gradyanları
        d_Q = d_scores @ self.K  # Transpose etmeden kullanıyoruz
        d_K = d_scores.transpose(0,1,3,2) @ self.Q

        # Gradyanları uygun şekilde şekillendir
        d_Q = d_Q.transpose(0,2,1,3).reshape(batch_size, seq_len, embed_size)
        d_K = d_K.transpose(0,2,1,3).reshape(batch_size, seq_len, embed_size)

        # W_q ve W_k gradyanları
        self.grad_W_q += self.x.reshape(-1, embed_size).T @ d_Q.reshape(-1, embed_size)
        self.grad_W_k += self.x.reshape(-1, embed_size).T @ d_K.reshape(-1, embed_size)

        d_x_q = d_Q @ self.W_q.T
        d_x_k = d_K @ self.W_k.T

        # Girdi gradyanı
        grad_input = d_x_q + d_x_k + d_x_v

        return grad_input
    
    def softmax_backward(self, grad_output):
        # Softmax'ın backward hesaplaması
        # grad_output: (batch_size, heads, seq_len, seq_len)
        p = self.weights
        grad_input = grad_output * p - p * np.sum(grad_output * p, axis=-1, keepdims=True)
        return grad_input
        
    def zero_grad(self):
        self.grad_W_q.fill(0)
        self.grad_W_k.fill(0)
        self.grad_W_v.fill(0)
        self.grad_W_o.fill(0)

class FeedForward:
    def __init__(self, embed_size, forward_expansion=4):
        self.embed_size = embed_size
        self.forward_expansion = forward_expansion
        self.W1 = np.random.randn(embed_size, embed_size * forward_expansion) * (1./np.sqrt(embed_size))
        self.b1 = np.zeros((embed_size * forward_expansion,))
        self.W2 = np.random.randn(embed_size * forward_expansion, embed_size) * (1./np.sqrt(embed_size * forward_expansion))
        self.b2 = np.zeros((embed_size,))
        
        # Gradyanlar
        self.grad_W1 = np.zeros_like(self.W1)
        self.grad_b1 = np.zeros_like(self.b1)
        self.grad_W2 = np.zeros_like(self.W2)
        self.grad_b2 = np.zeros_like(self.b2)
    
    def set_parameters(self, parameters):
        self.W1 = parameters[0]
        self.b1 = parameters[1]
        self.W2 = parameters[2]
        self.b2 = parameters[3]

    def forward(self, x):
        self.x = x
        self.z1 = x @ self.W1 + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        return self.z2
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_backward(self, z, grad_output):
        grad = grad_output.copy()
        grad[z <= 0] = 0
        return grad
    
    def backward(self, grad_output):
        grad_a1 = grad_output @ self.W2.T
        self.grad_W2 += self.a1.reshape(-1, self.a1.shape[-1]).T @ grad_output.reshape(-1, grad_output.shape[-1])
        self.grad_b2 += grad_output.sum(axis=(0,1))
        
        grad_z1 = self.relu_backward(self.z1, grad_a1)
        self.grad_W1 += self.x.reshape(-1, self.x.shape[-1]).T @ grad_z1.reshape(-1, grad_z1.shape[-1])
        self.grad_b1 += grad_z1.sum(axis=(0,1))
        
        grad_input = grad_z1 @ self.W1.T
        return grad_input
    
    def zero_grad(self):
        self.grad_W1.fill(0)
        self.grad_b1.fill(0)
        self.grad_W2.fill(0)
        self.grad_b2.fill(0)

class LayerNormalization:
    def __init__(self, embed_size, eps=1e-6):
        self.embed_size = embed_size
        self.eps = eps
        self.gamma = np.ones((embed_size,))
        self.beta = np.zeros((embed_size,))
        
        # Gradyanlar
        self.grad_gamma = np.zeros_like(self.gamma)
        self.grad_beta = np.zeros_like(self.beta)
        
    def forward(self, x):
        self.x = x
        self.mean = x.mean(-1, keepdims=True)
        self.var = x.var(-1, keepdims=True)
        self.x_norm = (x - self.mean) / np.sqrt(self.var + self.eps)
        out = self.gamma * self.x_norm + self.beta
        return out
        
    def backward(self, grad_output):
        N, L, D = grad_output.shape
        
        x_mu = self.x - self.mean
        std_inv = 1. / np.sqrt(self.var + self.eps)
        
        grad_x_norm = grad_output * self.gamma
        grad_var = np.sum(grad_x_norm * x_mu, axis=-1, keepdims=True) * -0.5 * std_inv**3
        grad_mean = np.sum(grad_x_norm * -std_inv, axis=-1, keepdims=True) + grad_var * np.mean(-2. * x_mu, axis=-1, keepdims=True)
        
        grad_input = grad_x_norm * std_inv + grad_var * 2 * x_mu / D + grad_mean / D
        
        self.grad_gamma += np.sum(grad_output * self.x_norm, axis=(0,1))
        self.grad_beta += np.sum(grad_output, axis=(0,1))
        
        return grad_input
    
    def zero_grad(self):
        self.grad_gamma.fill(0)
        self.grad_beta.fill(0)

class TransformerBlock:
    def __init__(self, embed_size, heads, forward_expansion, dropout):
        self.attention = MultiHeadSelfAttention(embed_size, heads)
        self.norm1 = LayerNormalization(embed_size)
        self.feed_forward = FeedForward(embed_size, forward_expansion)
        self.norm2 = LayerNormalization(embed_size)
        self.dropout = dropout  # Bu basit versiyonda kullanılmıyor

    def set_parameters(self, parameters):
        idx = 0
        # Attention katmanı
        num_params_attn = len(self.attention.get_parameters())
        self.attention.set_parameters(parameters[idx:idx+num_params_attn])
        idx += num_params_attn

        # Layer Norm 1
        self.norm1.gamma = parameters[idx]
        idx +=1
        self.norm1.beta = parameters[idx]
        idx +=1

        # Feed Forward katmanı
        num_params_ffn = len(self.feed_forward.get_parameters())
        self.feed_forward.set_parameters(parameters[idx:idx+num_params_ffn])
        idx += num_params_ffn

        # Layer Norm 2
        self.norm2.gamma = parameters[idx]
        idx +=1
        self.norm2.beta = parameters[idx]
    
    def forward(self, x):
        att_out = self.attention.forward(x)
        x = self.norm1.forward(x + att_out)
        ff_out = self.feed_forward.forward(x)
        x = self.norm2.forward(x + ff_out)
        return x

    def backward(self, grad_output):
        grad_norm2 = self.norm2.backward(grad_output)
        grad_ff = self.feed_forward.backward(grad_norm2)
        grad_residual = grad_ff + grad_norm2
        grad_norm1 = self.norm1.backward(grad_residual)
        grad_att = self.attention.backward(grad_norm1)
        grad_input = grad_att + grad_norm1
        return grad_input

    def get_parameters(self):
        params = []
        # Multi-Head Attention parametreleri
        params.extend([self.attention.W_q, self.attention.W_k, self.attention.W_v, self.attention.W_o])
        # Layer Normalization 1 parametreleri
        params.extend([self.norm1.gamma, self.norm1.beta])
        # Feed Forward parametreleri
        params.extend([self.feed_forward.W1, self.feed_forward.b1, self.feed_forward.W2, self.feed_forward.b2])
        # Layer Normalization 2 parametreleri
        params.extend([self.norm2.gamma, self.norm2.beta])
        return params

    def get_gradients(self):
        grads = []
        # Multi-Head Attention gradyanları
        grads.extend([self.attention.grad_W_q, self.attention.grad_W_k, self.attention.grad_W_v, self.attention.grad_W_o])
        # Layer Normalization 1 gradyanları
        grads.extend([self.norm1.grad_gamma, self.norm1.grad_beta])
        # Feed Forward gradyanları
        grads.extend([self.feed_forward.grad_W1, self.feed_forward.grad_b1, self.feed_forward.grad_W2, self.feed_forward.grad_b2])
        # Layer Normalization 2 gradyanları
        grads.extend([self.norm2.grad_gamma, self.norm2.grad_beta])
        return grads

    def zero_grad(self):
        self.attention.zero_grad()
        self.norm1.zero_grad()
        self.feed_forward.zero_grad()
        self.norm2.zero_grad()

class Transformer:
    def __init__(self, vocab_size, embed_size, num_layers, heads, forward_expansion, max_length):
        self.embedding = Embedding(vocab_size, embed_size)
        self.positional_encoding = PositionalEncoding(embed_size, max_length)
        self.layers = [TransformerBlock(embed_size, heads, forward_expansion, dropout=0.1) for _ in range(num_layers)]
        self.fc_out = np.random.randn(embed_size, vocab_size) * (1./np.sqrt(embed_size))
        self.grad_fc_out = np.zeros_like(self.fc_out)
    
    def save_model(self, filename):
        parameters = self.get_parameters()
        np.save(filename, parameters)
    
    def load_model(self, filename):
        parameters = np.load(filename, allow_pickle=True)
        self.set_parameters(parameters)
    
    def set_parameters(self, parameters):
        idx = 0
        # Embedding katmanı
        self.embedding.weights = parameters[idx]
        idx += 1

        # Transformer blokları
        for layer in self.layers:
            num_params = len(layer.get_parameters())
            layer.set_parameters(parameters[idx:idx+num_params])
            idx += num_params

        # Son çıkış katmanı
        self.fc_out = parameters[idx]

    def forward(self, x):
        x = self.embedding.forward(x)
        x = self.positional_encoding.forward(x)
        for layer in self.layers:
            x = layer.forward(x)
        self.out = x
        logits = x @ self.fc_out
        self.logits = logits
        return logits
    
    def backward(self, grad_logits):
        batch_size, seq_len, vocab_size = grad_logits.shape
        grad_x = grad_logits @ self.fc_out.T
        self.grad_fc_out += self.out.reshape(-1, self.out.shape[-1]).T @ grad_logits.reshape(-1, grad_logits.shape[-1])
        for layer in reversed(self.layers):
            grad_x = layer.backward(grad_x)
        grad_x = self.positional_encoding.backward(grad_x)
        self.embedding.backward(grad_x)
    
    def get_parameters(self):
        parameters = []
        parameters.append(self.embedding.weights)
        parameters.extend([param for layer in self.layers for param in layer.get_parameters()])
        parameters.append(self.fc_out)
        return parameters
    
    def get_gradients(self):
        gradients = []
        gradients.append(self.embedding.grad_weights)
        gradients.extend([grad for layer in self.layers for grad in layer.get_gradients()])
        gradients.append(self.grad_fc_out)
        return gradients
    
    def zero_grad(self):
        self.embedding.zero_grad()
        for layer in self.layers:
            layer.zero_grad()
        self.grad_fc_out.fill(0)

class CrossEntropyLoss:
    def __init__(self):
        pass

    def softmax_function(self, x):
        # Numerically stable softmax
        x = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def forward(self, logits, targets):
        """
        logits: (batch_size, seq_len, vocab_size)
        targets: (batch_size, seq_len)
        """
        self.logits = logits
        self.targets = targets
        self.softmax = self.softmax_function(logits)
        
        batch_size, seq_len, vocab_size = logits.shape
        
        # Genişletilmiş İndeksler
        batch_indices = np.arange(batch_size)[:, None]
        pos_indices = np.arange(seq_len)[None, :]
        selected_softmax = self.softmax[batch_indices, pos_indices, targets]
        
        # Kayıp hesaplama
        loss = -np.log(selected_softmax + 1e-9)
        self.loss = np.sum(loss) / batch_size
        return self.loss

    def backward(self):
        batch_size, seq_len, vocab_size = self.logits.shape
        
        grad = self.softmax.copy()
        grad[np.arange(batch_size)[:, None], np.arange(seq_len)[None, :], self.targets] -= 1
        grad /= batch_size
        return grad

class Adam:
    def __init__(self, parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        self.parameters = parameters
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.m = [np.zeros_like(p) for p in parameters]
        self.v = [np.zeros_like(p) for p in parameters]
        self.t = 0
    
    def step(self, grads):
        self.t += 1
        for i, (param, grad) in enumerate(zip(self.parameters, grads)):
            self.m[i] = self.betas[0] * self.m[i] + (1 - self.betas[0]) * grad
            self.v[i] = self.betas[1] * self.v[i] + (1 - self.betas[1]) * (grad **2)
            
            m_hat = self.m[i] / (1 - self.betas[0] ** self.t)
            v_hat = self.v[i] / (1 - self.betas[1] ** self.t)
            
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)