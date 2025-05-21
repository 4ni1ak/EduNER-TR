import numpy as np
import time
import os
import matplotlib.pyplot as plt

class Embedding:
    def __init__(self, vocab_size, embedding_dim):
        """
        Kelime ID'lerini embedding vektörlerine dönüştüren katman
        
        Args:
            vocab_size: Kelime dağarcığı büyüklüğü
            embedding_dim: Embedding vektörlerinin boyutu
        """
        # Xavier/Glorot başlatma ile embedding matrisini oluştur
        self.weights = np.random.randn(vocab_size, embedding_dim) * 0.1
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
    def forward(self, word_ids):
        """
        Kelime ID'lerinden embedding vektörlerini döndür
        
        Args:
            word_ids: Kelime ID'leri, shape: (batch_size, seq_length)
            
        Returns:
            Embedding vektörleri, shape: (batch_size, seq_length, embedding_dim)
        """
        # ID'lere göre embedding matrisinden vektörleri al
        self.input_shape = word_ids.shape
        self.indices = word_ids
        return self.weights[word_ids]
    
    def backward(self, grad_output):
        """
        Gradyanları geri yayılım için hesapla
        
        Args:
            grad_output: Sonraki katmandan gelen gradyan, shape: (batch_size, seq_length, embedding_dim)
            
        Returns:
            Giriş gradyanı (bu katman için None, çünkü indekslerin gradyanı yoktur)
        """
        # Gradyanları topla ve embedding matrisini güncelle için hazırla
        self.grad_weights = np.zeros_like(self.weights)
        
        # Her bir indeks için gradyanları topla
        np.add.at(self.grad_weights, self.indices.flatten(), 
                 grad_output.reshape(-1, self.embedding_dim))
        
        # Giriş gradyanı hesaplanamaz (kelime ID'leri)
        return None
    
    def update_params(self, learning_rate):
        """
        SGD ile ağırlıkları güncelle
        
        Args:
            learning_rate: Öğrenme oranı
        """
        self.weights -= learning_rate * self.grad_weights


class LSTMCell:
    def __init__(self, input_size, hidden_size):
        """
        Tek bir LSTM hücresi
        
        Args:
            input_size: Giriş boyutu (embedding_dim)
            hidden_size: Gizli durum boyutu
        """
        # Ağırlıkları Xavier/Glorot başlatma ile oluştur
        k = np.sqrt(1 / hidden_size)
        
        # Input gate parametreleri
        self.Wi = np.random.uniform(-k, k, (input_size, hidden_size))
        self.Ui = np.random.uniform(-k, k, (hidden_size, hidden_size))
        self.bi = np.zeros(hidden_size)
        
        # Forget gate parametreleri
        self.Wf = np.random.uniform(-k, k, (input_size, hidden_size))
        self.Uf = np.random.uniform(-k, k, (hidden_size, hidden_size))
        self.bf = np.zeros(hidden_size)
        
        # Output gate parametreleri
        self.Wo = np.random.uniform(-k, k, (input_size, hidden_size))
        self.Uo = np.random.uniform(-k, k, (hidden_size, hidden_size))
        self.bo = np.zeros(hidden_size)
        
        # Cell state parametreleri
        self.Wc = np.random.uniform(-k, k, (input_size, hidden_size))
        self.Uc = np.random.uniform(-k, k, (hidden_size, hidden_size))
        self.bc = np.zeros(hidden_size)
        
        self.hidden_size = hidden_size
        self.input_size = input_size
        
        # Gradyanları saklamak için
        self.param_keys = ['Wi', 'Ui', 'bi', 'Wf', 'Uf', 'bf', 
                          'Wo', 'Uo', 'bo', 'Wc', 'Uc', 'bc']
        self.grads = {k: np.zeros_like(getattr(self, k)) for k in self.param_keys}
        
    def forward(self, x, h_prev, c_prev):
        """
        LSTM hücresinin ileri geçişi
        
        Args:
            x: Giriş, shape: (batch_size, input_size)
            h_prev: Önceki gizli durum, shape: (batch_size, hidden_size)
            c_prev: Önceki hücre durumu, shape: (batch_size, hidden_size)
            
        Returns:
            h_next: Sonraki gizli durum, shape: (batch_size, hidden_size)
            c_next: Sonraki hücre durumu, shape: (batch_size, hidden_size)
        """
        # LSTM kapılarını hesapla
        self.x = x
        self.h_prev = h_prev
        self.c_prev = c_prev
        
        # Input gate
        self.i_t = sigmoid(np.dot(x, self.Wi) + np.dot(h_prev, self.Ui) + self.bi)
        
        # Forget gate
        self.f_t = sigmoid(np.dot(x, self.Wf) + np.dot(h_prev, self.Uf) + self.bf)
        
        # Output gate
        self.o_t = sigmoid(np.dot(x, self.Wo) + np.dot(h_prev, self.Uo) + self.bo)
        
        # Cell state güncellemesi için aday
        self.c_tilde = np.tanh(np.dot(x, self.Wc) + np.dot(h_prev, self.Uc) + self.bc)
        
        # Cell state güncelleme
        self.c_next = self.f_t * c_prev + self.i_t * self.c_tilde
        
        # Hidden state güncelleme
        self.h_next = self.o_t * np.tanh(self.c_next)
        
        return self.h_next, self.c_next
    
    def backward(self, dh_next, dc_next):
        """
        LSTM hücresinin geri yayılımı
        
        Args:
            dh_next: Sonraki gizli durum için gradyan, shape: (batch_size, hidden_size)
            dc_next: Sonraki hücre durumu için gradyan, shape: (batch_size, hidden_size)
            
        Returns:
            dx: Giriş için gradyan
            dh_prev: Önceki gizli durum için gradyan
            dc_prev: Önceki hücre durumu için gradyan
        """
        # Geri yayılım için gradyanlar
        batch_size = dh_next.shape[0]

        # Hidden state'den gelen gradyan
        do_t = dh_next * np.tanh(self.c_next)
        dc_t = dc_next + dh_next * self.o_t * (1 - np.tanh(self.c_next)**2)
        
        # Forget gate gradyanı
        df_t = dc_t * self.c_prev
        # Input gate gradyanı
        di_t = dc_t * self.c_tilde
        # Cell state aday gradyanı
        dc_tilde = dc_t * self.i_t
        
        # Sigmoid gradyanı: sigmoid(x) * (1 - sigmoid(x))
        # Tanh gradyanı: 1 - tanh(x)^2
        
        # Kapı gradyanları
        di_t_input = di_t * self.i_t * (1 - self.i_t)
        df_t_input = df_t * self.f_t * (1 - self.f_t)
        do_t_input = do_t * self.o_t * (1 - self.o_t)
        dc_tilde_input = dc_tilde * (1 - self.c_tilde**2)
        
        # Ağırlık gradyanları
        # Input gate
        self.grads['Wi'] += np.dot(self.x.T, di_t_input)
        self.grads['Ui'] += np.dot(self.h_prev.T, di_t_input)
        self.grads['bi'] += np.sum(di_t_input, axis=0)
        
        # Forget gate
        self.grads['Wf'] += np.dot(self.x.T, df_t_input)
        self.grads['Uf'] += np.dot(self.h_prev.T, df_t_input)
        self.grads['bf'] += np.sum(df_t_input, axis=0)
        
        # Output gate
        self.grads['Wo'] += np.dot(self.x.T, do_t_input)
        self.grads['Uo'] += np.dot(self.h_prev.T, do_t_input)
        self.grads['bo'] += np.sum(do_t_input, axis=0)
        
        # Cell state
        self.grads['Wc'] += np.dot(self.x.T, dc_tilde_input)
        self.grads['Uc'] += np.dot(self.h_prev.T, dc_tilde_input)
        self.grads['bc'] += np.sum(dc_tilde_input, axis=0)
        
        # Giriş gradyanları
        dx = (np.dot(di_t_input, self.Wi.T) + 
              np.dot(df_t_input, self.Wf.T) + 
              np.dot(do_t_input, self.Wo.T) + 
              np.dot(dc_tilde_input, self.Wc.T))
        
        # Önceki durumlar için gradyanlar
        dh_prev = (np.dot(di_t_input, self.Ui.T) + 
                   np.dot(df_t_input, self.Uf.T) + 
                   np.dot(do_t_input, self.Uo.T) + 
                   np.dot(dc_tilde_input, self.Uc.T))
        
        # Önceki hücre durumu için gradyan
        dc_prev = dc_t * self.f_t
        
        return dx, dh_prev, dc_prev
    
    def update_params(self, learning_rate):
        """
        SGD ile ağırlıkları güncelle
        
        Args:
            learning_rate: Öğrenme oranı
        """
        for key in self.param_keys:
            param = getattr(self, key)
            grad = self.grads[key]
            param -= learning_rate * grad
            # Gradyanları sıfırla
            self.grads[key] = np.zeros_like(param)


class LSTMModel:
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size):
        """
        LSTM modeli
        
        Args:
            vocab_size: Kelime dağarcığı büyüklüğü
            embedding_dim: Embedding vektörlerinin boyutu
            hidden_size: LSTM gizli durum boyutu
            output_size: Çıkış sınıflarının sayısı
        """
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.lstm_cell = LSTMCell(embedding_dim, hidden_size)
        
        # Çıkış katmanı
        self.Wy = np.random.randn(hidden_size, output_size) * 0.01
        self.by = np.zeros(output_size)
        
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.vocab_size = vocab_size
        
    def forward(self, inputs, labels=None):
        """
        İleri geçiş
        
        Args:
            inputs: Kelime ID'leri, shape: (batch_size, seq_length)
            labels: Etiketler, shape: (batch_size, seq_length)
            
        Returns:
            predictions: Tahminler, shape: (batch_size, seq_length, output_size)
            loss: Kayıp değeri (eğer labels verilirse)
        """
        batch_size, seq_length = inputs.shape
        
        # Embeddingler
        embeds = self.embedding.forward(inputs)  # (batch_size, seq_length, embedding_dim)
        
        # LSTM çıkışları
        h = np.zeros((batch_size, self.hidden_size))  # Initial hidden state
        c = np.zeros((batch_size, self.hidden_size))  # Initial cell state
        
        self.h_states = [h]  # (t=0 için başlangıç durumu)
        self.c_states = [c]  # (t=0 için başlangıç durumu)
        self.x_inputs = []  # Her zaman adımındaki girişleri sakla
        
        # Her zaman adımı için LSTM hücresini çalıştır
        for t in range(seq_length):
            x_t = embeds[:, t, :]  # t anındaki batch girişi
            self.x_inputs.append(x_t)
            
            h, c = self.lstm_cell.forward(x_t, self.h_states[-1], self.c_states[-1])
            self.h_states.append(h)
            self.c_states.append(c)
        
        # Her zaman adımı için çıkış
        outputs = []
        for h_t in self.h_states[1:]:  # İlk (t=0) durumu kullanma
            y_t = np.dot(h_t, self.Wy) + self.by
            outputs.append(y_t)
        
        # Çıkışları birleştir: (batch_size, seq_length, output_size)
        self.outputs = np.stack(outputs, axis=1)
        
        # Softmax çıkışları
        self.probs = softmax(self.outputs)
        
        # Eğer etiketler verilmişse loss hesapla
        loss = None
        if labels is not None:
            # Cross-entropy loss
            batch_size, seq_length = labels.shape
            loss = 0
            for b in range(batch_size):
                for t in range(seq_length):
                    loss -= np.log(self.probs[b, t, labels[b, t]] + 1e-10)
            loss /= (batch_size * seq_length)
            
            # Doğruluk hesapla
            self.predictions = np.argmax(self.probs, axis=2)
            self.accuracy = np.mean(self.predictions == labels)
            
        return self.probs, loss
    
    def backward(self, labels):
        """
        Geri yayılım
        
        Args:
            labels: Gerçek etiketler, shape: (batch_size, seq_length)
        """
        batch_size, seq_length = labels.shape
        
        # Çıkış gradyanı - softmax cross entropy
        dL_dout = self.probs.copy()  # (batch_size, seq_length, output_size)
        
        # Her batch ve seq için gerçek sınıf olasılığından 1 çıkar
        for b in range(batch_size):
            for t in range(seq_length):
                dL_dout[b, t, labels[b, t]] -= 1
        
        # Batch ve sequence ortalama kayba göre normalize et
        dL_dout /= (batch_size * seq_length)
        
        # Çıkış katmanı gradyanları
        self.dWy = np.zeros_like(self.Wy)
        self.dby = np.zeros_like(self.by)
        
        # LSTM geri yayılım - tüm zaman adımları için
        dh_next = np.zeros((batch_size, self.hidden_size))
        dc_next = np.zeros((batch_size, self.hidden_size))
        
        # Zaman adımlarını tersine çevir
        for t in reversed(range(seq_length)):
            # Çıkış katmanı gradyanı
            dy = dL_dout[:, t, :]  # (batch_size, output_size)
            
            # Hidden state gradyanı
            dh = np.dot(dy, self.Wy.T) + dh_next  # Sonraki zamandan gelen gradyan
            
            # Çıkış katmanı ağırlık gradyanları
            self.dWy += np.dot(self.h_states[t+1].T, dy)
            self.dby += np.sum(dy, axis=0)
            
            # LSTM hücresi geri yayılım
            dx, dh_next, dc_next = self.lstm_cell.backward(dh, dc_next)
            
            # Embedding geri yayılım için gradyanları sakla (her zaman adımı için)
            if t == seq_length - 1:
                dembed = np.zeros((batch_size, seq_length, self.embedding.embedding_dim))
            
            dembed[:, t, :] = dx
        
        # Embedding geri yayılım
        self.embedding.backward(dembed)
        
    def update_params(self, learning_rate):
        """
        SGD ile ağırlıkları güncelle
        
        Args:
            learning_rate: Öğrenme oranı
        """
        # Embedding güncelle
        self.embedding.update_params(learning_rate)
        
        # LSTM hücresi güncelle
        self.lstm_cell.update_params(learning_rate)
        
        # Çıkış katmanı güncelle
        self.Wy -= learning_rate * self.dWy
        self.by -= learning_rate * self.dby
        
    def save_checkpoint(self, filepath):
        """
        Model ağırlıklarını kaydet
        
        Args:
            filepath: Kaydedilecek dosya yolu
        """
        checkpoint = {
            'embedding_weights': self.embedding.weights,
            'lstm_Wi': self.lstm_cell.Wi,
            'lstm_Ui': self.lstm_cell.Ui,
            'lstm_bi': self.lstm_cell.bi,
            'lstm_Wf': self.lstm_cell.Wf,
            'lstm_Uf': self.lstm_cell.Uf,
            'lstm_bf': self.lstm_cell.bf,
            'lstm_Wo': self.lstm_cell.Wo,
            'lstm_Uo': self.lstm_cell.Uo,
            'lstm_bo': self.lstm_cell.bo,
            'lstm_Wc': self.lstm_cell.Wc,
            'lstm_Uc': self.lstm_cell.Uc,
            'lstm_bc': self.lstm_cell.bc,
            'Wy': self.Wy,
            'by': self.by
        }
        np.save(filepath, checkpoint)
        print(f"Model kaydedildi: {filepath}")
        
    def load_checkpoint(self, filepath):
        """
        Model ağırlıklarını yükle
        
        Args:
            filepath: Yüklenecek dosya yolu
        """
        checkpoint = np.load(filepath, allow_pickle=True).item()
        
        self.embedding.weights = checkpoint['embedding_weights']
        self.lstm_cell.Wi = checkpoint['lstm_Wi']
        self.lstm_cell.Ui = checkpoint['lstm_Ui']
        self.lstm_cell.bi = checkpoint['lstm_bi']
        self.lstm_cell.Wf = checkpoint['lstm_Wf']
        self.lstm_cell.Uf = checkpoint['lstm_Uf']
        self.lstm_cell.bf = checkpoint['lstm_bf']
        self.lstm_cell.Wo = checkpoint['lstm_Wo']
        self.lstm_cell.Uo = checkpoint['lstm_Uo']
        self.lstm_cell.bo = checkpoint['lstm_bo']
        self.lstm_cell.Wc = checkpoint['lstm_Wc']
        self.lstm_cell.Uc = checkpoint['lstm_Uc']
        self.lstm_cell.bc = checkpoint['lstm_bc']
        self.Wy = checkpoint['Wy']
        self.by = checkpoint['by']
        
        print(f"Model yüklendi: {filepath}")


# Yardımcı aktivasyon fonksiyonları
def sigmoid(x):
    """Sigmoid aktivasyon fonksiyonu"""
    return 1 / (1 + np.exp(-np.clip(x, -15, 15)))  # Taşmayı önlemek için clipping

def softmax(x):
    """Softmax aktivasyon fonksiyonu"""
    # Numerik stabilite için maksimum değeri çıkar
    e_x = np.exp(x - np.max(x, axis=2, keepdims=True))
    return e_x / np.sum(e_x, axis=2, keepdims=True)


def load_data(data_dir):
    """
    Veri setlerini yükle
    
    Args:
        data_dir: Veri dizini
        
    Returns:
        train_data, val_data, test_data: Her biri (sentences, labels) tuplesi
    """
    train_sentences = np.load(os.path.join(data_dir, 'train_sentences.npy'), allow_pickle=True)
    train_labels = np.load(os.path.join(data_dir, 'train_labels.npy'), allow_pickle=True)
    
    val_sentences = np.load(os.path.join(data_dir, 'validation_sentences.npy'), allow_pickle=True)
    val_labels = np.load(os.path.join(data_dir, 'validation_labels.npy'), allow_pickle=True)
    
    test_sentences = np.load(os.path.join(data_dir, 'test_sentences.npy'), allow_pickle=True)
    test_labels = np.load(os.path.join(data_dir, 'test_labels.npy'), allow_pickle=True)
    
    return (train_sentences, train_labels), (val_sentences, val_labels), (test_sentences, test_labels)


def pad_sequences(sequences, max_len=None):
    """
    Dizileri aynı uzunluğa pad et
    
    Args:
        sequences: Padlenecek diziler listesi
        max_len: Maksimum uzunluk (None ise en uzun diziye göre)
        
    Returns:
        Padlenmiş numpy array
    """
    if max_len is None:
        max_len = max(len(seq) for seq in sequences)
    
    padded = np.zeros((len(sequences), max_len), dtype=np.int32)
    
    for i, seq in enumerate(sequences):
        seq_len = min(len(seq), max_len)
        padded[i, :seq_len] = seq[:seq_len]
    
    return padded


def create_batches(sentences, labels, batch_size, shuffle=True):
    """
    Mini-batch'ler oluştur
    
    Args:
        sentences: Cümleler
        labels: Etiketler
        batch_size: Batch büyüklüğü
        shuffle: Karıştırma yapılsın mı
        
    Returns:
        batches: (batch_sentences, batch_labels) tuplesi listesi
    """
    indices = np.arange(len(sentences))
    if shuffle:
        np.random.shuffle(indices)
    
    batches = []
    
    for start_idx in range(0, len(sentences), batch_size):
        end_idx = min(start_idx + batch_size, len(sentences))
        batch_indices = indices[start_idx:end_idx]
        
        # Her batch için maksimum uzunluğu bul
        batch_sentences = [sentences[i] for i in batch_indices]
        batch_labels = [labels[i] for i in batch_indices]
        
        # Batch içinde pad et
        max_len = max(len(seq) for seq in batch_sentences)
        padded_sentences = pad_sequences(batch_sentences, max_len)
        padded_labels = pad_sequences(batch_labels, max_len)
        
        batches.append((padded_sentences, padded_labels))
    
    return batches


def plot_training_history(history, save_path=None):
    """
    Eğitim geçmişini görselleştir
    
    Args:
        history: Eğitim geçmişi (epoch, train_loss, train_acc, val_loss, val_acc)
        save_path: Kaydedilecek dosya yolu (None ise kaydetmez)
    """
    epochs = [h[0] for h in history]
    
    plt.figure(figsize=(12, 5))
    
    # Loss grafiği
    plt.subplot(1, 2, 1)
    plt.plot(epochs, [h[1] for h in history], 'b-', label='Eğitim Kaybı')
    plt.plot(epochs, [h[3] for h in history], 'r-', label='Doğrulama Kaybı')
    plt.title('Eğitim ve Doğrulama Kaybı')
    plt.xlabel('Epoch')
    plt.ylabel('Kayıp')
    plt.legend()
    plt.grid(True)
    
    # Accuracy grafiği
    plt.subplot(1, 2, 2)
    plt.plot(epochs, [h[2] for h in history], 'b-', label='Eğitim Doğruluğu')
    plt.plot(epochs, [h[4] for h in history], 'r-', label='Doğrulama Doğruluğu')
    plt.title('Eğitim ve Doğrulama Doğruluğu')
    plt.xlabel('Epoch')
    plt.ylabel('Doğruluk')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Grafik kaydedildi: {save_path}")
    
    plt.show()


def train_model(model, train_data, val_data, batch_size=32, epochs=10, 
               learning_rate=0.01, checkpoint_dir="checkpoints"):
    """
    Modeli eğit
    
    Args:
        model: Eğitilecek LSTM modeli
        train_data: Eğitim verisi (sentences, labels)
        val_data: Doğrulama verisi (sentences, labels)
        batch_size: Batch büyüklüğü
        epochs: Epoch sayısı
        learning_rate: Öğrenme oranı
        checkpoint_dir: Checkpoint dizini
        
    Returns:
        history: Eğitim geçmişi (epoch, train_loss, train_acc, val_loss, val_acc) listesi
    """
    train_sentences, train_labels = train_data
    val_sentences, val_labels = val_data
    
    # Checkpoint dizini oluştur
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Early stopping için
    best_val_loss = float('inf')
    patience = 3
    wait = 0
    
    # Eğitim geçmişini sakla
    history = []
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        start_time = time.time()
        
        # Eğitim
        train_batches = create_batches(train_sentences, train_labels, batch_size, shuffle=True)
        train_loss = 0
        train_acc = 0
        
        for batch_idx, (batch_sentences, batch_labels) in enumerate(train_batches):
            # İleri geçiş
            _, loss = model.forward(batch_sentences, batch_labels)
            
            # Geri yayılım
            model.backward(batch_labels)
            
            # Güncelleme
            model.update_params(learning_rate)
            
            # İstatistikler
            train_loss += loss
            train_acc += model.accuracy
            
            # İlerlemeyi göster
            if (batch_idx + 1) % 10 == 0:
                print(f"Batch {batch_idx+1}/{len(train_batches)} - "
                      f"Loss: {loss:.4f} - Acc: {model.accuracy:.4f}")
        
        # Ortalama eğitim istatistikleri
        train_loss /= len(train_batches)
        train_acc /= len(train_batches)
        
        # Doğrulama
        val_batches = create_batches(val_sentences, val_labels, batch_size, shuffle=False)
        val_loss = 0
        val_acc = 0
        
        for batch_sentences, batch_labels in val_batches:
            # İleri geçiş (sadece)
            _, loss = model.forward(batch_sentences, batch_labels)
            
            # İstatistikler
            val_loss += loss
            val_acc += model.accuracy
        
        # Ortalama doğrulama istatistikleri
        val_loss /= len(val_batches)
        val_acc /= len(val_batches)
        
        # Eğitim geçmişine ekle
        history.append((epoch + 1, train_loss, train_acc, val_loss, val_acc))
        
        # Epoch sonuçlarını göster
        elapsed_time = time.time() - start_time
        print(f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f} - "
              f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f} - "
              f"Time: {elapsed_time:.2f}s")
        
        # En iyi modeli kaydet
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            wait = 0
            checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}_valloss_{val_loss:.4f}.npy")
            model.save_checkpoint(checkpoint_path)
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping triggered! No improvement for {patience} epochs.")
                break
    
    return history


def evaluate_model(model, test_data, batch_size=32):
    """
    Modeli değerlendir
    
    Args:
        model: Değerlendirilecek LSTM modeli
        test_data: Test verisi (sentences, labels)
        batch_size: Batch büyüklüğü
        
    Returns:
        test_loss: Test kaybı
        test_acc: Test doğruluğu
    """
    test_sentences, test_labels = test_data
    
    # Test batches
    test_batches = create_batches(test_sentences, test_labels, batch_size, shuffle=False)
    test_loss = 0
    test_acc = 0
    
    for batch_sentences, batch_labels in test_batches:
        # İleri geçiş (sadece)
        _, loss = model.forward(batch_sentences, batch_labels)
        
        # İstatistikler
        test_loss += loss
        test_acc += model.accuracy
    
    # Ortalama test istatistikleri
    test_loss /= len(test_batches)
    test_acc /= len(test_batches)
    
    print(f"\nTest sonuçları:")
    print(f"Test Loss: {test_loss:.4f} - Test Acc: {test_acc:.4f}")
    
    return test_loss, test_acc

def main():
    # Parametreler
    data_dir = "numerized_data"
    embedding_dim = 100
    hidden_size = 128
    batch_size = 32
    epochs = 10
    learning_rate = 0.01
    
    # Veriyi yükle
    print("Veri yükleniyor...")
    train_data, val_data, test_data = load_data(data_dir)
    
    # Kelime ve etiket dağarcığı büyüklüğünü bul
    all_word_ids = np.concatenate([train_data[0], val_data[0], test_data[0]])
    vocab_size = np.max([np.max(seq) for seq in all_word_ids]) + 1
    
    all_label_ids = np.concatenate([train_data[1], val_data[1], test_data[1]])
    output_size = np.max([np.max(seq) for seq in all_label_ids]) + 1
    
    print(f"Kelime dağarcığı büyüklüğü: {vocab_size}")
    print(f"Etiket sayısı: {output_size}")
    
    # Modeli oluştur
    print("Model oluşturuluyor...")
    model = LSTMModel(vocab_size, embedding_dim, hidden_size, output_size)
    
    # Modeli eğit
    print("Eğitim başlıyor...")
    history = train_model(model, train_data, val_data, batch_size, epochs, learning_rate)
    
    # Eğitim geçmişini görselleştir
    plot_training_history(history, save_path="training_history.png")
    
    # Test et
    print("Test ediliyor...")
    evaluate_model(model, test_data, batch_size)


if __name__ == "__main__":
    main()