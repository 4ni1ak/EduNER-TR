import os
import json
import re
from typing import Dict, Tuple, List, Optional
import numpy as np


def load_vocab_and_labels(vocab_path: str, labels_path: str) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Kelime dağarcığı ve etiket haritasını JSON dosyalarından yükler.
    
    Args:
        vocab_path (str): Kelime dağarcığı JSON dosyasının yolu
        labels_path (str): Etiket haritası JSON dosyasının yolu
        
    Returns:
        Tuple[Dict[str, int], Dict[str, int]]: Kelime dağarcığı ve etiket haritası
    """
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    
    with open(labels_path, 'r', encoding='utf-8') as f:
        label_map = json.load(f)
    
    return vocab, label_map


def numericalize_data(directory_path: str, vocab: Dict[str, int], label_map: Dict[str, int], 
                      max_seq_length: Optional[int] = None) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Belirtilen dizindeki sentences.txt ve labels.txt dosyalarındaki metinleri ve etiketleri 
    sayısal ID'lere çevirir ve sabit uzunluğa getirir.
    
    Args:
        directory_path (str): sentences.txt ve labels.txt dosyalarının bulunduğu dizinin yolu
        vocab (Dict[str, int]): Kelime dağarcığı (kelimelerin ID eşleşmeleri)
        label_map (Dict[str, int]): Etiket haritası (etiketlerin ID eşleşmeleri)
        max_seq_length (int, optional): Maksimum cümle uzunluğu. None ise en uzun cümlenin
                                        uzunluğu + 12 olarak hesaplanır.
        
    Returns:
        Tuple[List[List[int]], List[List[int]]]: Sayısallaştırılmış cümleler ve etiketler
    """
    sentences_file = os.path.join(directory_path, "sentences.txt")
    labels_file = os.path.join(directory_path, "labels.txt")
    
    if not os.path.exists(sentences_file) or not os.path.exists(labels_file):
        raise FileNotFoundError(f"sentences.txt veya labels.txt dosyası {directory_path} dizininde bulunamadı.")
    
    # Cümleleri ve etiketleri oku
    with open(sentences_file, 'r', encoding='utf-8') as s_file, open(labels_file, 'r', encoding='utf-8') as l_file:
        sentence_lines = [line.strip() for line in s_file if line.strip()]
        label_lines = [line.strip() for line in l_file if line.strip()]
    
    # Verileri sayısallaştır
    numeric_sentences = []
    numeric_labels = []
    
    # En uzun cümleyi bul (kelime sayısı olarak)
    max_sentence_length = max(len(line.split()) for line in sentence_lines) if sentence_lines else 0
    
    # Maksimum uzunluk hesaplama (istenen şekilde en uzun cümlenin 12 fazlası)
    if max_seq_length is None:
        max_seq_length = max_sentence_length + 12
        print(f"Maksimum cümle uzunluğu: {max_seq_length} (En uzun cümle: {max_sentence_length} kelime + 12)")
    
    # Regex pattern (etiketleri tanımlamak için)
    pattern = r'(?:B|I)-[A-ZĞÜŞİÖÇ]+(?:\[[^\]]*\])*|O'
    
    for sentence, label_line in zip(sentence_lines, label_lines):
        # Kelimelere ve etiketlere ayır
        words = sentence.split()
        labels = re.findall(pattern, label_line)
        
        # Her kelimeyi sayısallaştır
        numeric_sentence = [vocab.get(word, vocab["UNK"]) for word in words]
        
        # Her etiketi sayısallaştır
        numeric_label = [label_map.get(label, 0) for label in labels]  # 0, "O" etiketi için
        
        # Kelime ve etiket sayılarını kontrol et
        if len(numeric_sentence) != len(numeric_label):
            print(f"Uyarı: Kelime ve etiket sayıları eşleşmiyor!")
            print(f"  Cümle ({len(numeric_sentence)} kelime): {sentence}")
            print(f"  Etiket ({len(numeric_label)} etiket): {label_line}")
            # Duruma göre boyutları eşleştir
            min_len = min(len(numeric_sentence), len(numeric_label))
            numeric_sentence = numeric_sentence[:min_len]
            numeric_label = numeric_label[:min_len]
        
        # PAD ekleme (sabit uzunluğa getirme)
        if len(numeric_sentence) < max_seq_length:
            # Kelimelere PAD ekle
            numeric_sentence.extend([vocab["PAD"]] * (max_seq_length - len(numeric_sentence)))
            # Etiketlere PAD ekle (genellikle "O" etiketinin ID'si kullanılır)
            numeric_label.extend([0] * (max_seq_length - len(numeric_label)))
        else:
            # Uzun cümleleri kırp
            numeric_sentence = numeric_sentence[:max_seq_length]
            numeric_label = numeric_label[:max_seq_length]
        
        numeric_sentences.append(numeric_sentence)
        numeric_labels.append(numeric_label)
    
    return numeric_sentences, numeric_labels


def save_numerized_data(numeric_sentences: List[List[int]], numeric_labels: List[List[int]], 
                       output_dir: str, prefix: str) -> None:
    """
    Sayısallaştırılmış verileri NumPy dizisi olarak kaydeder.
    
    Args:
        numeric_sentences (List[List[int]]): Sayısallaştırılmış cümleler
        numeric_labels (List[List[int]]): Sayısallaştırılmış etiketler
        output_dir (str): Çıktı dizini
        prefix (str): Dosya ön eki (örn. "train", "validation", "test")
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # NumPy dizileri oluştur
    sentences_array = np.array(numeric_sentences, dtype=np.int32)
    labels_array = np.array(numeric_labels, dtype=np.int32)
    
    # Dosyalara kaydet
    np.save(os.path.join(output_dir, f"{prefix}_sentences.npy"), sentences_array)
    np.save(os.path.join(output_dir, f"{prefix}_labels.npy"), labels_array)
    
    print(f"{prefix.title()} verileri kaydedildi.")
    print(f"- Cümleler: {sentences_array.shape}")
    print(f"- Etiketler: {labels_array.shape}")


def process_all_for_numericalization(base_path: str, vocab_dir: str, output_dir: str, 
                                   max_seq_length: Optional[int] = None, use_combined: bool = True) -> None:
    """
    Tüm alt dizinlerdeki verileri sayısallaştırır ve kaydeder.
    
    Args:
        base_path (str): Veri dizinlerinin bulunduğu ana dizin
        vocab_dir (str): Kelime dağarcığı ve etiket haritası dosyalarının bulunduğu dizin
        output_dir (str): Sayısallaştırılmış verilerin kaydedileceği dizin
        max_seq_length (int, optional): Maksimum cümle uzunluğu
        use_combined (bool): Birleştirilmiş vocab ve etiket haritası kullanılsın mı?
    """
    # İşlenecek alt dizinler
    subdirs = ["train", "validation", "test"]
    
    # Eğer birleştirilmiş vocab ve etiket haritası kullanılacaksa
    if use_combined:
        vocab_path = os.path.join(vocab_dir, "combined_vocab.json")
        labels_path = os.path.join(vocab_dir, "combined_labels.json")
        vocab, label_map = load_vocab_and_labels(vocab_path, labels_path)
        print(f"Birleştirilmiş vocab ve etiket haritası yüklendi.")
        print(f"- Vocab boyutu: {len(vocab)}")
        print(f"- Etiket haritası boyutu: {len(label_map)}")
    
    # En uzun cümleyi bul (tüm alt dizinlerdeki en uzun)
    longest_sentence = 0
    for subdir in subdirs:
        sentences_file = os.path.join(base_path, subdir, "sentences.txt")
        if os.path.exists(sentences_file):
            with open(sentences_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        longest_sentence = max(longest_sentence, len(line.strip().split()))
    
    # Maksimum uzunluğu ayarla
    if max_seq_length is None:
        max_seq_length = longest_sentence + 12
        print(f"Maksimum cümle uzunluğu: {max_seq_length} (En uzun cümle: {longest_sentence} kelime + 12)")
    
    # Her alt dizin için veriyi sayısallaştır
    for subdir in subdirs:
        dir_path = os.path.join(base_path, subdir)
        if os.path.isdir(dir_path):
            try:
                # Eğer her alt dizin için ayrı vocab ve etiket haritası kullanılacaksa
                if not use_combined:
                    vocab_path = os.path.join(vocab_dir, f"{subdir}_vocab.json")
                    labels_path = os.path.join(vocab_dir, f"{subdir}_labels.json")
                    vocab, label_map = load_vocab_and_labels(vocab_path, labels_path)
                    print(f"{subdir} için vocab ve etiket haritası yüklendi.")
                    print(f"- Vocab boyutu: {len(vocab)}")
                    print(f"- Etiket haritası boyutu: {len(label_map)}")
                
                # Sayısallaştır
                print(f"{subdir} verileri sayısallaştırılıyor...")
                numeric_sentences, numeric_labels = numericalize_data(
                    dir_path, vocab, label_map, max_seq_length
                )
                
                # Kaydet
                save_numerized_data(numeric_sentences, numeric_labels, output_dir, subdir)
                
            except FileNotFoundError as e:
                print(f"Uyarı: {e}")


if __name__ == "__main__":
    # Parametreleri yapılandırın
    base_directory = "output/"  # Veri dizinlerinin bulunduğu ana dizin
    vocab_directory = "vocabulary_files/"  # Kelime dağarcığı ve etiket haritası dosyalarının bulunduğu dizin
    output_directory = "numerized_data/"  # Sayısallaştırılmış verilerin kaydedileceği dizin
    
    # Tüm alt dizinleri işle ve sayısallaştırılmış verileri oluştur
    process_all_for_numericalization(
        base_directory, 
        vocab_directory, 
        output_directory,
        max_seq_length=None,  # Otomatik hesaplanacak (en uzun cümle + 12)
        use_combined=True  # Birleştirilmiş vocab ve etiket haritası kullan
    )