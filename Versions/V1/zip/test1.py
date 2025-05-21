import os
import json
import re
from typing import Dict, Tuple, List


def build_vocab_and_label_map_from_directory(directory_path: str) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Belirtilen dizindeki sentences.txt ve labels.txt dosyalarını okuyarak kelime dağarcığı ve etiket haritası oluşturur.
    
    Args:
        directory_path (str): sentences.txt ve labels.txt dosyalarının bulunduğu dizinin yolu
        
    Returns:
        Tuple[Dict[str, int], Dict[str, int]]: Kelime dağarcığı ve etiket haritası sözlükleri
    """
    # Dosya yollarını oluştur
    sentences_file = os.path.join(directory_path, "sentences.txt")
    labels_file = os.path.join(directory_path, "labels.txt")
    
    # Dosyaların varlığını kontrol et
    if not os.path.exists(sentences_file) or not os.path.exists(labels_file):
        raise FileNotFoundError(f"sentences.txt veya labels.txt dosyası {directory_path} dizininde bulunamadı.")
    
    # Kelime dağarcığı ve etiket haritası için boş sözlükler oluştur
    vocab = {"PAD": 0, "UNK": 1}
    label_map = {"O": 0}  # "O" etiketini tanımsız (etiket olmayan) durumlar için ekle
    
    # Cümleleri oku ve kelime dağarcığını oluştur
    vocab_id = 2  # PAD ve UNK için 0 ve 1 atandığından 2'den başla
    with open(sentences_file, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            
            words = line.split()
            for word in words:
                if word and word not in vocab:
                    vocab[word] = vocab_id
                    vocab_id += 1
    
    # Etiketleri oku ve etiket haritasını oluştur
    label_id = 1  # "O" etiketine 0 verdiğimiz için 1'den başlıyoruz
    
    # Etiketleri tanımlamak için daha kompleks bir yaklaşım
    with open(labels_file, 'r', encoding='utf-8') as file:
        for line_number, line in enumerate(file, 1):
            line = line.strip()
            if not line:
                continue
            
            # Cümledeki tüm etiketleri bul
            # Etiketler genellikle B-XXX veya I-XXX şeklinde başlayıp köşeli parantezlerle devam eder
            pattern = r'(?:B|I)-[A-ZĞÜŞİÖÇ]+(?:\[[^\]]*\])*'
            tags = re.findall(pattern, line)
            
            # Eğer regex ile etiket bulunamadıysa ve satırda kelimeler varsa
            if not tags and len(line.split()) > 0:
                # Bu satırın kelimeleri için "O" etiketini kullan
                print(f"Uyarı: Satır {line_number}'de etiket bulunamadı, 'O' etiketine sahip kelimeler var.")
                words_in_line = line.split()
                for word in words_in_line:
                    if word not in label_map:
                        # Eğer standart etiket formatına uymuyorsa, kelimeyi "O" olarak etiketle
                        # Burada gerçek bir etiket oluşturmuyoruz, sadece uyarı veriyoruz
                        pass
            else:
                for tag in tags:
                    if tag and tag not in label_map:
                        label_map[tag] = label_id
                        label_id += 1
    
    return vocab, label_map


def process_all_directories(base_path: str, output_dir: str) -> None:
    """
    Ana dizin altındaki tüm alt dizinlerdeki (validation/test/train) veri dosyalarını işler,
    her biri için ayrı kelime dağarcığı ve etiket haritası oluşturur, ayrıca 
    tüm veriler için birleştirilmiş kelime dağarcığı ve etiket haritası oluşturur.
    
    Args:
        base_path (str): Alt dizinleri içeren ana dizinin yolu (örn. 'output/')
        output_dir (str): Oluşturulan JSON dosyalarının kaydedileceği dizin
    """
    # Çıktı dizini yoksa oluştur
    os.makedirs(output_dir, exist_ok=True)
    
    # Birleştirilmiş veri yapıları
    combined_vocab = {"PAD": 0, "UNK": 1}
    combined_label_map = {"O": 0}  # "O" etiketini tanımsız durumlar için ekle
    
    vocab_id = 2
    label_id = 1  # "O" etiketine 0 verdiğimiz için 1'den başlıyoruz
    
    # İşlenecek alt dizinler
    subdirs = ["train", "validation", "test"]
    
    for subdir in subdirs:
        dir_path = os.path.join(base_path, subdir)
        if os.path.isdir(dir_path):
            try:
                # Alt dizin için kelime dağarcığı ve etiket haritası oluştur
                vocab, label_map = build_vocab_and_label_map_from_directory(dir_path)
                
                # Alt dizine özel dosyaları kaydet
                save_json(vocab, os.path.join(output_dir, f"{subdir}_vocab.json"))
                save_json(label_map, os.path.join(output_dir, f"{subdir}_labels.json"))
                
                print(f"{subdir} dizini için dosyalar oluşturuldu.")
                print(f"- Kelime dağarcığı boyutu: {len(vocab)}")
                print(f"- Etiket haritası boyutu: {len(label_map)}")
                
                # Kelimeleri birleştirilmiş sözlüğe ekle
                for word in vocab:
                    if word not in combined_vocab and word not in ["PAD", "UNK"]:
                        combined_vocab[word] = vocab_id
                        vocab_id += 1
                
                # Etiketleri birleştirilmiş haritaya ekle
                for label in label_map:
                    if label not in combined_label_map:
                        combined_label_map[label] = label_id
                        label_id += 1
                        if label != "O":  # "O" zaten 0 olarak eklenmiş durumda
                            label_id += 1
                
            except FileNotFoundError as e:
                print(f"Uyarı: {e}")
    
    # Birleştirilmiş dosyaları kaydet
    save_json(combined_vocab, os.path.join(output_dir, "combined_vocab.json"))
    save_json(combined_label_map, os.path.join(output_dir, "combined_labels.json"))
    
    print("\nBirleştirilmiş dosyalar oluşturuldu:")
    print(f"- Birleştirilmiş kelime dağarcığı boyutu: {len(combined_vocab)}")
    print(f"- Birleştirilmiş etiket haritası boyutu: {len(combined_label_map)}")


def save_json(data: Dict, file_path: str) -> None:
    """
    Veriyi JSON formatında dosyaya kaydeder.
    
    Args:
        data (Dict): Kaydedilecek veri
        file_path (str): Dosya yolu
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def check_sequence_alignment(directory_path: str) -> None:
    """
    Belirtilen dizindeki sentences.txt ve labels.txt dosyalarının satır sayısı ve
    her satırdaki kelime sayısının eşleşip eşleşmediğini kontrol eder.
    
    Args:
        directory_path (str): sentences.txt ve labels.txt dosyalarının bulunduğu dizinin yolu
    """
    sentences_file = os.path.join(directory_path, "sentences.txt")
    labels_file = os.path.join(directory_path, "labels.txt")
    
    if not os.path.exists(sentences_file) or not os.path.exists(labels_file):
        print(f"Dosyalar bulunamadı: {sentences_file} veya {labels_file}")
        return
    
    with open(sentences_file, 'r', encoding='utf-8') as s_file, open(labels_file, 'r', encoding='utf-8') as l_file:
        sentence_lines = s_file.readlines()
        label_lines = l_file.readlines()
        
        if len(sentence_lines) != len(label_lines):
            print(f"Uyarı: Satır sayıları eşleşmiyor! Sentences: {len(sentence_lines)}, Labels: {len(label_lines)}")
            return
        
        for line_idx, (sentence, label) in enumerate(zip(sentence_lines, label_lines), 1):
            sentence_words = sentence.strip().split()
            label_words = re.findall(r'(?:B|I)-[A-ZĞÜŞİÖÇ]+(?:\[[^\]]*\])*|O', label.strip())
            
            if len(sentence_words) != len(label_words):
                print(f"Uyarı: Satır {line_idx}'de kelime sayıları eşleşmiyor!")
                print(f"  Cümle ({len(sentence_words)} kelime): {sentence.strip()}")
                print(f"  Etiket ({len(label_words)} etiket): {label.strip()}")
    
    print("Kontrol tamamlandı.")


if __name__ == "__main__":
    # Parametreleri yapılandırın
    base_directory = "output/"  # Ana dizin (train, validation, test alt dizinlerini içeren)
    output_directory = "vocabulary_files/"  # Çıktı dosyalarının kaydedileceği dizin
    
    # İşleme başlamadan önce veri doğruluğunu kontrol et
    for subdir in ["train", "validation", "test"]:
        print(f"{subdir} dizini için doğruluk kontrolü:")
        check_sequence_alignment(os.path.join(base_directory, subdir))
        print()
    
    # Tüm dizinleri işle ve dosyaları oluştur
    process_all_directories(base_directory, output_directory)