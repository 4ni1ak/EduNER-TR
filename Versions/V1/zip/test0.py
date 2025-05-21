import os
import csv
import random
from typing import List, Dict, Tuple


def load_simple_text_data(file_path: str) -> List[str]:
    """Basit text dosyasını satır satır okur ve listeye ekler."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return [line.strip() for line in file if line.strip()]
    except Exception as e:
        print(f"Dosya okuma hatası ({file_path}): {e}")
        return []


def load_csv_data(file_path: str, column_name: str = None) -> List[str]:
    """CSV dosyasını okur. Eğer column_name belirtilmezse ilk sütunu alır."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            # İlk satırda başlık var mı kontrol et
            first_line = file.readline().strip()
            file.seek(0)  # Dosya göstergesini başa al
            
            if ',' in first_line:  # CSV yapısı muhtemelen var
                reader = csv.reader(file)
                headers = next(reader, None)  # Başlıkları oku
                
                if column_name and column_name in headers:
                    col_index = headers.index(column_name)
                else:
                    col_index = 0  # Varsayılan olarak ilk sütun
                
                return [row[col_index] for row in reader if row and row[col_index].strip()]
            else:  # Basit liste formatı
                return [line.strip() for line in file if line.strip()]
    except Exception as e:
        print(f"Dosya okuma hatası ({file_path}): {e}")
        return []


def load_program_data(file_path: str) -> Dict[str, List[str]]:
    """Program CSV dosyasını okur ve tür bazında programları sözlük olarak döndürür."""
    result = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            headers = next(reader, None)  # Başlıkları oku
            
            if headers and len(headers) >= 2:
                tur_index = 0  # 'Tür' sütunu için indeks
                program_index = 1  # 'Program' sütunu için indeks
                
                for row in reader:
                    if row and len(row) >= 2:
                        tur = row[tur_index].strip()
                        program = row[program_index].strip()
                        
                        if tur not in result:
                            result[tur] = []
                        
                        if program:
                            result[tur].append(program)
            else:
                print(f"Uygun başlık bulunamadı: {file_path}")
        
        return result
    except Exception as e:
        print(f"Program dosyası okuma hatası ({file_path}): {e}")
        return {}


def load_university_data(file_path: str) -> Dict[str, List[str]]:
    """Üniversite CSV dosyasını okur ve tür bazında üniversiteleri sözlük olarak döndürür."""
    result = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            headers = next(reader, None)  # Başlıkları oku
            
            if headers and len(headers) >= 2:
                tur_index = 0  # 'Tür' sütunu için indeks
                uni_index = 1  # 'Üniversite Adı' sütunu için indeks
                
                for row in reader:
                    if row and len(row) >= 2:
                        tur = row[tur_index].strip()
                        uni = row[uni_index].strip()
                        
                        if tur not in result:
                            result[tur] = []
                        
                        if uni:
                            result[tur].append(uni)
            else:
                print(f"Uygun başlık bulunamadı: {file_path}")
        
        return result
    except Exception as e:
        print(f"Üniversite dosyası okuma hatası ({file_path}): {e}")
        return {}


def generate_random_number() -> str:
    """9 haneli rastgele öğrenci numarası oluşturur."""
    return ''.join(random.choice('0123456789') for _ in range(9))


def create_sentence_and_labels() -> Tuple[List[str], List[str]]:
    """Rastgele sıralama ile bir cümle ve etiketlerini oluşturur."""
    # Veri bileşenlerini hazırla
    components = []
    
    # İsim ve cinsiyet seçimi
    # İsim ve cinsiyet seçimi
    # İsim ve cinsiyet seçimi
    # İsim ve cinsiyet seçimi
    # İsim ve cinsiyet seçimi
    #ToDo
    cinsiyet_secimi = random.choice(["ERKEK", "KADIN", "BILINMIYOR"])
    
    if cinsiyet_secimi == "ERKEK":
        isim = random.choice(erkek_isimleri)
        components.append((isim.lower(), f"B-ISIM[ERKEK]"))
    elif cinsiyet_secimi == "KADIN":
        isim = random.choice(kadin_isimleri)
        components.append((isim.lower(), f"B-ISIM[KADIN]"))
    else:  # BILINMIYOR
        # Rastgele bir isim seç (erkek veya kadın listesinden)
        if random.choice([True, False]):
            isim = random.choice(erkek_isimleri)
        else:
            isim = random.choice(kadin_isimleri)
        components.append((isim.lower(), f"B-ISIM[BILINMIYOR]"))
    
    # Soyisim seçimi
    soyisim = random.choice(soyisimler)
    components.append((soyisim.lower(), "B-SOYISIM"))
    
    # Numara oluşturma
    numara = generate_random_number()
    components.append((numara, "B-NUMARA"))
    
    # Program seçimi (lisans veya önlisans)
    if random.choice([True, False]):  # Lisans veya önlisans seçimi
        program_type = random.choice(list(lisans_programlari.keys()))
        program = random.choice(lisans_programlari[program_type])
        program_level = "LISANS"
    else:
        program_type = random.choice(list(onlisans_programlari.keys()))
        program = random.choice(onlisans_programlari[program_type])
        program_level = "ONLISANS"
    
    # Program adını kaydet (etiket formatına eklenecek)
    program_tam_ad = program.upper()
    
    # Programın kelimelerini ayır ve etiketle
    program_words = program.lower().split()
    for i, word in enumerate(program_words):
        if i == 0:
            components.append((word, f"B-BOLUM[{program_level}][{program_tam_ad}]"))
        else:
            components.append((word, f"I-BOLUM[{program_level}][{program_tam_ad}]"))
    
    
    # Üniversite seçimi
    if random.choice([True, False]):  # Devlet veya özel üniversite seçimi
        uni_type = random.choice(list(lisans_uni.keys()))
        universite = random.choice(lisans_uni[uni_type])
        
        # Üniversite türünü belirle
        if uni_type == "Devlet Üniversiteleri":
            uni_type_tag = "DEVLET"
        else:
            uni_type_tag = "VAKIF"
    else:
        uni_type = random.choice(list(onlisans_uni.keys()))
        universite = random.choice(onlisans_uni[uni_type])
        
        # Üniversite türünü belirle
        if uni_type == "Devlet Üniversiteleri":
            uni_type_tag = "DEVLET"
        else:
            uni_type_tag = "VAKIF"
    
    # Üniversite adını kaydet (etiket formatına eklenecek)
    universite_tam_ad = universite.upper()
    
    # Üniversite adının kelimelerini ayır ve etiketle
    universite_words = universite.lower().split()
    for i, word in enumerate(universite_words):
        if i == 0:
            components.append((word, f"B-UNIVERSITE[{universite_tam_ad}][{uni_type_tag}]"))
        else:
            components.append((word, f"I-UNIVERSITE[{universite_tam_ad}][{uni_type_tag}]"))
    
    # Bileşenleri karıştır
    random.shuffle(components)
    
    # Cümle ve etiketleri ayrı listelere ayır
    sentence = [item[0] for item in components]
    labels = [item[1] for item in components]
    
    return sentence, labels


def save_to_files(sentences: List[List[str]], labels: List[List[str]], output_dir: str = "output") -> None:
    """Cümleleri ve etiketleri dosyalara kaydeder."""
    # Output klasörünü oluştur (yoksa)
    os.makedirs(output_dir, exist_ok=True)
    
    # Tek bir dosyaya tüm cümle ve etiketleri kaydet
    with open(f"{output_dir}/sentences.txt", "w", encoding="utf-8") as sent_file, \
         open(f"{output_dir}/labels.txt", "w", encoding="utf-8") as label_file:
        
        for i, (sentence, label) in enumerate(zip(sentences, labels)):
            sent_file.write(" ".join(sentence) + "\n")
            label_file.write(" ".join(label) + "\n")
    
    # Ayrı ayrı dosyalara da kaydet
    for i, (sentence, label) in enumerate(zip(sentences, labels)):
        with open(f"{output_dir}/sentence_{i+1:03d}.txt", "w", encoding="utf-8") as sent_file, \
             open(f"{output_dir}/labels_{i+1:03d}.txt", "w", encoding="utf-8") as label_file:
            
            sent_file.write(" ".join(sentence) + "\n")
            label_file.write(" ".join(label) + "\n")
    
    print(f"{len(sentences)} adet cümle ve etiket başarıyla {output_dir} klasörüne kaydedildi.")


def split_and_save_data(sentences: List[List[str]], labels: List[List[str]], 
                       train_ratio: float = 0.7, val_ratio: float = 0.15, 
                       output_dir: str = "output") -> None:
    """
    Veriyi eğitim, doğrulama ve test setlerine böler ve her bir örneği ayrı dosyalara kaydeder.
    
    Args:
        sentences: Cümlelerin listesi (her cümle kelimelerin listesi olarak).
        labels: Etiketlerin listesi (her etiket seti, etiketlerin listesi olarak).
        train_ratio: Eğitim seti için ayrılacak verinin oranı (0.0-1.0 arası).
        val_ratio: Doğrulama seti için ayrılacak verinin oranı (0.0-1.0 arası).
        output_dir: Çıktı dosyalarının kaydedileceği dizin.
    
    Not: 
        test_ratio otomatik olarak hesaplanır (1.0 - train_ratio - val_ratio).
    """
    # Output klasörünü oluştur (yoksa)
    os.makedirs(output_dir, exist_ok=True)
    
    # Veriyi birleştir (cümle ve etiket çiftleri)
    data = list(zip(sentences, labels))
    
    # Veriyi karıştır
    random.shuffle(data)
    
    # Veri sayısını hesapla
    total_samples = len(data)
    train_size = int(total_samples * train_ratio)
    val_size = int(total_samples * val_ratio)
    
    # Veriyi böl
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]
    
    # İstatistikleri yazdır
    print(f"Toplam veri: {total_samples} örnek")
    print(f"Eğitim seti: {len(train_data)} örnek ({len(train_data)/total_samples:.1%})")
    print(f"Doğrulama seti: {len(val_data)} örnek ({len(val_data)/total_samples:.1%})")
    print(f"Test seti: {len(test_data)} örnek ({len(test_data)/total_samples:.1%})")
    
    # Setlere göre klasörler oluştur
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "validation") 
    test_dir = os.path.join(output_dir, "test")
    
    for directory in [train_dir, val_dir, test_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # Her bir set için hem toplu dosya hem de ayrı dosyalar oluştur
    
    # 1. Eğitim seti
    # Toplu dosya
    with open(f"{train_dir}/sentences.txt", "w", encoding="utf-8") as sent_file, \
         open(f"{train_dir}/labels.txt", "w", encoding="utf-8") as label_file:
        
        for sentence, label in train_data:
            sent_file.write(" ".join(sentence) + "\n")
            label_file.write(" ".join(label) + "\n")
    
    # Ayrı dosyalar
    for i, (sentence, label) in enumerate(train_data):
        with open(f"{train_dir}/sentence_{i+1:03d}.txt", "w", encoding="utf-8") as sent_file, \
             open(f"{train_dir}/labels_{i+1:03d}.txt", "w", encoding="utf-8") as label_file:
            
            sent_file.write(" ".join(sentence) + "\n")
            label_file.write(" ".join(label) + "\n")
            
    # 2. Doğrulama seti
    # Toplu dosya
    with open(f"{val_dir}/sentences.txt", "w", encoding="utf-8") as sent_file, \
         open(f"{val_dir}/labels.txt", "w", encoding="utf-8") as label_file:
        
        for sentence, label in val_data:
            sent_file.write(" ".join(sentence) + "\n")
            label_file.write(" ".join(label) + "\n")
    
    # Ayrı dosyalar
    for i, (sentence, label) in enumerate(val_data):
        with open(f"{val_dir}/sentence_{i+1:03d}.txt", "w", encoding="utf-8") as sent_file, \
             open(f"{val_dir}/labels_{i+1:03d}.txt", "w", encoding="utf-8") as label_file:
            
            sent_file.write(" ".join(sentence) + "\n")
            label_file.write(" ".join(label) + "\n")
    
    # 3. Test seti
    # Toplu dosya
    with open(f"{test_dir}/sentences.txt", "w", encoding="utf-8") as sent_file, \
         open(f"{test_dir}/labels.txt", "w", encoding="utf-8") as label_file:
        
        for sentence, label in test_data:
            sent_file.write(" ".join(sentence) + "\n")
            label_file.write(" ".join(label) + "\n")
    
    # Ayrı dosyalar
    for i, (sentence, label) in enumerate(test_data):
        with open(f"{test_dir}/sentence_{i+1:03d}.txt", "w", encoding="utf-8") as sent_file, \
             open(f"{test_dir}/labels_{i+1:03d}.txt", "w", encoding="utf-8") as label_file:
            
            sent_file.write(" ".join(sentence) + "\n")
            label_file.write(" ".join(label) + "\n")
    
    print(f"Veriler başarıyla bölündü ve {output_dir} klasörüne kaydedildi.")
    print(f"  - Eğitim seti: {train_dir} ({len(train_data)} dosya)")
    print(f"  - Doğrulama seti: {val_dir} ({len(val_data)} dosya)")
    print(f"  - Test seti: {test_dir} ({len(test_data)} dosya)")


def main():
    """Ana fonksiyon"""
    global erkek_isimleri, kadin_isimleri, soyisimler
    global lisans_programlari, onlisans_programlari
    global lisans_uni, onlisans_uni
    
    # Dosya yollarını tanımla
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_files = {
        'erkek_isimleri': os.path.join(current_dir, 'turkish-names', 'erkek_isimleri.csv'),
        'kadin_isimleri': os.path.join(current_dir, 'turkish-names', 'kadn_isimleri.csv'),
        'soyisimler': os.path.join(current_dir, 'turkish-names', 'soy_isimler.csv'),
        'lisans_programlari': os.path.join(current_dir, 'data', 'lisans_programları.csv'),
        'lisans_uni': os.path.join(current_dir, 'data', 'lisans_üni.csv'),
        'onlisans_programlari': os.path.join(current_dir, 'data', 'önlisans_programları.csv'),
        'onlisans_uni': os.path.join(current_dir, 'data', 'önllisans_üni.csv')
    }
    
    # Verileri oku - başlık satırı yok, doğrudan isimleri içeren dosyalar
    erkek_isimleri = load_simple_text_data(data_files['erkek_isimleri'])
    kadin_isimleri = load_simple_text_data(data_files['kadin_isimleri'])
    soyisimler = load_simple_text_data(data_files['soyisimler'])
    
    # Diğer dosyaları CSV olarak oku
    lisans_programlari = load_program_data(data_files['lisans_programlari'])
    onlisans_programlari = load_program_data(data_files['onlisans_programlari'])
    
    lisans_uni = load_university_data(data_files['lisans_uni'])
    onlisans_uni = load_university_data(data_files['onlisans_uni'])
    
    # Yüklenen verilerin kontrolü
    if not erkek_isimleri or not kadin_isimleri or not soyisimler:
        print("İsimler veya soyisimler yüklenemedi. Dosya yollarını kontrol edin.")
        return
    
    if not lisans_programlari or not onlisans_programlari:
        print("Program verileri yüklenemedi. Dosya yollarını kontrol edin.")
        return
    
    if not lisans_uni or not onlisans_uni:
        print("Üniversite verileri yüklenemedi. Dosya yollarını kontrol edin.")
        return
    
    # Kullanıcıdan üretilecek örnek sayısını al
    while True:
        try:
            ornekler_sayisi = int(input("Kaç adet örnek üretmek istiyorsunuz? "))
            if ornekler_sayisi <= 0:
                print("Lütfen pozitif bir sayı girin.")
                continue
            break
        except ValueError:
            print("Lütfen geçerli bir sayı girin.")
    
    # Örnekleri oluştur
    sentences = []
    labels = []
    for _ in range(ornekler_sayisi):
        sentence, label = create_sentence_and_labels()
        sentences.append(sentence)
        labels.append(label)
    
    # Kullanıcıya veri bölme seçeneği sun
    while True:
        bolunecek_mi = input("Veriyi eğitim/doğrulama/test olarak bölmek istiyor musunuz? (E/H): ").strip().upper()
        if bolunecek_mi in ['E', 'H']:
            break
        print("Lütfen 'E' veya 'H' girin.")
    
    if bolunecek_mi == 'E':
        # Bölme oranlarını belirle
        while True:
            try:
                train_ratio = float(input("Eğitim seti oranı (örn: 0.7): "))
                val_ratio = float(input("Doğrulama seti oranı (örn: 0.15): "))
                
                # Oran kontrolü
                if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1.0:
                    print("Geçersiz oranlar! Eğitim ve doğrulama oranlarının toplamı 1.0'dan küçük olmalıdır.")
                    continue
                break
            except ValueError:
                print("Lütfen geçerli bir sayı girin.")
        
        # Veriyi böl ve kaydet
        split_and_save_data(sentences, labels, train_ratio, val_ratio)
    else:
        # Tüm veriyi tek bir dosyaya kaydet
        save_to_files(sentences, labels)


if __name__ == "__main__":
    main()