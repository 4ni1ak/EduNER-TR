import os
import csv
import random
import unicodedata
from typing import List, Dict, Tuple


def load_simple_text_data(file_path: str) -> List[str]:
    """Reads a simple text file line by line and adds them to a list."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return [line.strip() for line in file if line.strip()]
    except Exception as e:
        print(f"File reading error ({file_path}): {e}")
        return []

20
def generate_random_number() -> str:
    """Generates a 9-digit random student number."""
    return ''.join(random.choice('0123456789') for _ in range(9))


def normalize_turkish_text(text: str) -> str:
    """Normalizes Turkish text, especially correcting the letter 'i'."""
    # First, convert all text to lowercase
    text = text.lower()
    
    # Normalize (NFC: composes characters and diacritical marks)
    text = unicodedata.normalize('NFC', text)
    
    # Special Turkish character corrections
    replacements = {
        'i̇': 'i',  # Dotted lowercase i correction
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    return text


def create_sentence_and_labels() -> Tuple[List[str], List[str]]:
    """Creates a sentence and its labels in random order."""
    # Prepare data components
    components = []
    
    # Name selection (without gender distinction)
    tum_isimler = erkek_isimleri + kadin_isimleri
    isim = random.choice(tum_isimler)
    components.append((normalize_turkish_text(isim), "B-ISIM"))
    
    # Surname selection
    soyisim = random.choice(soyisimler)
    components.append((normalize_turkish_text(soyisim), "B-SOYISIM"))
    
    # Number generation
    numara = generate_random_number()
    components.append((numara, "B-NUMARA"))
    
    # Program selection (undergraduate)
    program = random.choice(program_listesi)
    
    # Split and label the program's words
    program_words = normalize_turkish_text(program).split()
    for i, word in enumerate(program_words):
        if i == 0:
            components.append((word, "B-BOLUM"))
        else:
            components.append((word, "I-BOLUM"))
    
    # University selection (without State/Foundation distinction)
    universite = random.choice(universite_listesi)
    
    # Split and label the university name's words
    universite_words = normalize_turkish_text(universite).split()
    for i, word in enumerate(universite_words):
        if i == 0:
            components.append((word, "B-UNIVERSITE"))
        else:
            components.append((word, "I-UNIVERSITE"))
    
    # Shuffle components
    random.shuffle(components)
    
    # Separate sentence and labels into different lists
    sentence = [item[0] for item in components]
    labels = [item[1] for item in components]
    
    return sentence, labels


def save_to_files(sentences: List[List[str]], labels: List[List[str]], output_dir: str = "01_query") -> None:
    """Saves sentences and labels to separate files."""
    # Create the output folder (if it doesn't exist)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to separate files - sentence and label in separate files for each sample
    for i, (sentence, label) in enumerate(zip(sentences, labels)):
        # Sentence file
        with open(f"{output_dir}/sentence_{i+1}.txt", "w", encoding="utf-8") as sent_file:
            sent_file.write(" ".join(sentence) + "\n")
        
        # Label file
        with open(f"{output_dir}/label_{i+1}.txt", "w", encoding="utf-8") as label_file:
            label_file.write(" ".join(label) + "\n")
    
    # Save all sentences to one file, all labels to a separate file
    with open(f"{output_dir}/sentence_0.txt", "w", encoding="utf-8") as sent_file:
        for sentence in sentences:
            sent_file.write(" ".join(sentence) + "\n")
    
    with open(f"{output_dir}/label_0.txt", "w", encoding="utf-8") as label_file:
        for label in labels:
            label_file.write(" ".join(label) + "\n")
    
    print(f"{len(sentences)} sentences and labels were successfully saved to the {output_dir} folder.")
    print(f"All sentences were saved to 'sentence_0.txt' and all labels to 'label_0.txt'.")


def split_and_save_data(sentences: List[List[str]], labels: List[List[str]], 
                      train_ratio: float = 0.7, val_ratio: float = 0.15, 
                      output_dir: str = "01_query") -> None:
    """
    Splits the data into training, validation, and test sets and saves each set to separate folders.
    
    Args:
        sentences: List of sentences
        labels: List of labels
        train_ratio: Training set ratio (between 0-1)
        val_ratio: Validation set ratio (between 0-1)
        output_dir: Output directory
    """
    # Combine data
    data = list(zip(sentences, labels))
    random.shuffle(data)  # Shuffle
    
    # Calculate split indices
    total = len(data)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    
    # Split data
    train_data = data[:train_size]
    val_data = data[train_size:train_size+val_size]
    test_data = data[train_size+val_size:]
    
    # Create folders
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "validation")
    test_dir = os.path.join(output_dir, "test")
    
    for directory in [train_dir, val_dir, test_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # Save data
    # 1. Train dataset
    train_sentences = [item[0] for item in train_data]
    train_labels = [item[1] for item in train_data]
    
    # Save all sentences to a single file
    with open(f"{train_dir}/sentence_0.txt", "w", encoding="utf-8") as sent_file:
        for sentence in train_sentences:
            sent_file.write(" ".join(sentence) + "\n")
    
    # Save all labels to a single file
    with open(f"{train_dir}/label_0.txt", "w", encoding="utf-8") as label_file:
        for label in train_labels:
            label_file.write(" ".join(label) + "\n")
    
    # Separate files for each sample (sentence and label in separate files)
    for i, (sentence, label) in enumerate(zip(train_sentences, train_labels)):
        # Sentence file
        with open(f"{train_dir}/sentence_{i+1}.txt", "w", encoding="utf-8") as sent_file:
            sent_file.write(" ".join(sentence) + "\n")
        
        # Label file
        with open(f"{train_dir}/label_{i+1}.txt", "w", encoding="utf-8") as label_file:
            label_file.write(" ".join(label) + "\n")
    
    # 2. Validation dataset
    val_sentences = [item[0] for item in val_data]
    val_labels = [item[1] for item in val_data]
    
    # Save all sentences to a single file
    with open(f"{val_dir}/sentence_0.txt", "w", encoding="utf-8") as sent_file:
        for sentence in val_sentences:
            sent_file.write(" ".join(sentence) + "\n")
    
    # Save all labels to a single file
    with open(f"{val_dir}/label_0.txt", "w", encoding="utf-8") as label_file:
        for label in val_labels:
            label_file.write(" ".join(label) + "\n")
    
    # Separate files for each sample (sentence and label in separate files)
    for i, (sentence, label) in enumerate(zip(val_sentences, val_labels)):
        # Sentence file
        with open(f"{val_dir}/sentence_{i+1}.txt", "w", encoding="utf-8") as sent_file:
            sent_file.write(" ".join(sentence) + "\n")
        
        # Label file
        with open(f"{val_dir}/label_{i+1}.txt", "w", encoding="utf-8") as label_file:
            label_file.write(" ".join(label) + "\n")
    
    # 3. Test dataset
    test_sentences = [item[0] for item in test_data]
    test_labels = [item[1] for item in test_data]
    
    # Save all sentences to a single file
    with open(f"{test_dir}/sentence_0.txt", "w", encoding="utf-8") as sent_file:
        for sentence in test_sentences:
            sent_file.write(" ".join(sentence) + "\n")
    
    # Save all labels to a single file
    with open(f"{test_dir}/label_0.txt", "w", encoding="utf-8") as label_file:
        for label in test_labels:
            label_file.write(" ".join(label) + "\n")
    
    # Separate files for each sample (sentence and label in separate files)
    for i, (sentence, label) in enumerate(zip(test_sentences, test_labels)):
        # Sentence file
        with open(f"{test_dir}/sentence_{i+1}.txt", "w", encoding="utf-8") as sent_file:
            sent_file.write(" ".join(sentence) + "\n")
        
        # Label file
        with open(f"{test_dir}/label_{i+1}.txt", "w", encoding="utf-8") as label_file:
            label_file.write(" ".join(label) + "\n")
    
    # Statistical information
    print(f"Dataset split:")
    print(f"  Train: {len(train_data)} samples ({len(train_data)/total:.1%})")
    print(f"  Validation: {len(val_data)} samples ({len(val_data)/total:.1%})")
    print(f"  Test: {len(test_data)} samples ({len(test_data)/total:.1%})")
    print(f"All sets were saved to the following folders: {train_dir}, {val_dir}, {test_dir}")


def main():
    """Main function"""
    global erkek_isimleri, kadin_isimleri, soyisimler
    global program_listesi, universite_listesi
    
    # Define file paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_files = {
        'erkek_isimleri': os.path.join(current_dir, '00.data_cleaning', 'data', 'erkek_isimleri.csv'),
        'kadin_isimleri': os.path.join(current_dir, '00.data_cleaning', 'data', 'kadn_isimleri.csv'),
        'soyisimler': os.path.join(current_dir, '00.data_cleaning', 'data', 'soy_isimler.csv'),
        'lisans_programlari': os.path.join(current_dir, '00.data_cleaning', 'data', 'lisans_programları.csv'),
        'uni_listesi': os.path.join(current_dir, '00.data_cleaning', 'data', 'lisans_üni.csv'),
    }
    
    # Read data - name and surname data
    erkek_isimleri = load_simple_text_data(data_files['erkek_isimleri'])[1:]  # Skip header
    kadin_isimleri = load_simple_text_data(data_files['kadin_isimleri'])[1:]  # Skip header
    soyisimler = load_simple_text_data(data_files['soyisimler'])[1:]  # Skip header
    
    # Read program and university data as CSV
    program_listesi = []
    with open(data_files['lisans_programlari'], 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header line
        for row in reader:
            if row and len(row) >= 2:
                program = row[1].strip()
                if program:
                    program_listesi.append(program)
    
    universite_listesi = []
    with open(data_files['uni_listesi'], 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header line
        for row in reader:
            if row and len(row) >= 2:
                universite = row[1].strip()
                if universite:
                    universite_listesi.append(universite)
    
    # Check loaded data
    if not erkek_isimleri or not kadin_isimleri or not soyisimler:
        print("Names or surnames could not be loaded. Check file paths.")
        return
    
    if not program_listesi:
        print("Program data could not be loaded. Check file paths.")
        return
    
    if not universite_listesi:
        print("University data could not be loaded. Check file paths.")
        return
    
    # Get the number of samples to generate from the user
    while True:
        try:
            ornekler_sayisi = int(input("How many samples do you want to generate? "))
            if ornekler_sayisi <= 0:
                print("Please enter a positive number.")
                continue
            break
        except ValueError:
            print("Please enter a valid number.")
    
    # Create samples
    sentences = []
    labels = []
    for _ in range(ornekler_sayisi):
        sentence, label = create_sentence_and_labels()
        sentences.append(sentence)
        labels.append(label)
    
    # Let's offer the user data saving options
    while True:
        secim = input("How should the data be saved?\n"
                     "1: All in one folder\n"
                     "2: Split as Train/Validation/Test\n"
                     "Your choice (1/2): ")
        if secim in ['1', '2']:
            break
        print("Invalid choice, try again.")
    
    if secim == '1':
        # Save all data in a single folder
        save_to_files(sentences, labels)
    else:
        # Split as Train/Validation/Test
        train_orani = 0.7
        val_orani = 0.15
        
        print("Dataset split ratios:")
        print(f"  Train: {train_orani:.1%}")
        print(f"  Validation: {val_orani:.1%}")
        print(f"  Test: {1-train_orani-val_orani:.1%}")
        
        # Do they want to change the ratios?
        degistir = input("Do you want to change these ratios? (Y/N): ").strip().lower()
        if degistir == 'y': # Changed 'e' to 'y' for English convention
            while True:
                try:
                    train_orani = float(input("Enter the Train ratio (e.g., 0.7): "))
                    val_orani = float(input("Enter the Validation ratio (e.g., 0.15): "))
                    if train_orani <= 0 or val_orani <= 0 or train_orani + val_orani >= 1:
                        print("Invalid ratios! The sum of Train + Validation must be less than 1.")
                        continue
                    break
                except ValueError:
                    print("Invalid value, please enter a decimal number between 0-1.")
        
        # Split and save the data
        split_and_save_data(sentences, labels, train_orani, val_orani)


if __name__ == "__main__":
    main()