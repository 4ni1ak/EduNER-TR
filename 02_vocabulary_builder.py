import os
import json
import re
from typing import Dict, Tuple, List


def build_vocab_and_label_map_from_directory(directory_path: str) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Creates vocabulary and label maps by reading sentence_0.txt and label_0.txt files in the specified directory.
    
    Args:
        directory_path (str): Path to the directory containing sentence_0.txt and label_0.txt files
        
    Returns:
        Tuple[Dict[str, int], Dict[str, int]]: Vocabulary and label map dictionaries
    """
    # Create file paths - new format
    sentences_file = os.path.join(directory_path, "sentence_0.txt")
    labels_file = os.path.join(directory_path, "label_0.txt")
    
    # Check if files exist
    if not os.path.exists(sentences_file) or not os.path.exists(labels_file):
        raise FileNotFoundError(f"sentence_0.txt or label_0.txt files not found in the {directory_path} directory.")
    
    # Create empty dictionaries for vocabulary and label map
    vocab = {"PAD": 0, "UNK": 1}
    label_map = {"O": 0}  # Add "O" label for undefined (non-label) situations
    
    # Read sentences and create vocabulary
    vocab_id = 2  # Start from 2 since 0 and 1 are assigned to PAD and UNK
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
    
    # Read labels and create label map
    label_id = 1  # Start from 1 since we gave 0 to the "O" label
    with open(labels_file, 'r', encoding='utf-8') as file:
        for line_number, line in enumerate(file, 1):
            line = line.strip()
            if not line:
                continue
            
            labels = line.split()
            for label in labels:
                if label and label not in label_map:
                    label_map[label] = label_id
                    label_id += 1
    
    return vocab, label_map


def process_all_directories(base_path: str, output_dir: str) -> None:
    """
    Processes data files in all subdirectories (validation/test/train) under the main directory,
    creates separate vocabulary and label maps for each, and also creates
    combined vocabulary and label maps for all data.
    
    Args:
        base_path (str): Path to the main directory containing subdirectories (e.g. '01_query/')
        output_dir (str): Directory where generated JSON files will be saved
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Combined data structures
    combined_vocab = {"PAD": 0, "UNK": 1}
    combined_label_map = {"O": 0}  # Add "O" label for undefined situations
    
    vocab_id = 2
    label_id = 1  # Start from 1 since we gave 0 to the "O" label
    
    # Subdirectories to process
    subdirs = ["train", "validation", "test"]
    
    for subdir in subdirs:
        dir_path = os.path.join(base_path, subdir)
        if os.path.isdir(dir_path):
            try:
                # Create vocabulary and label map for subdirectory
                vocab, label_map = build_vocab_and_label_map_from_directory(dir_path)
                
                # Save subdirectory-specific files
                save_json(vocab, os.path.join(output_dir, f"{subdir}_vocab.json"))
                save_json(label_map, os.path.join(output_dir, f"{subdir}_labels.json"))
                
                print(f"Files created for {subdir} directory.")
                print(f"- Vocabulary size: {len(vocab)}")
                print(f"- Label map size: {len(label_map)}")
                
                # Add words to combined vocabulary
                for word in vocab:
                    if word not in combined_vocab and word not in ["PAD", "UNK"]:
                        combined_vocab[word] = vocab_id
                        vocab_id += 1
                
                # Add labels to combined map
                for label in label_map:
                    if label not in combined_label_map:
                        combined_label_map[label] = label_id
                        label_id += 1
            except FileNotFoundError as e:
                print(f"Warning: {e}")
    
    # Save combined files
    save_json(combined_vocab, os.path.join(output_dir, "combined_vocab.json"))
    save_json(combined_label_map, os.path.join(output_dir, "combined_labels.json"))
    
    print("\nCombined files created:")
    print(f"- Combined vocabulary size: {len(combined_vocab)}")
    print(f"- Combined label map size: {len(combined_label_map)}")


def save_json(data: Dict, file_path: str) -> None:
    """
    Saves data in JSON format to a file.
    
    Args:
        data (Dict): Data to be saved
        file_path (str): File path
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def check_sequence_alignment(directory_path: str) -> None:
    """
    Checks whether the number of lines in sentence_0.txt and label_0.txt files in the
    specified directory match, and whether the word count in each line matches.
    
    Args:
        directory_path (str): Path to directory containing sentence_0.txt and label_0.txt files
    """
    sentences_file = os.path.join(directory_path, "sentence_0.txt")
    labels_file = os.path.join(directory_path, "label_0.txt")
    
    if not os.path.exists(sentences_file) or not os.path.exists(labels_file):
        print(f"Files not found: {sentences_file} or {labels_file}")
        return
    
    with open(sentences_file, 'r', encoding='utf-8') as s_file, open(labels_file, 'r', encoding='utf-8') as l_file:
        sentence_lines = s_file.readlines()
        label_lines = l_file.readlines()
        
        if len(sentence_lines) != len(label_lines):
            print(f"Warning: Line counts don't match! Sentences: {len(sentence_lines)}, Labels: {len(label_lines)}")
            return
        
        for line_idx, (sentence, label) in enumerate(zip(sentence_lines, label_lines), 1):
            sentence_words = sentence.strip().split()
            label_words = label.strip().split()
            
            if len(sentence_words) != len(label_words):
                print(f"Warning: Word counts don't match in line {line_idx}!")
                print(f"  Sentence ({len(sentence_words)} words): {sentence.strip()}")
                print(f"  Label ({len(label_words)} labels): {label.strip()}")
    
    print("Check completed.")


if __name__ == "__main__":
    # Configure parameters
    base_directory = "01_query/"  # Main directory (containing train, validation, test subdirectories)
    output_directory = "02_vocabulary_files/"  # Directory where output files will be saved
    
    # Check data correctness before processing
    for subdir in ["train", "validation", "test"]:
        print(f"Correctness check for {subdir} directory:")
        check_sequence_alignment(os.path.join(base_directory, subdir))
        print()
    
    # Process all directories and create files
    process_all_directories(base_directory, output_directory) 