import os
import json
import numpy as np
from typing import Dict, Tuple, List, Optional


def load_vocabulary_and_labels(vocab_path: str, labels_path: str) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Load vocabulary and label mappings from JSON files.
    
    Args:
        vocab_path: Path to the vocabulary JSON file
        labels_path: Path to the labels JSON file
        
    Returns:
        Tuple containing vocabulary and label mappings (word/label -> id)
    
    Raises:
        FileNotFoundError: If either file doesn't exist
        json.JSONDecodeError: If files contain invalid JSON
    """
    try:
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        
        with open(labels_path, 'r', encoding='utf-8') as f:
            label_map = json.load(f)
        
        return vocab, label_map
    
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Vocabulary or label file not found: {e}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in vocabulary or label file: {e}")


def numerize_data(directory_path: str, vocab: Dict[str, int], label_map: Dict[str, int], 
                max_seq_length: Optional[int] = None) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Convert text sentences and labels from the specified directory into numerical IDs
    and pad/truncate to a fixed length.
    
    Args:
        directory_path: Path to directory containing sentence_0.txt and label_0.txt
        vocab: Vocabulary mapping (word -> id)
        label_map: Label mapping (label -> id)
        max_seq_length: Maximum sequence length. If None, will be calculated as
                        the length of the longest sentence + 12
        
    Returns:
        Tuple containing numerized sentences and labels (as lists of integer IDs)
    
    Raises:
        FileNotFoundError: If required files don't exist
    """
    # Define file paths
    sentences_file = os.path.join(directory_path, "sentence_0.txt")
    labels_file = os.path.join(directory_path, "label_0.txt")
    
    # Check if files exist
    if not os.path.exists(sentences_file) or not os.path.exists(labels_file):
        raise FileNotFoundError(f"sentence_0.txt or label_0.txt not found in {directory_path}")
    
    # Read sentences and labels
    with open(sentences_file, 'r', encoding='utf-8') as s_file, open(labels_file, 'r', encoding='utf-8') as l_file:
        sentence_lines = [line.strip() for line in s_file if line.strip()]
        label_lines = [line.strip() for line in l_file if line.strip()]
    
    # Calculate max sequence length if not provided
    if max_seq_length is None:
        max_sentence_length = max(len(line.split()) for line in sentence_lines) if sentence_lines else 0
        max_seq_length = max_sentence_length + 12
        print(f"Max sequence length: {max_seq_length} (longest sentence: {max_sentence_length} words + 12 padding)")
    
    # Numerize sentences and labels
    numeric_sentences = []
    numeric_labels = []
    
    for line_idx, (sentence, label_line) in enumerate(zip(sentence_lines, label_lines), 1):
        # Split into words and labels
        words = sentence.split()
        labels = label_line.split()
        
        # Check if words and labels have the same length
        if len(words) != len(labels):
            print(f"Warning: Mismatch in words ({len(words)}) and labels ({len(labels)}) at line {line_idx}")
            print(f"  Sentence: {sentence}")
            print(f"  Labels: {label_line}")
            # Use the minimum length to avoid index errors
            min_length = min(len(words), len(labels))
            words = words[:min_length]
            labels = labels[:min_length]
        
        # Convert words to IDs (use UNK for unknown words)
        numeric_sentence = [vocab.get(word, vocab.get("UNK", 1)) for word in words]
        
        # Convert labels to IDs (use O for unknown labels)
        numeric_label = [label_map.get(label, label_map.get("O", 0)) for label in labels]
        
        # Pad or truncate to max_seq_length
        if len(numeric_sentence) < max_seq_length:
            # Pad sentences with PAD token
            padding_length = max_seq_length - len(numeric_sentence)
            numeric_sentence.extend([vocab.get("PAD", 0)] * padding_length)
            numeric_label.extend([label_map.get("O", 0)] * padding_length)
        else:
            # Truncate if longer than max_seq_length
            numeric_sentence = numeric_sentence[:max_seq_length]
            numeric_label = numeric_label[:max_seq_length]
        
        numeric_sentences.append(numeric_sentence)
        numeric_labels.append(numeric_label)
    
    return numeric_sentences, numeric_labels


def save_numerized_data(numeric_sentences: List[List[int]], numeric_labels: List[List[int]], 
                       output_dir: str, prefix: str) -> None:
    """
    Save numerized data as NumPy arrays.
    
    Args:
        numeric_sentences: List of numerized sentences
        numeric_labels: List of numerized labels
        output_dir: Directory to save the output files
        prefix: Prefix for output files (e.g., "train", "validation", "test")
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to NumPy arrays
    sentences_array = np.array(numeric_sentences, dtype=np.int32)
    labels_array = np.array(numeric_labels, dtype=np.int32)
    
    # Save to files
    np.save(os.path.join(output_dir, f"{prefix}_sentences.npy"), sentences_array)
    np.save(os.path.join(output_dir, f"{prefix}_labels.npy"), labels_array)
    
    print(f"{prefix.title()} data saved:")
    print(f"- Sentences shape: {sentences_array.shape}")
    print(f"- Labels shape: {labels_array.shape}")


def process_query_data(base_path: str, vocab_dir: str, output_dir: str, 
                     max_seq_length: Optional[int] = None, 
                     use_combined: bool = True) -> None:
    """
    Process all data in the query directory structure, numerize it, and save it as NumPy arrays.
    
    Args:
        base_path: Base directory containing train/validation/test subdirectories
        vocab_dir: Directory containing vocabulary and label files
        output_dir: Directory to save the numerized data
        max_seq_length: Maximum sequence length (if None, calculated automatically)
        use_combined: Whether to use the combined vocabulary and label mappings
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Directories to process
    subdirs = ["train", "validation", "test"]
    
    # Find the longest sentence across all datasets (used if max_seq_length is None)
    if max_seq_length is None:
        longest_sentence = 0
        for subdir in subdirs:
            sentences_file = os.path.join(base_path, subdir, "sentence_0.txt")
            if os.path.exists(sentences_file):
                with open(sentences_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            longest_sentence = max(longest_sentence, len(line.split()))
        
        max_seq_length = longest_sentence + 12
        print(f"Automatically set max sequence length: {max_seq_length} (longest sentence: {longest_sentence} words + 12 padding)")
    
    # Load vocabularies and label maps
    if use_combined:
        # Use the combined vocabulary and label mapping for all datasets
        vocab_path = os.path.join(vocab_dir, "combined_vocab.json")
        labels_path = os.path.join(vocab_dir, "combined_labels.json")
        
        try:
            vocab, label_map = load_vocabulary_and_labels(vocab_path, labels_path)
            print(f"Loaded combined vocabulary and labels:")
            print(f"- Vocabulary size: {len(vocab)}")
            print(f"- Label map size: {len(label_map)}")
        except (FileNotFoundError, ValueError) as e:
            print(f"Error loading combined files: {e}")
            print("Falling back to individual vocabulary and label files.")
            use_combined = False
    
    # Process each subdirectory
    for subdir in subdirs:
        dir_path = os.path.join(base_path, subdir)
        if not os.path.isdir(dir_path):
            print(f"Warning: {dir_path} is not a directory. Skipping.")
            continue
        
        try:
            # Load specific vocabulary and labels if not using combined
            if not use_combined:
                vocab_path = os.path.join(vocab_dir, f"{subdir}_vocab.json")
                labels_path = os.path.join(vocab_dir, f"{subdir}_labels.json")
                vocab, label_map = load_vocabulary_and_labels(vocab_path, labels_path)
                print(f"Loaded {subdir} vocabulary and labels:")
                print(f"- Vocabulary size: {len(vocab)}")
                print(f"- Label map size: {len(label_map)}")
            
            # Numerize the data
            print(f"\nProcessing {subdir} data...")
            numeric_sentences, numeric_labels = numerize_data(
                dir_path, vocab, label_map, max_seq_length
            )
            
            # Save the numerized data
            save_numerized_data(numeric_sentences, numeric_labels, output_dir, subdir)
            
        except (FileNotFoundError, ValueError) as e:
            print(f"Error processing {subdir}: {e}")


def validate_query_data(directory_path: str) -> bool:
    """
    Validate that the query data is consistent (equal number of words and labels).
    
    Args:
        directory_path: Path to directory containing sentence_0.txt and label_0.txt
        
    Returns:
        True if validation passes, False otherwise
    """
    # Define file paths
    sentences_file = os.path.join(directory_path, "sentence_0.txt")
    labels_file = os.path.join(directory_path, "label_0.txt")
    
    # Check if files exist
    if not os.path.exists(sentences_file) or not os.path.exists(labels_file):
        print(f"Error: sentence_0.txt or label_0.txt not found in {directory_path}")
        return False
    
    # Read files
    try:
        with open(sentences_file, 'r', encoding='utf-8') as s_file, open(labels_file, 'r', encoding='utf-8') as l_file:
            sentence_lines = [line.strip() for line in s_file if line.strip()]
            label_lines = [line.strip() for line in l_file if line.strip()]
    except Exception as e:
        print(f"Error reading files: {e}")
        return False
    
    # Check if number of lines match
    if len(sentence_lines) != len(label_lines):
        print(f"Error: Number of sentences ({len(sentence_lines)}) doesn't match number of label lines ({len(label_lines)})")
        return False
    
    # Check if number of words matches number of labels for each line
    is_valid = True
    for line_idx, (sentence, label_line) in enumerate(zip(sentence_lines, label_lines), 1):
        words = sentence.split()
        labels = label_line.split()
        
        if len(words) != len(labels):
            print(f"Error at line {line_idx}: Words ({len(words)}) ≠ Labels ({len(labels)})")
            print(f"  Sentence: {sentence}")
            print(f"  Labels: {label_line}")
            is_valid = False
    
    if is_valid:
        print(f"Validation passed: {len(sentence_lines)} lines processed with matching word and label counts.")
    
    return is_valid


if __name__ == "__main__":
    # Configuration
    BASE_DIR = "01_query"                  # Base directory containing train/validation/test subdirectories
    VOCAB_DIR = "02_vocabulary_files"      # Directory containing vocabulary and label files
    OUTPUT_DIR = "03_numerized_data"       # Directory to save the numerized data
    
    # Validate data before processing
    print("Validating query data...")
    for subdir in ["train", "validation", "test"]:
        dir_path = os.path.join(BASE_DIR, subdir)
        if os.path.isdir(dir_path):
            print(f"\nValidating {subdir} data:")
            validate_query_data(dir_path)
    
    # Process data
    print("\nProcessing query data...")
    process_query_data(
        base_path=BASE_DIR,
        vocab_dir=VOCAB_DIR,
        output_dir=OUTPUT_DIR,
        max_seq_length=None,  # Auto-calculate
        use_combined=True     # Use combined vocabulary and labels
    ) 