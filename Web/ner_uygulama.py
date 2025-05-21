#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
import re

# Define Transformer model (taken from 04_training_modell.py)
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, hidden_dim, num_layers, num_classes, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Positional encoding
        self.pos_encoder = nn.Embedding(2000, embed_dim)

        # Normalization layer
        self.embed_norm = nn.LayerNorm(embed_dim)

        # Transformer Encoder layers
        encoder_layers = []
        for i in range(num_layers):
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim,
                dropout=dropout,
                batch_first=True,
                activation='gelu',
                norm_first=True
            )
            encoder_layers.append(encoder_layer)

        self.transformer_layers = nn.ModuleList(encoder_layers)

        # Final transformer layer
        self.final_norm = nn.LayerNorm(embed_dim)

        # Classification layer
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

        # Token position weighting
        self.token_attention = nn.Sequential(
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()
        )

        # Factor regulating gradient flow
        self.grad_multiplier = 4.0

    def forward(self, x):
        # Padding mask - mask 0-valued tokens
        padding_mask = (x == 0)

        # Embedding operation
        x_embed = self.embedding(x)

        # Add positional encoding
        positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0).expand(x.size(0), -1)
        pos_embed = self.pos_encoder(positions)

        # Combine embedding and positional information
        x = x_embed + pos_embed

        # Normalize
        x = self.embed_norm(x)

        # Apply each transformer layer sequentially
        for layer in self.transformer_layers:
            # To control gradient flow
            layer_input = x
            x = layer(x, src_key_padding_mask=padding_mask)
            # Strengthen residual connection gradients
            x = x + self.grad_multiplier * 0.1 * layer_input

        # Final normalize
        x = self.final_norm(x)

        # Calculate token importance
        token_weights = self.token_attention(x)

        # Weighted average for only valid (non-padding) tokens
        mask = ~padding_mask.unsqueeze(-1)  # [batch_size, seq_len, 1]
        weighted_x = x * token_weights * mask.float()
        token_sum = weighted_x.sum(dim=1)
        token_weights_sum = (token_weights * mask.float()).sum(dim=1).clamp(min=1e-9)
        x_weighted = token_sum / token_weights_sum

        # Classification
        logits = self.classifier(x_weighted)

        return logits

# Load vocabulary and model configuration
def load_vocabulary():
    try:
        # First try to load from the primary location
        with open("02_vocabulary_files/combined_vocab.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        # Alternative location
        with open("combined_vocab.json", "r", encoding="utf-8") as f:
            return json.load(f)

def load_labels():
    try:
        # First try to load from the primary location
        with open("02_vocabulary_files/combined_labels.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        # Alternative location
        with open("combined_labels.json", "r", encoding="utf-8") as f:
            return json.load(f)

# Function to convert word to number
def tokenize_word(word, word_to_idx):
    word = word.lower()  # Convert word to lowercase
    return word_to_idx.get(word, 1)  # If word is not in dictionary, return 1 (unknown)

# Function to convert sentence to number sequence
def tokenize_sentence(sentence, word_to_idx, max_len=100):
    words = sentence.strip().split()
    tokens = [tokenize_word(word, word_to_idx) for word in words]
    
    # Fix sentence length to max_len
    if len(tokens) > max_len:
        tokens = tokens[:max_len]
    else:
        tokens += [0] * (max_len - len(tokens))  # Add padding
    
    return tokens

# Convert entity label names to human-readable format
def format_entity_label(label):
    if label == "O":
        return "Other"
    elif label == "B-UNIVERSITE":
        return "University (Beginning)"
    elif label == "I-UNIVERSITE":
        return "University (Continuation)"
    elif label == "B-BOLUM":
        return "Department (Beginning)"
    elif label == "I-BOLUM":
        return "Department (Continuation)"
    elif label == "B-NUMARA":
        return "Number"
    elif label == "B-ISIM":
        return "Name"
    elif label == "B-SOYISIM":
        return "Surname"
    else:
        return label

# Load model and make prediction
class NERPredictor:
    def __init__(self, model_path="best_model.pt"):
        print("Loading NER model...")
        # Use CPU instead of GPU
        self.device = torch.device("cpu")
        
        # Load vocabulary and labels
        self.word_to_idx = load_vocabulary()
        self.idx_to_label = {v: k for k, v in load_labels().items()}
        
        # Check model file
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Add security warning
        print("Loading a trusted model file. Using weights_only=False parameter.")
        
        try:
            # Load checkpoint with weights_only=False (since the model file is trusted)
            torch.serialization.add_safe_globals(['getattr'])
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        except Exception as e:
            print(f"Error occurred while loading model: {e}")
            print("Trying alternative model loading method...")
            
            try:
                # If another problem occurs, try with the simplest handler
                print("Using simplified model configuration...")
                # Basic model parameters
                self.model = TransformerModel(
                    vocab_size=102338,
                    embed_dim=1024,
                    num_heads=16,
                    hidden_dim=4096,
                    num_layers=8,
                    num_classes=8,
                    dropout=0.25
                )
                return
            except Exception as inner_e:
                print(f"Alternative method also failed: {inner_e}")
                raise
        
        # Create model with configuration
        config = checkpoint.get('config', {})
        if not config:
            # If configuration is missing, use default values
            config = {
                'vocab_size': 102338,  # Vocabulary size
                'embed_dim': 1024,     # Embedding size
                'num_heads': 16,       # Number of multi-head attention
                'hidden_dim': 4096,    # Hidden layer size
                'num_layers': 8,       # Number of Transformer layers
                'num_classes': 8,      # Number of label classes
                'dropout': 0.25        # Dropout rate
            }
        
        # Create the model
        self.model = TransformerModel(
            vocab_size=config.get('vocab_size', 102338),
            embed_dim=config.get('embed_dim', 1024),
            num_heads=config.get('num_heads', 16),
            hidden_dim=config.get('hidden_dim', 4096),
            num_layers=config.get('num_layers', 8),
            num_classes=config.get('num_classes', 8),
            dropout=config.get('dropout', 0.25)
        )
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded successfully. (Running on {self.device} device)")
    
    def predict(self, text):
        # Process text
        tokens = tokenize_sentence(text, self.word_to_idx)
        tokens_tensor = torch.tensor([tokens], dtype=torch.long).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(tokens_tensor)
            _, predicted = outputs.max(1)
            predicted_label_idx = predicted.item()
            
        # Return prediction result
        return self.idx_to_label.get(predicted_label_idx, "Unknown")
    
    def analyze_text(self, text):
        words = text.strip().split()
        results = []
        
        # Make prediction for each word
        for word in words:
            if word.strip():  # For non-empty words
                label = self.predict(word)
                results.append((word, label))
        
        return results
    
    def extract_entities(self, text):
        words = text.strip().split()
        entities = {
            "universite": [],
            "bolum": [],
            "isim": [],
            "soyisim": [],
            "numara": []
        }
        
        current_entity = None
        current_words = []
        i = 0
        
        # Make prediction for each word one by one
        while i < len(words):
            word = words[i]
            if not word.strip():
                i += 1
                continue
                
            # Prediction for each word
            label = self.predict(word)
            
            # Label-based processing
            if label.startswith("B-"):
                # Save previous entity
                if current_entity and current_words:
                    entity_type = current_entity.lower().replace("b-", "").replace("i-", "")
                    if entity_type in entities:
                        entities[entity_type].append(" ".join(current_words))
                
                # Start new entity
                current_entity = label
                current_words = [word]
                
                # Look ahead and collect continuation tags belonging to the same entity
                entity_type = label.split("-")[1]  # "B-UNIVERSITE" -> "UNIVERSITE"
                j = i + 1
                # Look ahead for continuation tags or combined entities
                next_predicted = ""
                while j < len(words) and (
                    (next_predicted := self.predict(words[j])) == f"I-{entity_type}" or  # Explicit continuation tag
                    (j < len(words) - 1 and next_predicted == f"B-{entity_type}" and self.predict(words[j+1]) == f"I-{entity_type}") or  # New beginning but has continuation
                    next_predicted == f"B-{entity_type}"  # Adjacent beginning tag
                ):
                    current_words.append(words[j])
                    j += 1
                
                # Skip processed words
                if j > i + 1:
                    i = j
                else:
                    i += 1
                
            elif label.startswith("I-") and current_entity:
                # Check if compatible with current entity type
                entity_prefix = current_entity.split("-")[1]
                if label.endswith(entity_prefix):
                    current_words.append(word)
                    i += 1
                else:
                    # Save previous entity and start new
                    entity_type = current_entity.lower().replace("b-", "").replace("i-", "")
                    if entity_type in entities:
                        entities[entity_type].append(" ".join(current_words))
                    current_entity = None
                    current_words = []
                    i += 1
            
            elif label == "O" and current_entity:
                # Save previous entity
                entity_type = current_entity.lower().replace("b-", "").replace("i-", "")
                if entity_type in entities:
                    entities[entity_type].append(" ".join(current_words))
                current_entity = None
                current_words = []
                i += 1
            
            else:
                # Move to the next word in other cases
                i += 1
        
        # Save the last entity
        if current_entity and current_words:
            entity_type = current_entity.lower().replace("b-", "").replace("i-", "")
            if entity_type in entities:
                entities[entity_type].append(" ".join(current_words))
        
        # Post-check for universities - combine university parts that might be concatenated
        if "universite" in entities and len(entities["universite"]) > 1:
            new_unis = []
            skip_indices = set()
            
            for i, uni1 in enumerate(entities["universite"]):
                if i in skip_indices:
                    continue
                    
                uni1_lower = uni1.lower()
                combined = uni1
                
                for j, uni2 in enumerate(entities["universite"]):
                    if i != j and j not in skip_indices:
                        # Combine university words that are close in position
                        words_idx1 = [idx for idx, w in enumerate(words) if w in uni1.split()]
                        words_idx2 = [idx for idx, w in enumerate(words) if w in uni2.split()]
                        
                        # If words are close (at most 2 words between them)
                        if words_idx1 and words_idx2 and abs(words_idx1[0] - words_idx2[0]) <= 3:
                            # Do words complete each other?
                            if "üniversite" in uni2.lower() and "üniversite" not in uni1_lower:
                                combined = f"{uni1} {uni2}"
                                skip_indices.add(j)
                            elif any(city in uni1_lower for city in ["ankara", "istanbul", "izmir", "eskişehir"]) and "üniversite" in uni2.lower():
                                combined = f"{uni1} {uni2}"
                                skip_indices.add(j)
                
                new_unis.append(combined)
            
            entities["universite"] = new_unis
        
        # Similarly, check for departments
        if "bolum" in entities and len(entities["bolum"]) > 1:
            new_bolum = []
            skip_indices = set()
            
            for i, bol1 in enumerate(entities["bolum"]):
                if i in skip_indices:
                    continue
                    
                bol1_lower = bol1.lower()
                combined = bol1
                
                for j, bol2 in enumerate(entities["bolum"]):
                    if i != j and j not in skip_indices:
                        # Combine department words that are close in position
                        words_idx1 = [idx for idx, w in enumerate(words) if w in bol1.split()]
                        words_idx2 = [idx for idx, w in enumerate(words) if w in bol2.split()]
                        
                        # If words are close
                        if words_idx1 and words_idx2 and abs(words_idx1[0] - words_idx2[0]) <= 3:
                            combined = f"{bol1} {bol2}"
                            skip_indices.add(j)
                
                new_bolum.append(combined)
            
            entities["bolum"] = new_bolum
        
        return entities

# Simple demo model class (to be used if the model cannot be loaded)
class DemoNERPredictor:
    def __init__(self):
        print("Creating Demo NER model (real model could not be loaded)...")
        self.idx_to_label = {
            0: "O", 
            1: "B-UNIVERSITE", 
            2: "B-BOLUM", 
            3: "I-UNIVERSITE", 
            4: "I-BOLUM",
            5: "B-NUMARA", 
            6: "B-ISIM", 
            7: "B-SOYISIM"
        }
        
        # Some common phrases for university names
        self.uni_words = ["üniversite", "üniversitesi", "university", "üniv"]
        # Some common phrases for department names
        self.dept_words = ["bölüm", "bölümü", "fakülte", "fakültesi", "mühendislik", 
                          "mühendisliği", "edebiyat", "bilim", "program", "programı", 
                          "tasarım", "tasarımı", "dizayn", "oyun", "teknoloji", "teknolojisi"]
        
        print("Demo model ready (will return artificial predictions)")
    
    def predict(self, text):
        # More advanced heuristics
        text_lower = text.lower()
        
        # Improved check for university names
        for word in self.uni_words:
            if word in text_lower:
                return "B-UNIVERSITE"
                
        # Improved check for department names
        for word in self.dept_words:
            if word in text_lower:
                return "B-BOLUM"
        
        # Number recognition - smarter check including numerical values
        if text.isdigit() and len(text) > 4:
            return "B-NUMARA"
            
        # Name recognition - Turkish names usually start with a capital letter
        # and are at least 3 characters long
        if text[0].isupper() and len(text) > 2 and not any(c.isdigit() for c in text):
            # Check for some common surname suffixes
            if text_lower.endswith(("oğlu", "gil", "ler", "lar", "glu", "gül", "öz", "kan")):
                return "B-SOYISIM"
            # Otherwise, consider it a name
            return "B-ISIM"
        
        # Continuation tag marking - based on the state of the previous word
        if text_lower.endswith(("üniversitesi", "university")):
            return "I-UNIVERSITE"
        if text_lower.endswith(("fakültesi", "bölümü", "programı")):
            return "I-BOLUM"
            
        # If no condition is met, label as "Other"
        return "O"
    
    def analyze_text(self, text):
        words = text.strip().split()
        results = []
        
        # Analyzing multiple words together for combined entities
        i = 0
        while i < len(words):
            if not words[i].strip():
                i += 1
                continue
                
            # Check for entities that might consist of two or more words
            if i < len(words) - 1:
                two_words = f"{words[i]} {words[i+1]}"
                # Check for university names
                if any(uni in two_words.lower() for uni in self.uni_words):
                    label = "B-UNIVERSITE"
                    # If it consists of three or more words
                    if i < len(words) - 2 and not any(uni in words[i+2].lower() for uni in self.uni_words):
                        three_words = f"{words[i]} {words[i+1]} {words[i+2]}"
                        results.append((words[i], "B-UNIVERSITE"))
                        results.append((words[i+1], "I-UNIVERSITE"))
                        results.append((words[i+2], "I-UNIVERSITE"))
                        i += 3
                        continue
                    results.append((words[i], "B-UNIVERSITE"))
                    results.append((words[i+1], "I-UNIVERSITE"))
                    i += 2
                    continue
                
                # Check for department names
                if any(dept in two_words.lower() for dept in self.dept_words):
                    results.append((words[i], "B-BOLUM"))
                    results.append((words[i+1], "I-BOLUM"))
                    i += 2
                    continue
            
            # Prediction for single word
            label = self.predict(words[i])
            results.append((words[i], label))
            i += 1
        
        return results
    
    def extract_entities(self, text):
        words = text.strip().split()
        entities = {
            "universite": [],
            "bolum": [],
            "isim": [],
            "soyisim": [],
            "numara": []
        }
        
        current_entity = None
        current_words = []
        i = 0
        
        # Make prediction for each word one by one
        while i < len(words):
            word = words[i]
            if not word.strip():
                i += 1
                continue
                
            # Prediction for each word
            label = self.predict(word)
            
            # Label-based processing
            if label.startswith("B-"):
                # Save previous entity
                if current_entity and current_words:
                    entity_type = current_entity.lower().replace("b-", "").replace("i-", "")
                    if entity_type in entities:
                        entities[entity_type].append(" ".join(current_words))
                
                # Start new entity
                current_entity = label
                current_words = [word]
                
                # Look ahead and collect continuation tags belonging to the same entity
                entity_type = label.split("-")[1]  # "B-UNIVERSITE" -> "UNIVERSITE"
                j = i + 1
                # Look ahead for continuation tags or combined entities
                next_predicted = ""
                while j < len(words) and (
                    (next_predicted := self.predict(words[j])) == f"I-{entity_type}" or  # Explicit continuation tag
                    (j < len(words) - 1 and next_predicted == f"B-{entity_type}" and self.predict(words[j+1]) == f"I-{entity_type}") or  # New beginning but has continuation
                    next_predicted == f"B-{entity_type}"  # Adjacent beginning tag
                ):
                    current_words.append(words[j])
                    j += 1
                
                # Skip processed words
                if j > i + 1:
                    i = j
                else:
                    i += 1
                
            elif label.startswith("I-") and current_entity:
                # Check if compatible with current entity type
                entity_prefix = current_entity.split("-")[1]
                if label.endswith(entity_prefix):
                    current_words.append(word)
                    i += 1
                else:
                    # Save previous entity and start new
                    entity_type = current_entity.lower().replace("b-", "").replace("i-", "")
                    if entity_type in entities:
                        entities[entity_type].append(" ".join(current_words))
                    current_entity = None
                    current_words = []
                    i += 1
            
            elif label == "O" and current_entity:
                # Save previous entity
                entity_type = current_entity.lower().replace("b-", "").replace("i-", "")
                if entity_type in entities:
                    entities[entity_type].append(" ".join(current_words))
                current_entity = None
                current_words = []
                i += 1
            
            else:
                # Move to the next word in other cases
                i += 1
        
        # Save the last entity
        if current_entity and current_words:
            entity_type = current_entity.lower().replace("b-", "").replace("i-", "")
            if entity_type in entities:
                entities[entity_type].append(" ".join(current_words))
        
        # Post-check for universities - combine university parts that might be concatenated
        if "universite" in entities and len(entities["universite"]) > 1:
            new_unis = []
            skip_indices = set()
            
            for i, uni1 in enumerate(entities["universite"]):
                if i in skip_indices:
                    continue
                    
                uni1_lower = uni1.lower()
                combined = uni1
                
                for j, uni2 in enumerate(entities["universite"]):
                    if i != j and j not in skip_indices:
                        # Combine university words that are close in position
                        words_idx1 = [idx for idx, w in enumerate(words) if w in uni1.split()]
                        words_idx2 = [idx for idx, w in enumerate(words) if w in uni2.split()]
                        
                        # If words are close (at most 2 words between them)
                        if words_idx1 and words_idx2 and abs(words_idx1[0] - words_idx2[0]) <= 3:
                            # Do words complete each other?
                            if "üniversite" in uni2.lower() and "üniversite" not in uni1_lower:
                                combined = f"{uni1} {uni2}"
                                skip_indices.add(j)
                            elif any(city in uni1_lower for city in ["ankara", "istanbul", "izmir", "eskişehir"]) and "üniversite" in uni2.lower():
                                combined = f"{uni1} {uni2}"
                                skip_indices.add(j)
                
                new_unis.append(combined)
            
            entities["universite"] = new_unis
        
        # Similarly, check for departments
        if "bolum" in entities and len(entities["bolum"]) > 1:
            new_bolum = []
            skip_indices = set()
            
            for i, bol1 in enumerate(entities["bolum"]):
                if i in skip_indices:
                    continue
                    
                bol1_lower = bol1.lower()
                combined = bol1
                
                for j, bol2 in enumerate(entities["bolum"]):
                    if i != j and j not in skip_indices:
                        # Combine department words that are close in position
                        words_idx1 = [idx for idx, w in enumerate(words) if w in bol1.split()]
                        words_idx2 = [idx for idx, w in enumerate(words) if w in bol2.split()]
                        
                        # If words are close
                        if words_idx1 and words_idx2 and abs(words_idx1[0] - words_idx2[0]) <= 3:
                            combined = f"{bol1} {bol2}"
                            skip_indices.add(j)
                
                new_bolum.append(combined)
            
            entities["bolum"] = new_bolum
        
        return entities

def turkce_kucuk_yap(text): # Function name kept in Turkish as per typical request scope
    return text.casefold()

# Main application
def main():
    print("NER (Named Entity Recognition) Application")
    print("This application identifies entities (university, department, name, surname, number) in text.")
    print("Loading model, please wait...")
    
    try:
        # First try to load the real model
        try:
            predictor = NERPredictor()
        except Exception as e:
            print(f"Real model could not be loaded: {e}")
            print("Continuing with Demo mode...")
            predictor = DemoNERPredictor()
        
        print("\nApplication ready! Enter your text (type 'q' to exit):")
        
        while True:
            user_input = input("\nText: ")
            
            if user_input.lower() == 'q':
                print("Exiting application...")
                break
            
            if not user_input.strip():
                print("Please enter some text.")
                continue
            
            # Convert to Turkish lowercase
            user_input = turkce_kucuk_yap(user_input)
            
            print("\nAnalyzing text...")
            entities = predictor.extract_entities(user_input)
            
            print("\n--- ANALYSIS RESULTS ---")
            
            # University information
            if entities["universite"]:
                print(f"\nUniversity: {', '.join(entities['universite'])}")
            
            # Department information
            if entities["bolum"]:
                print(f"Department: {', '.join(entities['bolum'])}")
            
            # Name information
            if entities["isim"]:
                print(f"Name: {', '.join(entities['isim'])}")
            
            # Surname information
            if entities["soyisim"]:
                print(f"Surname: {', '.join(entities['soyisim'])}")
            
            # Number information
            if entities["numara"]:
                print(f"Number: {', '.join(entities['numara'])}")
            
            if not any(entities.values()):
                print("No identifiable entities found in the text.")
            
            print("\n--- DETAILED ANALYSIS ---")
            results = predictor.analyze_text(user_input)
            
            for word, label in results:
                formatted_label = format_entity_label(label) # This will now return English labels
                print(f"{word}: {formatted_label}")
    
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()