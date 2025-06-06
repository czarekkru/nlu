# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *

import torch
import torch.utils.data as data
import json
import os
from model import BERT_IAS
from functions import train_model
from sklearn.model_selection import train_test_split
from collections import Counter
from transformers import BertTokenizer

def load_data(path):
    """
    Load data from JSON file.
    
    Args:
        path (str): Path to the JSON file
        
    Returns:
        list: List of dictionaries containing the data
    """
    with open(path) as f:
        dataset = json.load(f)
    return dataset

class Lang:
    """
    Language class to handle vocabulary and mappings.
    """
    def __init__(self, intents, slots):
        self.slot2id = self.lab2id(slots)
        self.intent2id = self.lab2id(intents, pad=False)
        self.id2slot = {v:k for k, v in self.slot2id.items()}
        self.id2intent = {v:k for k, v in self.intent2id.items()}
    
    def lab2id(self, elements, pad=True):
        vocab = {}
        if pad:
            vocab['pad'] = 0
        for elem in elements:
            vocab[elem] = len(vocab)
        return vocab

class IntentsAndSlots(data.Dataset):
    """
    Dataset class for intent classification and slot filling with BERT tokenization.
    """
    def __init__(self, dataset, lang, tokenizer, max_len=128):
        self.utterances = []
        self.intents = []
        self.slots = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.lang = lang
        
        for x in dataset:
            self.utterances.append(x['utterance'])
            self.slots.append(x['slots'])
            self.intents.append(x['intent'])
        
        self.slot_ids = self.mapping_seq(self.slots, lang.slot2id)
        self.intent_ids = self.mapping_lab(self.intents, lang.intent2id)
    
    def __len__(self):
        return len(self.utterances)
    
    def __getitem__(self, idx):
        utterance = self.utterances[idx]
        slots = self.slots[idx]
        intent = self.intent_ids[idx]
        
        # Tokenize with BERT
        encoding = self.tokenizer(
            utterance,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Get word tokens for slot alignment
        words = utterance.split()
        
        # Create slot labels aligned with BERT tokens
        slot_labels = self.align_slot_labels(words, slots, encoding)
        
        # Create attention mask for slots
        slot_mask = (encoding['input_ids'][0] != self.tokenizer.pad_token_id).long()
        
        sample = {
            'input_ids': encoding['input_ids'][0],
            'attention_mask': encoding['attention_mask'][0],
            'token_type_ids': encoding['token_type_ids'][0],
            'slot_labels': slot_labels,
            'intent_labels': intent,
            'slot_mask': slot_mask,
            'words': words
        }
        return sample
    
    def mapping_lab(self, data, mapper):
        return [mapper[x] if x in mapper else mapper['pad'] for x in data]
    
    def mapping_seq(self, data, mapper):
        res = []
        for seq in data:
            tmp_seq = []
            for x in seq.split():
                if x in mapper:
                    tmp_seq.append(mapper[x])
                else:
                    tmp_seq.append(mapper['pad'])
            res.append(tmp_seq)
        return res
    
    def align_slot_labels(self, words, slots, encoding):
        """
        Align slot labels with BERT tokenization.
        """
        slot_labels = [-100] * self.max_len  # -100 is ignored in loss computation
        
        # Get BERT tokens
        bert_tokens = self.tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])
        
        # Map slots to words
        word_slots = dict(zip(words, slots.split()))
        
        # Align slots with BERT tokens
        word_idx = 0
        for token_idx, token in enumerate(bert_tokens):
            if token == '[CLS]' or token == '[SEP]' or token == '[PAD]':
                continue
                
            if token.startswith('##'):
                # This is a sub-token, use the same label as the previous token
                if word_idx > 0:
                    slot_labels[token_idx] = self.lang.slot2id[word_slots[words[word_idx-1]]]
            else:
                # This is a new word
                if word_idx < len(words):
                    slot_labels[token_idx] = self.lang.slot2id[word_slots[words[word_idx]]]
                    word_idx += 1
        
        return torch.LongTensor(slot_labels)

def collate_fn(data):
    """
    Collate function for DataLoader.
    """
    batch = {}
    for key in data[0].keys():
        if isinstance(data[0][key], torch.Tensor):
            batch[key] = torch.stack([d[key] for d in data])
        else:
            # Convert lists to tensors
            if key == 'intent_labels':
                batch[key] = torch.tensor([d[key] for d in data])
            else:
                batch[key] = [d[key] for d in data]
    return batch

if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dataset_path = os.path.normpath(os.path.join(base_path, "dataset", "ATIS"))
    train_raw = load_data(os.path.join(dataset_path, "train.json"))
    test_raw = load_data(os.path.join(dataset_path, "test.json"))
    
    # Create dev set
    portion = 0.10
    intents = [x['intent'] for x in train_raw]
    count_y = Counter(intents)
    
    labels = []
    inputs = []
    mini_train = []
    
    for id_y, y in enumerate(intents):
        if count_y[y] > 1:
            inputs.append(train_raw[id_y])
            labels.append(y)
        else:
            mini_train.append(train_raw[id_y])
    
    X_train, X_dev, y_train, y_dev = train_test_split(
        inputs, labels, test_size=portion, 
        random_state=42, 
        shuffle=True,
        stratify=labels
    )
    X_train.extend(mini_train)
    train_raw = X_train
    dev_raw = X_dev
    
    # Create vocabulary
    corpus = train_raw + dev_raw + test_raw
    slots = set(sum([line['slots'].split() for line in corpus],[]))
    intents = set([line['intent'] for line in corpus])
    
    lang = Lang(intents, slots)
    
    # Initialize BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Create datasets
    train_dataset = IntentsAndSlots(train_raw, lang, tokenizer)
    dev_dataset = IntentsAndSlots(dev_raw, lang, tokenizer)
    test_dataset = IntentsAndSlots(test_raw, lang, tokenizer)
    
    # Create dataloaders
    train_loader = data.DataLoader(
        train_dataset, 
        batch_size=8, # 16 changed
        collate_fn=collate_fn, 
        shuffle=True
    )
    dev_loader = data.DataLoader(
        dev_dataset, 
        batch_size=8, # 16 changed
        collate_fn=collate_fn
    )
    test_loader = data.DataLoader(
        test_dataset, 
        batch_size=8, # 16 changed
        collate_fn=collate_fn
    )
    
    # Model hyperparameters
    out_slot = len(lang.slot2id)
    out_int = len(lang.intent2id)
    
    # Create and initialize model
    model = BERT_IAS(
        out_slot=out_slot,
        out_int=out_int,
        bert_model_name='bert-base-uncased',
        dropout=0.5 # 0.1 changed
    ).to(device)
    
    # Train model
    best_model, test_metrics = train_model(
        model=model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        test_loader=test_loader,
        id2slot=lang.id2slot,
        id2intent=lang.id2intent,
        learning_rate=2e-5, # 2e-5 changed
        n_epochs=3, # 10 changed
        patience=3 # 3 changed
    )
    
    # Print final results
    print("\nFinal Results:")
    print(f"Slot F1: {test_metrics[0]['total']['f']:.4f}")
    print(f"Intent Accuracy: {test_metrics[1]['accuracy']:.4f}")
    
    # Save model
    os.makedirs("model_bin", exist_ok=True)
    torch.save({
        'model_state_dict': best_model.state_dict(),
        'lang': lang
    }, "model_bin/bert_ias_model.pt")
