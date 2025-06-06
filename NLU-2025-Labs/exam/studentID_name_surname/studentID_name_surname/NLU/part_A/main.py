# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *

import torch
import torch.utils.data as data
import json
import os
from model import ModelIAS
from functions import init_weights, train_model
from sklearn.model_selection import train_test_split
from collections import Counter

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
    def __init__(self, words, intents, slots, cutoff=0):
        self.word2id = self.w2id(words, cutoff=cutoff, unk=True)
        self.slot2id = self.lab2id(slots)
        self.intent2id = self.lab2id(intents, pad=False)
        self.id2word = {v:k for k, v in self.word2id.items()}
        self.id2slot = {v:k for k, v in self.slot2id.items()}
        self.id2intent = {v:k for k, v in self.intent2id.items()}
        
    def w2id(self, elements, cutoff=None, unk=True):
        vocab = {'pad': 0}
        if unk:
            vocab['unk'] = len(vocab)
        count = Counter(elements)
        for k, v in count.items():
            if v > cutoff:
                vocab[k] = len(vocab)
        return vocab
    
    def lab2id(self, elements, pad=True):
        vocab = {}
        if pad:
            vocab['pad'] = 0
        for elem in elements:
            vocab[elem] = len(vocab)
        return vocab

class IntentsAndSlots(data.Dataset):
    """
    Dataset class for intent classification and slot filling.
    """
    def __init__(self, dataset, lang, unk='unk'):
        self.utterances = []
        self.intents = []
        self.slots = []
        self.unk = unk
        
        for x in dataset:
            self.utterances.append(x['utterance'])
            self.slots.append(x['slots'])
            self.intents.append(x['intent'])
        
        self.utt_ids = self.mapping_seq(self.utterances, lang.word2id)
        self.slot_ids = self.mapping_seq(self.slots, lang.slot2id)
        self.intent_ids = self.mapping_lab(self.intents, lang.intent2id)
    
    def __len__(self):
        return len(self.utterances)
    
    def __getitem__(self, idx):
        utt = torch.LongTensor(self.utt_ids[idx])
        slots = torch.LongTensor(self.slot_ids[idx])
        intent = self.intent_ids[idx]
        sample = {'utterance': utt, 'slots': slots, 'intent': intent}
        return sample
    
    def mapping_lab(self, data, mapper):
        return [mapper[x] if x in mapper else mapper[self.unk] for x in data]
    
    def mapping_seq(self, data, mapper):
        res = []
        for seq in data:
            tmp_seq = []
            for x in seq.split():
                if x in mapper:
                    tmp_seq.append(mapper[x])
                else:
                    tmp_seq.append(mapper[self.unk])
            res.append(tmp_seq)
        return res

def collate_fn(data):
    """
    Collate function for DataLoader.
    """
    def merge(sequences):
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths)==0 else max(lengths)
        padded_seqs = torch.LongTensor(len(sequences),max_len).fill_(0)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq
        padded_seqs = padded_seqs.detach()
        return padded_seqs, lengths
    
    data.sort(key=lambda x: len(x['utterance']), reverse=True)
    new_item = {}
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]
    
    src_utt, _ = merge(new_item['utterance'])
    y_slots, y_lengths = merge(new_item['slots'])
    intent = torch.LongTensor(new_item['intent'])
    
    src_utt = src_utt.to(device)
    y_slots = y_slots.to(device)
    intent = intent.to(device)
    y_lengths = torch.LongTensor(y_lengths).to(device)
    
    new_item['utterances'] = src_utt
    new_item['intents'] = intent
    new_item['y_slots'] = y_slots
    new_item['slots_len'] = y_lengths
    return new_item

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
    words = sum([x['utterance'].split() for x in train_raw], [])
    corpus = train_raw + dev_raw + test_raw
    slots = set(sum([line['slots'].split() for line in corpus],[]))
    intents = set([line['intent'] for line in corpus])
    
    lang = Lang(words, intents, slots, cutoff=0)
    
    # Create datasets
    train_dataset = IntentsAndSlots(train_raw, lang)
    dev_dataset = IntentsAndSlots(dev_raw, lang)
    test_dataset = IntentsAndSlots(test_raw, lang)
    
    # Create dataloaders
    train_loader = data.DataLoader(
        train_dataset, 
        batch_size=128, 
        collate_fn=collate_fn, 
        shuffle=True
    )
    dev_loader = data.DataLoader(
        dev_dataset, 
        batch_size=64, 
        collate_fn=collate_fn
    )
    test_loader = data.DataLoader(
        test_dataset, 
        batch_size=64, 
        collate_fn=collate_fn
    )
    
    # Model hyperparameters
    hid_size = 200
    emb_size = 300
    out_slot = len(lang.slot2id)
    out_int = len(lang.intent2id)
    vocab_len = len(lang.word2id)
    
    # Create and initialize model
    model = ModelIAS(
        hid_size=hid_size,
        out_slot=out_slot,
        out_int=out_int,
        emb_size=emb_size,
        vocab_len=vocab_len,
        pad_index=0,
        dropout=0.1
    ).to(device)
    
    model.apply(init_weights)
    
    # Train model
    best_model, test_metrics = train_model(
        model=model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        test_loader=test_loader,
        lang=lang,
        learning_rate=0.001,
        n_epochs=50, # 100 changed
        patience=3
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
    }, "model_bin/ias_model.pt")
