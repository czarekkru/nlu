# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *

import torch
import torch.utils.data as data
from model import LM_LSTM_Enhanced
import os

def read_file(path, eos_token="<eos>"):
    """Read file and add end of sentence token."""
    output = []
    with open(path, "r") as f:
        for line in f.readlines():
            output.append(line.strip() + " " + eos_token)
    return output

class Lang:
    """Language class to handle vocabulary."""
    def __init__(self, corpus, special_tokens=[]):
        self.word2id = self.get_vocab(corpus, special_tokens)
        self.id2word = {v:k for k, v in self.word2id.items()}
        
    def get_vocab(self, corpus, special_tokens=[]):
        output = {}
        i = 0 
        for st in special_tokens:
            output[st] = i
            i += 1
        for sentence in corpus:
            for w in sentence.split():
                if w not in output:
                    output[w] = i
                    i += 1
        return output

class PennTreeBank(data.Dataset):
    """Penn TreeBank dataset class."""
    def __init__(self, corpus, lang, device='cpu'):
        self.source = []
        self.target = []
        self.device = device
        
        for sentence in corpus:
            self.source.append(sentence.split()[0:-1])
            self.target.append(sentence.split()[1:])
        
        self.source_ids = self.mapping_seq(self.source, lang)
        self.target_ids = self.mapping_seq(self.target, lang)

    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        src = torch.LongTensor(self.source_ids[idx]).to(self.device)
        trg = torch.LongTensor(self.target_ids[idx]).to(self.device)
        sample = {'source': src, 'target': trg}
        return sample
    
    def mapping_seq(self, data, lang):
        res = []
        for seq in data:
            tmp_seq = []
            for x in seq:
                if x in lang.word2id:
                    tmp_seq.append(lang.word2id[x])
                else:
                    print('OOV found!')
                    print('You have to deal with that')
                    break
            res.append(tmp_seq)
        return res

def collate_fn(data, pad_token, device='cpu'):
    """Collate function for DataLoader."""
    def merge(sequences):
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths)==0 else max(lengths)
        padded_seqs = torch.LongTensor(len(sequences),max_len).fill_(pad_token).to(device)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq
        padded_seqs = padded_seqs.detach()
        return padded_seqs, lengths
    
    data.sort(key=lambda x: len(x["source"]), reverse=True)
    new_item = {}
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]

    source, _ = merge(new_item["source"])
    target, lengths = merge(new_item["target"])
    
    new_item["source"] = source
    new_item["target"] = target
    new_item["number_tokens"] = sum(lengths)
    return new_item

if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dataset_path = os.path.normpath(os.path.join(base_path, "dataset", "PennTreeBank"))
    train_raw = read_file(os.path.join(dataset_path, "ptb.train.txt"))
    dev_raw = read_file(os.path.join(dataset_path, "ptb.valid.txt"))
    test_raw = read_file(os.path.join(dataset_path, "ptb.test.txt"))
    
    # Create vocabulary
    lang = Lang(train_raw, ["<pad>", "<eos>"])
    
    # Create datasets
    train_dataset = PennTreeBank(train_raw, lang, device)
    dev_dataset = PennTreeBank(dev_raw, lang, device)
    test_dataset = PennTreeBank(test_raw, lang, device)
    
    # Create dataloaders
    train_loader = data.DataLoader(
        train_dataset, 
        batch_size=16, # 64 changed
        collate_fn=lambda x: collate_fn(x, lang.word2id["<pad>"], device),
        shuffle=True
    )
    dev_loader = data.DataLoader(
        dev_dataset, 
        batch_size=32, # 128 changed
        collate_fn=lambda x: collate_fn(x, lang.word2id["<pad>"], device)
    )
    test_loader = data.DataLoader(
        test_dataset, 
        batch_size=32, # 128 changed
        collate_fn=lambda x: collate_fn(x, lang.word2id["<pad>"], device)
    )
    
    # Model hyperparameters
    emb_size = 400 # 300 changed
    hidden_size = 400  # Changed from 200 to match emb_size for weight tying
    vocab_len = len(lang.word2id)
    
    # Create and initialize model
    model = LM_LSTM_Enhanced(
        emb_size=emb_size,
        hidden_size=hidden_size,
        output_size=vocab_len,
        pad_index=lang.word2id["<pad>"],
        emb_dropout=0.1,
        out_dropout=0.1,
        use_weight_tying=True,
        use_variational_dropout=True
    ).to(device)
    
    model.apply(init_weights)
    
    # Train model with AvSGD
    best_model, best_ppl, final_ppl = train_model(
        model=model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        test_loader=test_loader,
        learning_rate=1,  # 0.1 changed # Higher learning rate for AvSGD
        n_epochs=3, # 100 changed
        patience=3,
        use_avsgd=True
    )
    
    print(f"\nBest validation PPL: {best_ppl:.2f}")
    print(f"Final test PPL: {final_ppl:.2f}")
    
    # Save model
    os.makedirs("model_bin", exist_ok=True)
    torch.save(model.state_dict(), "model_bin/lstm_lm_enhanced.pt")
