# Add the class of your model only
# Here is where you define the architecture of your model using pytorch

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report

def train_loop(data, optimizer, model, device):
    """
    Training loop for one epoch.
    
    Args:
        data (DataLoader): Training data loader
        optimizer (torch.optim.Optimizer): Optimizer
        model (nn.Module): Model to train
        device (torch.device): Device to use
        
    Returns:
        tuple: (total_loss, slot_loss, intent_loss)
    """
    model.train()
    total_loss_array = []
    slot_loss_array = []
    intent_loss_array = []
    
    for batch in tqdm(data, desc="Training"):
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        slot_labels = batch['slot_labels'].to(device)
        intent_labels = batch['intent_labels'].to(device)
        slot_mask = batch['slot_mask'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        slot_logits, intent_logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # Compute loss
        total_loss, slot_loss, intent_loss = model.compute_loss(
            slot_logits=slot_logits,
            intent_logits=intent_logits,
            slot_labels=slot_labels,
            intent_labels=intent_labels,
            slot_mask=slot_mask
        )
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        # Store losses
        total_loss_array.append(total_loss.item())
        slot_loss_array.append(slot_loss.item())
        intent_loss_array.append(intent_loss.item())
    
    return (
        np.mean(total_loss_array),
        np.mean(slot_loss_array),
        np.mean(intent_loss_array)
    )

def eval_loop(data, model, device, id2slot, id2intent):
    """
    Evaluation loop.
    
    Args:
        data (DataLoader): Evaluation data loader
        model (nn.Module): Model to evaluate
        device (torch.device): Device to use
        id2slot (dict): Mapping from slot IDs to slot names
        id2intent (dict): Mapping from intent IDs to intent names
        
    Returns:
        tuple: (slot_metrics, intent_metrics, total_loss, slot_loss, intent_loss)
    """
    model.eval()
    total_loss_array = []
    slot_loss_array = []
    intent_loss_array = []
    
    ref_intents = []
    hyp_intents = []
    ref_slots = []
    hyp_slots = []
    
    with torch.no_grad():
        for batch in tqdm(data, desc="Evaluating"):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            slot_labels = batch['slot_labels'].to(device)
            intent_labels = batch['intent_labels'].to(device)
            slot_mask = batch['slot_mask'].to(device)
            words = batch['words']
            
            # Forward pass
            slot_logits, intent_logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            
            # Compute loss
            total_loss, slot_loss, intent_loss = model.compute_loss(
                slot_logits=slot_logits,
                intent_logits=intent_logits,
                slot_labels=slot_labels,
                intent_labels=intent_labels,
                slot_mask=slot_mask
            )
            
            # Store losses
            total_loss_array.append(total_loss.item())
            slot_loss_array.append(slot_loss.item())
            intent_loss_array.append(intent_loss.item())
            
            # Intent inference
            intent_preds = torch.argmax(intent_logits, dim=1)
            ref_intents.extend([id2intent[i] for i in intent_labels.cpu().numpy()])
            hyp_intents.extend([id2intent[i] for i in intent_preds.cpu().numpy()])
            
            # Slot inference
            slot_preds = torch.argmax(slot_logits, dim=2)
            for i, (pred, label, mask, word) in enumerate(zip(
                slot_preds.cpu().numpy(),
                slot_labels.cpu().numpy(),
                slot_mask.cpu().numpy(),
                words
            )):
                # Filter out padding, special tokens and ignore_index (-100)
                valid_preds = [id2slot[p] for p, m in zip(pred, mask) if m and p != -100]
                valid_labels = [id2slot[l] for l, m in zip(label, mask) if m and l != -100]
                valid_words = [w for w, m in zip(word, mask) if m]
                
                ref_slots.append([(w, l) for w, l in zip(valid_words, valid_labels)])
                hyp_slots.append([(w, p) for w, p in zip(valid_words, valid_preds)])
    
    # Calculate metrics
    try:
        slot_metrics = evaluate(ref_slots, hyp_slots)
    except Exception as ex:
        print("Warning:", ex)
        ref_s = set([x[1] for x in ref_slots])
        hyp_s = set([x[1] for x in hyp_slots])
        print(hyp_s.difference(ref_s))
        slot_metrics = {"total": {"f": 0}}
    
    intent_metrics = classification_report(ref_intents, hyp_intents, output_dict=True)
    
    return (
        slot_metrics,
        intent_metrics,
        np.mean(total_loss_array),
        np.mean(slot_loss_array),
        np.mean(intent_loss_array)
    )

def train_model(model, train_loader, dev_loader, test_loader, id2slot, id2intent,
                learning_rate=2e-5, n_epochs=10, patience=3):
    """
    Train the model with early stopping.
    
    Args:
        model (nn.Module): Model to train
        train_loader (DataLoader): Training data loader
        dev_loader (DataLoader): Validation data loader
        test_loader (DataLoader): Test data loader
        id2slot (dict): Mapping from slot IDs to slot names
        id2intent (dict): Mapping from intent IDs to intent names
        learning_rate (float): Learning rate
        n_epochs (int): Maximum number of epochs
        patience (int): Early stopping patience
        
    Returns:
        tuple: (best_model, best_metrics)
    """
    device = next(model.parameters()).device
    
    # Initialize optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Training loop
    best_f1 = 0
    best_model = None
    patience_counter = patience
    
    for epoch in range(1, n_epochs + 1):
        # Training
        total_loss, slot_loss, intent_loss = train_loop(
            train_loader, optimizer, model, device
        )
        
        # Evaluation
        slot_metrics, intent_metrics, dev_total_loss, dev_slot_loss, dev_intent_loss = eval_loop(
            dev_loader, model, device, id2slot, id2intent
        )
        
        # Print metrics
        print(f"\nEpoch {epoch}")
        print(f"Train - Total Loss: {total_loss:.4f}, Slot Loss: {slot_loss:.4f}, Intent Loss: {intent_loss:.4f}")
        print(f"Dev - Total Loss: {dev_total_loss:.4f}, Slot Loss: {dev_slot_loss:.4f}, Intent Loss: {dev_intent_loss:.4f}")
        print(f"Dev - Slot F1: {slot_metrics['total']['f']:.4f}")
        print(f"Dev - Intent Accuracy: {intent_metrics['accuracy']:.4f}")
        
        # Early stopping
        if slot_metrics['total']['f'] > best_f1:
            best_f1 = slot_metrics['total']['f']
            best_model = model.state_dict().copy()
            patience_counter = patience
        else:
            patience_counter -= 1
            
        if patience_counter <= 0:
            print("Early stopping triggered")
            break
    
    # Load best model and evaluate on test set
    model.load_state_dict(best_model)
    test_metrics = eval_loop(test_loader, model, device, id2slot, id2intent)
    
    return model, test_metrics


import re

"""
Modified version of https://pypi.org/project/conlleval/
"""


def stats():
    return {'cor': 0, 'hyp': 0, 'ref': 0}


def evaluate(ref, hyp, otag='O'):
    # evaluation for NLTK
    aligned = align_hyp(ref, hyp)
    return conlleval(aligned, otag=otag)


def align_hyp(ref, hyp):
    # align references and hypotheses for evaluation
    # add last element of token tuple in hyp to ref
    if len(ref) != len(hyp):
        raise ValueError("Size Mismatch: ref: {} & hyp: {}".format(len(ref), len(hyp)))

    out = []
    for i in range(len(ref)):
        if len(ref[i]) != len(hyp[i]):
            raise ValueError("Size Mismatch: ref: {} & hyp: {}".format(len(ref), len(hyp)))
        out.append([(*ref[i][j], hyp[i][j][-1]) for j in range(len(ref[i]))])
    return out


def conlleval(data, otag='O'):
    # token, segment & class level counts for TP, TP+FP, TP+FN
    tok = stats()
    seg = stats()
    cls = {}

    for sent in data:

        prev_ref = otag      # previous reference label
        prev_hyp = otag      # previous hypothesis label
        prev_ref_iob = None  # previous reference label IOB
        prev_hyp_iob = None  # previous hypothesis label IOB

        in_correct = False  # currently processed chunks is correct until now

        for token in sent:

            hyp_iob, hyp = parse_iob(token[-1])
            ref_iob, ref = parse_iob(token[-2])

            ref_e = is_eoc(ref, ref_iob, prev_ref, prev_ref_iob, otag)
            hyp_e = is_eoc(hyp, hyp_iob, prev_hyp, prev_hyp_iob, otag)

            ref_b = is_boc(ref, ref_iob, prev_ref, prev_ref_iob, otag)
            hyp_b = is_boc(hyp, hyp_iob, prev_hyp, prev_hyp_iob, otag)

            if not cls.get(ref) and ref:
                cls[ref] = stats()

            if not cls.get(hyp) and hyp:
                cls[hyp] = stats()

            # segment-level counts
            if in_correct:
                if ref_e and hyp_e and prev_hyp == prev_ref:
                    in_correct = False
                    seg['cor'] += 1
                    cls[prev_ref]['cor'] += 1

                elif ref_e != hyp_e or hyp != ref:
                    in_correct = False

            if ref_b and hyp_b and hyp == ref:
                in_correct = True

            if ref_b:
                seg['ref'] += 1
                cls[ref]['ref'] += 1

            if hyp_b:
                seg['hyp'] += 1
                cls[hyp]['hyp'] += 1

            # token-level counts
            if ref == hyp and ref_iob == hyp_iob:
                tok['cor'] += 1

            tok['ref'] += 1

            prev_ref = ref
            prev_hyp = hyp
            prev_ref_iob = ref_iob
            prev_hyp_iob = hyp_iob

        if in_correct:
            seg['cor'] += 1
            cls[prev_ref]['cor'] += 1

    return summarize(seg, cls)


def parse_iob(t):
    m = re.match(r'^([^-]*)-(.*)$', t)
    return m.groups() if m else (t, None)


def is_boc(lbl, iob, prev_lbl, prev_iob, otag='O'):
    """
    is beginning of a chunk

    supports: IOB, IOBE, BILOU schemes
        - {E,L} --> last
        - {S,U} --> unit

    :param lbl: current label
    :param iob: current iob
    :param prev_lbl: previous label
    :param prev_iob: previous iob
    :param otag: out-of-chunk label
    :return:
    """
    boc = False

    boc = True if iob in ['B', 'S', 'U'] else boc
    boc = True if iob in ['E', 'L'] and prev_iob in ['E', 'L', 'S', otag] else boc
    boc = True if iob == 'I' and prev_iob in ['S', 'L', 'E', otag] else boc

    boc = True if lbl != prev_lbl and iob != otag and iob != '.' else boc

    # these chunks are assumed to have length 1
    boc = True if iob in ['[', ']'] else boc

    return boc


def is_eoc(lbl, iob, prev_lbl, prev_iob, otag='O'):
    """
    is end of a chunk

    supports: IOB, IOBE, BILOU schemes
        - {E,L} --> last
        - {S,U} --> unit

    :param lbl: current label
    :param iob: current iob
    :param prev_lbl: previous label
    :param prev_iob: previous iob
    :param otag: out-of-chunk label
    :return:
    """
    eoc = False

    eoc = True if iob in ['E', 'L', 'S', 'U'] else eoc
    eoc = True if iob == 'B' and prev_iob in ['B', 'I'] else eoc
    eoc = True if iob in ['S', 'U'] and prev_iob in ['B', 'I'] else eoc

    eoc = True if iob == otag and prev_iob in ['B', 'I'] else eoc

    eoc = True if lbl != prev_lbl and iob != otag and prev_iob != '.' else eoc

    # these chunks are assumed to have length 1
    eoc = True if iob in ['[', ']'] else eoc

    return eoc


def score(cor_cnt, hyp_cnt, ref_cnt):
    # precision
    p = 1 if hyp_cnt == 0 else cor_cnt / hyp_cnt
    # recall
    r = 0 if ref_cnt == 0 else cor_cnt / ref_cnt
    # f-measure (f1)
    f = 0 if p+r == 0 else (2*p*r)/(p+r)
    return {"p": p, "r": r, "f": f, "s": ref_cnt}


def summarize(seg, cls):
    # class-level
    res = {lbl: score(cls[lbl]['cor'], cls[lbl]['hyp'], cls[lbl]['ref']) for lbl in set(cls.keys())}
    # micro
    res.update({"total": score(seg.get('cor', 0), seg.get('hyp', 0), seg.get('ref', 0))})
    return res


def read_corpus_conll(corpus_file, fs="\t"):
    """
    read corpus in CoNLL format
    :param corpus_file: corpus in conll format
    :param fs: field separator
    :return: corpus
    """
    featn = None  # number of features for consistency check
    sents = []  # list to hold words list sequences
    words = []  # list to hold feature tuples

    for line in open(corpus_file):
        line = line.strip()
        if len(line.strip()) > 0:
            feats = tuple(line.strip().split(fs))
            if not featn:
                featn = len(feats)
            elif featn != len(feats) and len(feats) != 0:
                raise ValueError("Unexpected number of columns {} ({})".format(len(feats), featn))

            words.append(feats)
        else:
            if len(words) > 0:
                sents.append(words)
                words = []
    return sents


def get_chunks(corpus_file, fs="\t", otag="O"):
    sents = read_corpus_conll(corpus_file, fs=fs)
    return set([parse_iob(token[-1])[1] for sent in sents for token in sent if token[-1] != otag])
