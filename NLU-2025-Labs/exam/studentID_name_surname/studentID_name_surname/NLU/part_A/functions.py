# Add the class of your model only
# Here is where you define the architecture of your model using pytorch

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report

def init_weights(model):
    """
    Initialize model weights using Xavier uniform initialization for input weights
    and orthogonal initialization for hidden weights.
    
    Args:
        model (nn.Module): Model to initialize
    """
    for m in model.modules():
        if type(m) in [nn.LSTM]:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.xavier_uniform_(param[idx*mul:(idx+1)*mul])
                elif 'weight_hh' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.orthogonal_(param[idx*mul:(idx+1)*mul])
                elif 'bias' in name:
                    param.data.fill_(0)
        else:
            if type(m) in [nn.Linear]:
                torch.nn.init.uniform_(m.weight, -0.01, 0.01)
                if m.bias is not None:
                    m.bias.data.fill_(0.01)

def train_loop(data, optimizer, criterion_slots, criterion_intents, model, clip=5):
    """
    Training loop for one epoch.
    
    Args:
        data (DataLoader): Training data loader
        optimizer (torch.optim.Optimizer): Optimizer
        criterion_slots (nn.Module): Loss function for slot filling
        criterion_intents (nn.Module): Loss function for intent classification
        model (nn.Module): Model to train
        clip (float): Gradient clipping value
        
    Returns:
        tuple: (slot_loss, intent_loss)
    """
    model.train()
    slot_loss_array = []
    intent_loss_array = []
    
    for sample in data:
        optimizer.zero_grad()
        
        # Forward pass
        slots, intent = model(sample['utterances'], sample['slots_len'])
        
        # Calculate losses
        loss_intent = criterion_intents(intent, sample['intents'])
        loss_slot = criterion_slots(slots, sample['y_slots'])
        
        # Combined loss
        loss = loss_intent + loss_slot
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        # Update weights
        optimizer.step()
        
        # Store losses
        slot_loss_array.append(loss_slot.item())
        intent_loss_array.append(loss_intent.item())
    
    return np.mean(slot_loss_array), np.mean(intent_loss_array)

def eval_loop(data, criterion_slots, criterion_intents, model, lang):
    """
    Evaluation loop.
    
    Args:
        data (DataLoader): Evaluation data loader
        criterion_slots (nn.Module): Loss function for slot filling
        criterion_intents (nn.Module): Loss function for intent classification
        model (nn.Module): Model to evaluate
        lang (Lang): Language object containing mappings
        
    Returns:
        tuple: (slot_metrics, intent_metrics, slot_loss, intent_loss)
    """
    model.eval()
    slot_loss_array = []
    intent_loss_array = []
    
    ref_intents = []
    hyp_intents = []
    ref_slots = []
    hyp_slots = []
    
    with torch.no_grad():
        for sample in data:
            # Forward pass
            slots, intent = model(sample['utterances'], sample['slots_len'])
            
            # Calculate losses
            loss_intent = criterion_intents(intent, sample['intents'])
            loss_slot = criterion_slots(slots, sample['y_slots'])
            
            # Store losses
            slot_loss_array.append(loss_slot.item())
            intent_loss_array.append(loss_intent.item())
            
            # Intent inference
            out_intents = [lang.id2intent[x] for x in torch.argmax(intent, dim=1).tolist()]
            gt_intents = [lang.id2intent[x] for x in sample['intents'].tolist()]
            ref_intents.extend(gt_intents)
            hyp_intents.extend(out_intents)
            
            # Slot inference
            output_slots = torch.argmax(slots, dim=1)
            for id_seq, seq in enumerate(output_slots):
                length = sample['slots_len'].tolist()[id_seq]
                utt_ids = sample['utterances'][id_seq][:length].tolist()
                gt_ids = sample['y_slots'][id_seq].tolist()
                gt_slots = [lang.id2slot[elem] for elem in gt_ids[:length]]
                utterance = [lang.id2word[elem] for elem in utt_ids]
                to_decode = seq[:length].tolist()
                
                ref_slots.append([(utterance[id_el], elem) for id_el, elem in enumerate(gt_slots)])
                tmp_seq = []
                for id_el, elem in enumerate(to_decode):
                    tmp_seq.append((utterance[id_el], lang.id2slot[elem]))
                hyp_slots.append(tmp_seq)
    
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
    
    return slot_metrics, intent_metrics, np.mean(slot_loss_array), np.mean(intent_loss_array)

def train_model(model, train_loader, dev_loader, test_loader, lang, 
                learning_rate=0.001, n_epochs=100, patience=3):
    """
    Train the model with early stopping.
    
    Args:
        model (nn.Module): Model to train
        train_loader (DataLoader): Training data loader
        dev_loader (DataLoader): Validation data loader
        test_loader (DataLoader): Test data loader
        lang (Lang): Language object containing mappings
        learning_rate (float): Learning rate
        n_epochs (int): Maximum number of epochs
        patience (int): Early stopping patience
        
    Returns:
        tuple: (best_model, best_metrics)
    """
    # Initialize optimizer and loss functions
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion_slots = nn.CrossEntropyLoss(ignore_index=model.pad_index)
    criterion_intents = nn.CrossEntropyLoss()
    
    # Training loop
    best_f1 = 0
    best_model = None
    patience_counter = patience
    
    for epoch in tqdm(range(1, n_epochs + 1)):
        # Training
        slot_loss, intent_loss = train_loop(
            train_loader, optimizer, criterion_slots, criterion_intents, model
        )
        
        # Evaluation
        slot_metrics, intent_metrics, dev_slot_loss, dev_intent_loss = eval_loop(
            dev_loader, criterion_slots, criterion_intents, model, lang
        )
        
        # Print metrics
        print(f"\nEpoch {epoch}")
        print(f"Train - Slot Loss: {slot_loss:.4f}, Intent Loss: {intent_loss:.4f}")
        print(f"Dev - Slot Loss: {dev_slot_loss:.4f}, Intent Loss: {dev_intent_loss:.4f}")
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
    test_metrics = eval_loop(test_loader, criterion_slots, criterion_intents, model, lang)
    
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
