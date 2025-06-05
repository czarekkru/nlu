# Add the class of your model only
# Here is where you define the architecture of your model using pytorch

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
from conll import evaluate

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
                # Filter out padding and special tokens
                valid_preds = [id2slot[p] for p, m in zip(pred, mask) if m]
                valid_labels = [id2slot[l] for l, m in zip(label, mask) if m]
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
