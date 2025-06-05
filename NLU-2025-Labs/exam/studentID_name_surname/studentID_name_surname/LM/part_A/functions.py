# Add the class of your model only
# Here is where you define the architecture of your model using pytorch

import torch
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

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

def train_loop(data, optimizer, criterion, model, clip=5):
    """
    Training loop for one epoch.
    
    Args:
        data (DataLoader): Training data loader
        optimizer (torch.optim.Optimizer): Optimizer
        criterion (nn.Module): Loss function
        model (nn.Module): Model to train
        clip (float): Gradient clipping value
        
    Returns:
        float: Average loss per token
    """
    model.train()
    loss_array = []
    number_of_tokens = []
    
    for sample in data:
        optimizer.zero_grad()
        output = model(sample['source'])
        loss = criterion(output, sample['target'])
        loss_array.append(loss.item() * sample["number_tokens"])
        number_of_tokens.append(sample["number_tokens"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        
    return sum(loss_array)/sum(number_of_tokens)

def eval_loop(data, eval_criterion, model):
    """
    Evaluation loop.
    
    Args:
        data (DataLoader): Evaluation data loader
        eval_criterion (nn.Module): Loss function
        model (nn.Module): Model to evaluate
        
    Returns:
        tuple: (perplexity, average loss per token)
    """
    model.eval()
    loss_array = []
    number_of_tokens = []
    
    with torch.no_grad():
        for sample in data:
            output = model(sample['source'])
            loss = eval_criterion(output, sample['target'])
            loss_array.append(loss.item())
            number_of_tokens.append(sample["number_tokens"])
            
    ppl = math.exp(sum(loss_array) / sum(number_of_tokens))
    loss_to_return = sum(loss_array) / sum(number_of_tokens)
    return ppl, loss_to_return

def plot_training_curves(train_losses, dev_losses, sampled_epochs, title="Training Curves"):
    """
    Plot training and validation loss curves.
    
    Args:
        train_losses (list): Training losses
        dev_losses (list): Validation losses
        sampled_epochs (list): Epoch numbers
        title (str): Plot title
    """
    plt.figure(figsize=(10, 6))
    plt.plot(sampled_epochs, train_losses, label='Training Loss')
    plt.plot(sampled_epochs, dev_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def train_model(model, train_loader, dev_loader, test_loader, 
                learning_rate=0.001, n_epochs=100, patience=3):
    """
    Train the model with early stopping.
    
    Args:
        model (nn.Module): Model to train
        train_loader (DataLoader): Training data loader
        dev_loader (DataLoader): Validation data loader
        test_loader (DataLoader): Test data loader
        learning_rate (float): Learning rate
        n_epochs (int): Maximum number of epochs
        patience (int): Early stopping patience
        
    Returns:
        tuple: (best_model, best_ppl, final_ppl)
    """
    # Initialize optimizer (AdamW)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Initialize loss functions
    criterion_train = nn.CrossEntropyLoss(ignore_index=model.pad_token)
    criterion_eval = nn.CrossEntropyLoss(ignore_index=model.pad_token, reduction='sum')
    
    # Training loop
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_ppl = math.inf
    best_model = None
    
    pbar = tqdm(range(1, n_epochs + 1))
    for epoch in pbar:
        loss = train_loop(train_loader, optimizer, criterion_train, model)
        
        if epoch % 1 == 0:
            sampled_epochs.append(epoch)
            losses_train.append(np.asarray(loss).mean())
            ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
            losses_dev.append(np.asarray(loss_dev).mean())
            pbar.set_description(f"PPL: {ppl_dev:.2f}")
            
            if ppl_dev < best_ppl:
                best_ppl = ppl_dev
                best_model = model.state_dict().copy()
                patience = 3
            else:
                patience -= 1
                
            if patience <= 0:
                break
    
    # Load best model and evaluate on test set
    model.load_state_dict(best_model)
    final_ppl, _ = eval_loop(test_loader, criterion_eval, model)
    
    # Plot training curves
    plot_training_curves(losses_train, losses_dev, sampled_epochs)
    
    return model, best_ppl, final_ppl
