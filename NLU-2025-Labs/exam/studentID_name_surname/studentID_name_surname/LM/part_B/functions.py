# Add the class of your model only
# Here is where you define the architecture of your model using pytorch

import torch
import torch.nn as nn
import torch.optim as optim
import math

def init_weights(m):
    """Initialize weights for the model."""
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

def train_model(model, train_loader, dev_loader, test_loader, learning_rate=0.1, 
                n_epochs=100, patience=3, use_avsgd=True):
    """
    Train the language model.
    
    Args:
        model: The language model to train
        train_loader: DataLoader for training data
        dev_loader: DataLoader for validation data
        test_loader: DataLoader for test data
        learning_rate: Learning rate for optimization
        n_epochs: Number of epochs to train
        patience: Number of epochs to wait for improvement before early stopping
        use_avsgd: Whether to use AvSGD optimizer
    
    Returns:
        tuple: (best_model, best_ppl, final_ppl)
    """
    device = next(model.parameters()).device
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 0 is pad token
    
    if use_avsgd:
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_ppl = float('inf')
    best_model = None
    patience_counter = 0
    
    for epoch in range(n_epochs):
        # Training
        model.train()
        total_loss = 0
        total_tokens = 0
        
        for batch in train_loader:
            source = batch['source'].to(device)
            target = batch['target'].to(device)
            
            optimizer.zero_grad()
            output = model(source)  # Shape: (batch_size, vocab_size, seq_len)
            
            # Reshape output and target for loss calculation
            # Output: (batch_size * seq_len, vocab_size)
            # Target: (batch_size * seq_len)
            output = output.permute(0, 2, 1).contiguous().view(-1, output.size(1))
            target = target.contiguous().view(-1)
            
            loss = criterion(output, target)
            loss.backward()
            
            if use_avsgd:
                # AvSGD: Update learning rate based on gradient statistics
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad.data = param.grad.data / (1 + torch.norm(param.grad.data))
            
            optimizer.step()
            
            total_loss += loss.item() * batch['number_tokens']
            total_tokens += batch['number_tokens']
        
        train_ppl = math.exp(total_loss / total_tokens)
        
        # Validation
        model.eval()
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in dev_loader:
                source = batch['source'].to(device)
                target = batch['target'].to(device)
                
                output = model(source)
                output = output.permute(0, 2, 1).contiguous().view(-1, output.size(1))
                target = target.contiguous().view(-1)
                
                loss = criterion(output, target)
                total_loss += loss.item() * batch['number_tokens']
                total_tokens += batch['number_tokens']
        
        val_ppl = math.exp(total_loss / total_tokens)
        
        print(f'Epoch {epoch+1}/{n_epochs}:')
        print(f'Training PPL: {train_ppl:.2f}')
        print(f'Validation PPL: {val_ppl:.2f}')
        
        if use_avsgd:
            scheduler.step(val_ppl)
        
        # Early stopping
        if val_ppl < best_ppl:
            best_ppl = val_ppl
            best_model = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping triggered after {epoch+1} epochs')
                break
    
    # Load best model and evaluate on test set
    model.load_state_dict(best_model)
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in test_loader:
            source = batch['source'].to(device)
            target = batch['target'].to(device)
            
            output = model(source)
            output = output.permute(0, 2, 1).contiguous().view(-1, output.size(1))
            target = target.contiguous().view(-1)
            
            loss = criterion(output, target)
            total_loss += loss.item() * batch['number_tokens']
            total_tokens += batch['number_tokens']
    
    final_ppl = math.exp(total_loss / total_tokens)
    
    return best_model, best_ppl, final_ppl
