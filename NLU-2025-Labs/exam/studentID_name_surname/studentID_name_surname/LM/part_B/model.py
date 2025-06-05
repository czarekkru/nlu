import torch
import torch.nn as nn
import torch.nn.functional as F

class VariationalDropout(nn.Module):
    """
    Variational Dropout implementation.
    Applies the same dropout mask across all timesteps.
    """
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        
    def forward(self, x):
        if not self.training or self.p == 0:
            return x
            
        # Create dropout mask for the entire sequence
        mask = torch.bernoulli(torch.ones_like(x[:, 0]) * (1 - self.p)) / (1 - self.p)
        mask = mask.unsqueeze(1).expand_as(x)
        return x * mask

class LM_LSTM_Enhanced(nn.Module):
    """
    Enhanced LSTM-based Language Model with:
    - Weight Tying (sharing weights between embedding and output layers)
    - Variational Dropout
    - Support for AvSGD optimization
    
    Args:
        emb_size (int): Size of the embedding vectors
        hidden_size (int): Size of the LSTM hidden state
        output_size (int): Size of the vocabulary (output space)
        pad_index (int): Index of the padding token in vocabulary
        emb_dropout (float): Dropout probability for embedding layer
        out_dropout (float): Dropout probability for output layer
        n_layers (int): Number of LSTM layers
        use_weight_tying (bool): Whether to use weight tying
        use_variational_dropout (bool): Whether to use variational dropout
    """
    
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, 
                 emb_dropout=0.1, out_dropout=0.1, n_layers=1,
                 use_weight_tying=True, use_variational_dropout=True):
        super(LM_LSTM_Enhanced, self).__init__()
        
        # Token ids to vectors
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        
        # Dropout layers
        if use_variational_dropout:
            self.emb_dropout = VariationalDropout(emb_dropout)
            self.out_dropout = VariationalDropout(out_dropout)
        else:
            self.emb_dropout = nn.Dropout(emb_dropout)
            self.out_dropout = nn.Dropout(out_dropout)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=emb_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True
        )
        
        # Linear layer to project the hidden layer to output space
        self.output = nn.Linear(hidden_size, output_size)
        
        # Weight tying
        if use_weight_tying:
            self.output.weight = self.embedding.weight
        
        self.pad_token = pad_index
        self.use_weight_tying = use_weight_tying
        
    def forward(self, input_sequence):
        """
        Forward pass of the model.
        
        Args:
            input_sequence (torch.Tensor): Input sequence of token indices
                Shape: (batch_size, sequence_length)
                
        Returns:
            torch.Tensor: Output logits for each position
                Shape: (batch_size, vocab_size, sequence_length)
        """
        # Get embeddings and apply dropout
        emb = self.embedding(input_sequence)
        emb = self.emb_dropout(emb)
        
        # Get LSTM outputs
        lstm_out, _ = self.lstm(emb)
        
        # Apply dropout before final layer
        lstm_out = self.out_dropout(lstm_out)
        
        # Project to vocabulary space
        output = self.output(lstm_out)
        
        # Permute to match expected shape (batch_size, vocab_size, sequence_length)
        output = output.permute(0, 2, 1)
        
        return output

