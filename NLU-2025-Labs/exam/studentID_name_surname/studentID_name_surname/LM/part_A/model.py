import torch
import torch.nn as nn

class LM_LSTM(nn.Module):
    """
    LSTM-based Language Model with dropout layers.
    
    This model implements a language model using LSTM cells instead of vanilla RNN.
    It includes two dropout layers:
    1. After the embedding layer
    2. Before the final linear layer
    
    Args:
        emb_size (int): Size of the embedding vectors
        hidden_size (int): Size of the LSTM hidden state
        output_size (int): Size of the vocabulary (output space)
        pad_index (int): Index of the padding token in vocabulary
        emb_dropout (float): Dropout probability for embedding layer
        out_dropout (float): Dropout probability for output layer
        n_layers (int): Number of LSTM layers
    """
    
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, 
                 emb_dropout=0.1, out_dropout=0.1, n_layers=1):
        super(LM_LSTM, self).__init__()
        
        # Token ids to vectors
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        
        # Dropout after embedding
        self.emb_dropout = nn.Dropout(emb_dropout)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=emb_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True
        )
        
        # Dropout before final layer
        self.out_dropout = nn.Dropout(out_dropout)
        
        # Linear layer to project the hidden layer to output space
        self.output = nn.Linear(hidden_size, output_size)
        
        self.pad_token = pad_index
        
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

