import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class ModelIAS(nn.Module):
    """
    Enhanced Intent and Slot Filling model with:
    - Bidirectional LSTM
    - Dropout layers
    - Joint training for intent classification and slot filling
    
    Args:
        hid_size (int): Size of the LSTM hidden state
        out_slot (int): Number of slot labels
        out_int (int): Number of intent labels
        emb_size (int): Size of the embedding vectors
        vocab_len (int): Size of the vocabulary
        n_layer (int): Number of LSTM layers
        pad_index (int): Index of the padding token
        dropout (float): Dropout probability
    """
    
    def __init__(self, hid_size, out_slot, out_int, emb_size, vocab_len, n_layer=1, pad_index=0, dropout=0.1):
        super(ModelIAS, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_len, emb_size, padding_idx=pad_index)
        
        # Dropout after embedding
        self.emb_dropout = nn.Dropout(dropout)
        
        # Bidirectional LSTM
        self.utt_encoder = nn.LSTM(
            input_size=emb_size,
            hidden_size=hid_size,
            num_layers=n_layer,
            bidirectional=True,
            batch_first=True
        )
        
        # Dropout after LSTM
        self.lstm_dropout = nn.Dropout(dropout)
        
        # Output layers
        # For slot filling: 2 * hid_size because of bidirectional
        self.slot_out = nn.Linear(hid_size * 2, out_slot)
        # For intent classification: 2 * hid_size because of bidirectional
        self.intent_out = nn.Linear(hid_size * 2, out_int)
        
        self.pad_index = pad_index
        
    def forward(self, utterance, seq_lengths):
        """
        Forward pass of the model.
        
        Args:
            utterance (torch.Tensor): Input sequence of token indices
                Shape: (batch_size, sequence_length)
            seq_lengths (torch.Tensor): Lengths of sequences in the batch
                Shape: (batch_size,)
                
        Returns:
            tuple: (slot_logits, intent_logits)
                slot_logits: Shape (batch_size, out_slot, sequence_length)
                intent_logits: Shape (batch_size, out_int)
        """
        # Get embeddings and apply dropout
        utt_emb = self.embedding(utterance)  # (batch_size, seq_len, emb_size)
        utt_emb = self.emb_dropout(utt_emb)
        
        # Pack sequences for efficient computation
        packed_input = pack_padded_sequence(
            utt_emb, 
            seq_lengths.cpu().numpy(), 
            batch_first=True
        )
        
        # Process through LSTM
        packed_output, (last_hidden, cell) = self.utt_encoder(packed_input)
        
        # Unpack the sequence
        utt_encoded, _ = pad_packed_sequence(packed_output, batch_first=True)
        
        # Apply dropout after LSTM
        utt_encoded = self.lstm_dropout(utt_encoded)
        
        # Get the last hidden state for intent classification
        # Concatenate forward and backward hidden states
        last_hidden = torch.cat((last_hidden[-2,:,:], last_hidden[-1,:,:]), dim=1)
        
        # Compute slot logits
        slots = self.slot_out(utt_encoded)  # (batch_size, seq_len, out_slot)
        slots = slots.permute(0, 2, 1)  # (batch_size, out_slot, seq_len)
        
        # Compute intent logits
        intent = self.intent_out(last_hidden)  # (batch_size, out_int)
        
        return slots, intent


