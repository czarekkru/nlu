import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class BERT_IAS(nn.Module):
    """
    BERT-based Intent and Slot Filling model with:
    - Pre-trained BERT encoder
    - Multi-task learning for intent classification and slot filling
    - Sub-tokenization handling
    
    Args:
        out_slot (int): Number of slot labels
        out_int (int): Number of intent labels
        bert_model_name (str): Name of the pre-trained BERT model
        dropout (float): Dropout probability
    """
    
    def __init__(self, out_slot, out_int, bert_model_name='bert-base-uncased', dropout=0.1):
        super(BERT_IAS, self).__init__()
        
        # Load pre-trained BERT
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        
        # Get BERT's hidden size
        self.hidden_size = self.bert.config.hidden_size
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Intent classification head
        self.intent_classifier = nn.Linear(self.hidden_size, out_int)
        
        # Slot filling head
        self.slot_classifier = nn.Linear(self.hidden_size, out_slot)
        
        # Loss weights for multi-task learning
        self.intent_loss_weight = 0.5
        self.slot_loss_weight = 0.5
        
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """
        Forward pass of the model.
        
        Args:
            input_ids (torch.Tensor): Input token IDs
                Shape: (batch_size, sequence_length)
            attention_mask (torch.Tensor): Attention mask
                Shape: (batch_size, sequence_length)
            token_type_ids (torch.Tensor, optional): Token type IDs
                Shape: (batch_size, sequence_length)
                
        Returns:
            tuple: (slot_logits, intent_logits)
                slot_logits: Shape (batch_size, sequence_length, out_slot)
                intent_logits: Shape (batch_size, out_int)
        """
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # Get sequence output and pooled output
        sequence_output = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)
        pooled_output = outputs.pooler_output  # (batch_size, hidden_size)
        
        # Apply dropout
        sequence_output = self.dropout(sequence_output)
        pooled_output = self.dropout(pooled_output)
        
        # Intent classification
        intent_logits = self.intent_classifier(pooled_output)  # (batch_size, out_int)
        
        # Slot filling
        slot_logits = self.slot_classifier(sequence_output)  # (batch_size, seq_len, out_slot)
        
        return slot_logits, intent_logits
    
    def compute_loss(self, slot_logits, intent_logits, slot_labels, intent_labels, slot_mask):
        """
        Compute the combined loss for both tasks.
        
        Args:
            slot_logits (torch.Tensor): Slot classification logits
            intent_logits (torch.Tensor): Intent classification logits
            slot_labels (torch.Tensor): Ground truth slot labels
            intent_labels (torch.Tensor): Ground truth intent labels
            slot_mask (torch.Tensor): Mask for valid slot positions
            
        Returns:
            torch.Tensor: Combined loss
        """
        # Slot filling loss
        slot_loss = nn.CrossEntropyLoss(ignore_index=-100)(
            slot_logits.view(-1, slot_logits.size(-1)),
            slot_labels.view(-1)
        )
        
        # Intent classification loss
        intent_loss = nn.CrossEntropyLoss()(
            intent_logits,
            intent_labels
        )
        
        # Combine losses with weights
        total_loss = (
            self.slot_loss_weight * slot_loss +
            self.intent_loss_weight * intent_loss
        )
        
        return total_loss, slot_loss, intent_loss


