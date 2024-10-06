import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    """
    LSTM-based classifier that handles variable-length sequences using PyTorch's packing utilities.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)  # Fully connected layer for classification
        
    def forward(self, x, lengths):
        # Pack padded sequence
        packed_input = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        
        # LSTM forward pass
        packed_output, (hn, cn) = self.lstm(packed_input)
        
        # Only the hidden state of the last layer at the last time step is needed
        out = hn[-1]
        
        # Fully connected layer for classification
        out = self.fc(out)
        
        return out
