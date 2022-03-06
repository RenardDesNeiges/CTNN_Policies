import numpy as np                      # numpy  
import torch.nn as nn                   # pytorch neural network modules
import kerasncp as kncp                 # Keras LTC implementation (for Neural Circuit Policies)
from kerasncp.torch import LTCCell      # the LCP network module is actually part of the keras library
import torch                            # Pytorch
import pytorch_lightning as pl          # Pytorch Lightning for easy training
import torch.utils.data as data         # Data Utilities

class RNNSequence(nn.Module):
    def __init__(
        self,
        rnn_cell,
    ):
        super(RNNSequence, self).__init__()
        self.rnn_cell = rnn_cell

    def forward(self, x): # just recursively constructs a time-sequence with outputs that cover the entire sequence (which is required for BPTT)
        device = x.device
        batch_size = x.size(0)
        seq_len = x.size(1)
        hidden_state = torch.zeros(
            (batch_size, self.rnn_cell.state_size), device=device
        )
        outputs = []
        for t in range(seq_len):
            inputs = x[:, t]
            new_output, hidden_state = self.rnn_cell.forward(inputs, hidden_state)
            outputs.append(new_output)
        outputs = torch.stack(outputs, dim=1)  # return entire sequence
        return outputs