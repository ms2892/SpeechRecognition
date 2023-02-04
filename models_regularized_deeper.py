import torch.nn as nn

class BiLSTM(nn.Module):

    def __init__(self, num_layers, in_dims, hidden_dims, out_dims):
        super().__init__()

        self.lstm = nn.LSTM(in_dims, hidden_dims, num_layers, bidirectional=True)
        self.dropout = nn.Dropout(0.05)
        self.proj = nn.Linear(hidden_dims * 2, hidden_dims)
        self.proj2 = nn.Linear(hidden_dims, out_dims)

    def forward(self, feat):
        hidden, _ = self.lstm(feat)
        hidden = self.dropout(hidden)
        output = self.proj(hidden)
        output = self.proj2(output)
#         output = self.dropout(output)
        return output
