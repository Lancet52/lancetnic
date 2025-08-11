import torch
import torch.nn as nn
import torch.nn.functional as F

# Структура модели для многоклассовой классификации
class LancetMC(nn.Module):
    """The standard model for multiclass classification using the LSTM layer"""

    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout):
        super(LancetMC, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=self.dropout)
        self.fc = nn.Linear(self.hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# Структура модели для многоклассовой классификации с механикой внимания
class LancetMCA(nn.Module):
    """A model for multiclass classification using the LSTM layer and attention mechanics"""

    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout):
        super(LancetMCA, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=self.dropout)
        self.attn_weight = nn.Parameter(torch.randn(hidden_size))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.hidden_size, num_classes)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out)
        energy = torch.sum(self.attn_weight * out, dim=2)
        attn_weights = F.softmax(energy, dim=1).unsqueeze(2)
        context_vector = torch.bmm(out.transpose(1,2), attn_weights).squeeze(2)
        out = self.fc(context_vector)
        return out

