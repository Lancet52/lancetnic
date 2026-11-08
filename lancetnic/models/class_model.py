import torch
import torch.nn as nn
import torch.nn.functional as F

from lancetnic.models.layers import TokenEmbedding, masked_attention


# Структура модели для многоклассовой классификации
class LancetMC(nn.Module):
    """Модель для многоклассовой классификации с LSTM-слоем.

    Работает в двух режимах:
        - TF-IDF (vocab_sizes=None): как раньше, вектор признаков - один шаг LSTM.
        - 'sequence': LSTM читает слова по порядку, берётся состояние
          после последнего настоящего слова.

    Args:
        input_size: Размер входа пайплайна.
        hidden_size: Размер скрытого состояния (и вектора слова).
        num_layers: Количество слоёв LSTM.
        num_classes: Количество классов.
        dropout: Вероятность dropout.
        vocab_sizes: Размеры словарей текстовых колонок (только для 'sequence').
        max_len: Длина последовательности одной колонки (только для 'sequence').
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                 num_classes: int, dropout: float,
                 vocab_sizes: list[int] | None = None,
                 max_len: int | None = None) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_sizes = vocab_sizes
        lstm_dropout = dropout if num_layers > 1 else 0.0

        if vocab_sizes is None:
            lstm_input, fc_input = input_size, hidden_size
        else:
            self.tokens = TokenEmbedding(vocab_sizes, max_len, hidden_size, dropout)
            n_scalar = input_size - len(vocab_sizes) * max_len
            lstm_input, fc_input = hidden_size, hidden_size * len(vocab_sizes) + n_scalar

        self.lstm = nn.LSTM(lstm_input, hidden_size, num_layers,
                            batch_first=True, dropout=lstm_dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(fc_input, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Режим TF-IDF (и старые сохранённые модели без атрибута vocab_sizes)
        if getattr(self, "vocab_sizes", None) is None:
            if x.dim() == 2:
                x = x.unsqueeze(1)
            out, _ = self.lstm(x)
            out = self.dropout(out[:, -1, :])
            return self.fc(out)

        # Режим 'sequence'
        batch = x.size(0)
        emb, mask, scalars = self.tokens(x)                  # [B*C, L, H], [B*C, L], [B, S]
        out, _ = self.lstm(emb)                              # [B*C, L, H]
        last_idx = mask.sum(dim=1).clamp(min=1) - 1          # [B*C] номер последнего слова
        rows = torch.arange(out.size(0), device=out.device)
        last = out[rows, last_idx]                           # [B*C, H]
        features = torch.cat([last.reshape(batch, -1), scalars], dim=1)   # [B, C*H + S]
        return self.fc(self.dropout(features))

# Структура модели для многоклассовой классификации с механикой внимания
class LancetMCA(nn.Module):
    """Модель для многоклассовой классификации с LSTM-слоем и вниманием.

    Работает в двух режимах:
        - TF-IDF (vocab_sizes=None): как раньше.
        - 'sequence': LSTM читает слова по порядку, внимание выбирает
          важные слова (пустые места игнорируются).

    Args:
        input_size: Размер входа пайплайна.
        hidden_size: Размер скрытого состояния (и вектора слова).
        num_layers: Количество слоёв LSTM.
        num_classes: Количество классов.
        dropout: Вероятность dropout.
        vocab_sizes: Размеры словарей текстовых колонок (только для 'sequence').
        max_len: Длина последовательности одной колонки (только для 'sequence').
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                 num_classes: int, dropout: float,
                 vocab_sizes: list[int] | None = None,
                 max_len: int | None = None) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_sizes = vocab_sizes
        lstm_dropout = dropout if num_layers > 1 else 0.0

        if vocab_sizes is None:
            lstm_input, fc_input = input_size, hidden_size
        else:
            self.tokens = TokenEmbedding(vocab_sizes, max_len, hidden_size, dropout)
            n_scalar = input_size - len(vocab_sizes) * max_len
            lstm_input, fc_input = hidden_size, hidden_size * len(vocab_sizes) + n_scalar

        self.lstm = nn.LSTM(lstm_input, hidden_size, num_layers,
                            batch_first=True, dropout=lstm_dropout)
        self.attn_weight = nn.Parameter(torch.randn(hidden_size))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(fc_input, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Режим TF-IDF (и старые сохранённые модели без атрибута vocab_sizes)
        if getattr(self, "vocab_sizes", None) is None:
            if x.dim() == 2:
                x = x.unsqueeze(1)
            out, _ = self.lstm(x)
            out = self.dropout(out)
            energy = out @ self.attn_weight
            attn = F.softmax(energy, dim=1)
            context = torch.bmm(attn.unsqueeze(1), out).squeeze(1)
            return self.fc(context)

        # Режим 'sequence'
        batch = x.size(0)
        emb, mask, scalars = self.tokens(x)                  # [B*C, L, H], [B*C, L], [B, S]
        out, _ = self.lstm(emb)                              # [B*C, L, H]
        context = masked_attention(self.dropout(out), mask, self.attn_weight)   # [B*C, H]
        features = torch.cat([context.reshape(batch, -1), scalars], dim=1)     # [B, C*H + S]
        return self.fc(features)
    
class LancetMCA_gru(nn.Module):
    """Модель для многоклассовой классификации с GRU-слоем и вниманием.

    То же, что LancetMCA, но вместо LSTM используется GRU
    (меньше параметров, обычно быстрее учится).

    Args:
        input_size: Размер входа пайплайна.
        hidden_size: Размер скрытого состояния (и вектора слова).
        num_layers: Количество слоёв GRU.
        num_classes: Количество классов.
        dropout: Вероятность dropout.
        vocab_sizes: Размеры словарей текстовых колонок (только для 'sequence').
        max_len: Длина последовательности одной колонки (только для 'sequence').
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                 num_classes: int, dropout: float,
                 vocab_sizes: list[int] | None = None,
                 max_len: int | None = None) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_sizes = vocab_sizes
        gru_dropout = dropout if num_layers > 1 else 0.0

        if vocab_sizes is None:
            gru_input, fc_input = input_size, hidden_size
        else:
            self.tokens = TokenEmbedding(vocab_sizes, max_len, hidden_size, dropout)
            n_scalar = input_size - len(vocab_sizes) * max_len
            gru_input, fc_input = hidden_size, hidden_size * len(vocab_sizes) + n_scalar

        self.gru = nn.GRU(gru_input, hidden_size, num_layers,
                          batch_first=True, dropout=gru_dropout)
        self.attn_weight = nn.Parameter(torch.randn(hidden_size))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(fc_input, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Режим TF-IDF (и старые сохранённые модели без атрибута vocab_sizes)
        if getattr(self, "vocab_sizes", None) is None:
            if x.dim() == 2:
                x = x.unsqueeze(1)
            out, _ = self.gru(x)
            out = self.dropout(out)
            energy = out @ self.attn_weight
            attn = F.softmax(energy, dim=1)
            context = torch.bmm(attn.unsqueeze(1), out).squeeze(1)
            return self.fc(context)

        # Режим 'sequence'
        batch = x.size(0)
        emb, mask, scalars = self.tokens(x)                  # [B*C, L, H], [B*C, L], [B, S]
        out, _ = self.gru(emb)                               # [B*C, L, H]
        context = masked_attention(self.dropout(out), mask, self.attn_weight)   # [B*C, H]
        features = torch.cat([context.reshape(batch, -1), scalars], dim=1)     # [B, C*H + S]
        return self.fc(features)

# Простая структура модели для многоклассовой классификации
class ScalpelMC(nn.Module):
    """Модель для многоклассовой классификации с полносвязными слоями"""

    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout):
        super(ScalpelMC, self).__init__()
        # Входной вектор
        self.input_size = input_size
        # Количество нейронов в скрытом слое
        self.hidden_size = hidden_size
        # Количество слоев
        self.num_layers = num_layers
        # Количество классов на выходе
        self.num_classes = num_classes

        # Список слоев
        layers = []
        layers.append(nn.Linear(self.input_size, self.hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
                
        for _ in range(self.num_layers-1):
            layers.append(nn.Linear(self.hidden_size, self.hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        # Выходной слой
        layers.append(nn.Linear(self.hidden_size, self.num_classes))
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        out = self.model(x)
        return out