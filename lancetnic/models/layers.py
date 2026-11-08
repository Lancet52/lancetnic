import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenEmbedding(nn.Module):
    """Превращает вход пайплайна в эмбеддинги слов.

    Вход пайплайна в режиме 'sequence' устроен так:
    [номера слов колонки 1 | номера слов колонки 2 | ... | числовые признаки].
    Каждая текстовая колонка занимает ровно max_len позиций.

    Args:
        vocab_sizes: Размер словаря каждой текстовой колонки.
        max_len: Длина последовательности одной колонки.
        embed_dim: Размер вектора одного слова.
        dropout: Вероятность dropout для эмбеддингов.
    """

    def __init__(self, vocab_sizes: list[int], max_len: int,
                 embed_dim: int, dropout: float) -> None:
        super().__init__()
        self.n_text = len(vocab_sizes)
        self.max_len = max_len

        self.embedding = nn.Embedding(sum(vocab_sizes), embed_dim)
        offsets = torch.tensor([0] + list(vocab_sizes[:-1])).cumsum(0)
        self.register_buffer("offsets", offsets.view(1, -1, 1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Разбирает вход на слова и числа.

        Args:
            x: Вход пайплайна формы [B, input_size].

        Returns:
            emb: Эмбеддинги слов [B * n_text, max_len, embed_dim].
            mask: True там, где настоящее слово, False - пустое место [B * n_text, max_len].
            scalars: Числовые признаки [B, n_scalar] (может быть 0 колонок).
        """
        batch = x.size(0)
        text_len = self.n_text * self.max_len
        ids = x[:, :text_len].long().reshape(batch, self.n_text, self.max_len)  
        scalars = x[:, text_len:]                                               
        mask = ids > 0                                                         
        emb = self.embedding(ids + self.offsets) * mask.unsqueeze(-1)           
        emb = self.dropout(emb)
        return (emb.reshape(batch * self.n_text, self.max_len, -1),
                mask.reshape(batch * self.n_text, self.max_len),
                scalars)


def masked_attention(out: torch.Tensor, mask: torch.Tensor,
                     attn_weight: torch.Tensor) -> torch.Tensor:
    """Внимание, которое не смотрит на пустые места.

    Args:
        out: Выходы LSTM/GRU [N, L, H].
        mask: True для настоящих слов [N, L].
        attn_weight: Обучаемый вектор внимания [H].

    Returns:
        Взвешенная сумма выходов [N, H].
    """
    valid = mask.clone()
    valid[:, 0] = True                      
    energy = (out @ attn_weight).masked_fill(~valid, float('-inf'))   
    attn = F.softmax(energy, dim=1)                                  
    return torch.bmm(attn.unsqueeze(1), out).squeeze(1)              