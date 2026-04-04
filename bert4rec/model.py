from __future__ import annotations

import torch
import torch.nn as nn


class BERT4Rec(nn.Module):
    """
    BERT4Rec model adapted from the original idea and architecture:
    https://github.com/FeiSun/BERT4Rec
    """

    def __init__(
        self,
        num_items: int,
        max_len: int,
        hidden_size: int = 256,
        num_heads: int = 4,
        num_layers: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_items = num_items
        self.max_len = max_len
        self.hidden_size = hidden_size

        self.pad_token_id = 0
        self.mask_token_id = num_items + 1
        self.vocab_size = num_items + 2

        self.item_embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=hidden_size,
            padding_idx=self.pad_token_id,
        )
        self.position_embedding = nn.Embedding(max_len, hidden_size)

        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_bias = nn.Parameter(torch.zeros(self.vocab_size))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: [batch_size, max_len]
        batch_size, seq_len = input_ids.size()
        device = input_ids.device

        pos_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)

        token_emb = self.item_embedding(input_ids)
        pos_emb = self.position_embedding(pos_ids)

        x = token_emb + pos_emb
        x = self.layer_norm(x)
        x = self.dropout(x)

        key_padding_mask = input_ids.eq(self.pad_token_id)
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)

        # Weight tying with item embedding matrix.
        logits = torch.matmul(x, self.item_embedding.weight.transpose(0, 1)) + self.output_bias
        return logits
