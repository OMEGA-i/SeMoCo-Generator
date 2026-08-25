"""Vendored HumanML3D ``text_mot_match`` encoders (motion/text co-embedding).

Upstream reference: text-to-motion ``networks/modules.py`` (three encoders used
by the classic HumanML3D matching evaluator).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


def init_weight(m: nn.Module) -> None:
    if isinstance(m, (nn.Conv1d, nn.Linear, nn.ConvTranspose1d)):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class MovementConvEncoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, 4, 2, 1),
            nn.Dropout(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(hidden_size, output_size, 4, 2, 1),
            nn.Dropout(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.out_net = nn.Linear(output_size, output_size)
        self.main.apply(init_weight)
        self.out_net.apply(init_weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: [B, T, C] -> [B, T', C]
        inputs = inputs.permute(0, 2, 1)
        outputs = self.main(inputs).permute(0, 2, 1)
        return self.out_net(outputs)


class TextEncoderBiGRUCo(nn.Module):
    def __init__(
        self,
        word_size: int,
        pos_size: int,
        hidden_size: int,
        output_size: int,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.device = device
        self.pos_emb = nn.Linear(pos_size, word_size)
        self.input_emb = nn.Linear(word_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.output_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, output_size),
        )
        self.input_emb.apply(init_weight)
        self.pos_emb.apply(init_weight)
        self.output_net.apply(init_weight)
        self.hidden_size = hidden_size
        self.hidden = nn.Parameter(torch.randn((2, 1, self.hidden_size), requires_grad=True))

    def forward(
        self,
        word_embs: torch.Tensor,
        pos_onehot: torch.Tensor,
        cap_lens: torch.Tensor,
    ) -> torch.Tensor:
        num_samples = word_embs.shape[0]
        pos_embs = self.pos_emb(pos_onehot)
        inputs = word_embs + pos_embs
        input_embs = self.input_emb(inputs)
        hidden = self.hidden.repeat(1, num_samples, 1)
        lengths = cap_lens.detach().cpu().tolist()
        emb = pack_padded_sequence(input_embs, lengths, batch_first=True, enforce_sorted=False)
        _gru_seq, gru_last = self.gru(emb, hidden)
        gru_last = torch.cat([gru_last[0], gru_last[1]], dim=-1)
        return self.output_net(gru_last)


class MotionEncoderBiGRUCo(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.device = device
        self.input_emb = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.output_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, output_size),
        )
        self.input_emb.apply(init_weight)
        self.output_net.apply(init_weight)
        self.hidden_size = hidden_size
        self.hidden = nn.Parameter(torch.randn((2, 1, self.hidden_size), requires_grad=True))

    def forward(self, inputs: torch.Tensor, m_lens: torch.Tensor) -> torch.Tensor:
        num_samples = inputs.shape[0]
        input_embs = self.input_emb(inputs)
        hidden = self.hidden.repeat(1, num_samples, 1)
        lengths = m_lens.detach().cpu().tolist()
        emb = pack_padded_sequence(input_embs, lengths, batch_first=True, enforce_sorted=False)
        _gru_seq, gru_last = self.gru(emb, hidden)
        gru_last = torch.cat([gru_last[0], gru_last[1]], dim=-1)
        return self.output_net(gru_last)


__all__ = [
    "MovementConvEncoder",
    "MotionEncoderBiGRUCo",
    "TextEncoderBiGRUCo",
]
