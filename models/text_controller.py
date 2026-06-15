import torch
import torch.nn as nn


class TextModalityController(nn.Module):
    def __init__(self, channels, embed_dim=256):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.embedding_available = nn.Parameter(torch.randn(self.embed_dim) * 1e-3)
        self.embedding_missing = nn.Parameter(torch.randn(self.embed_dim) * 1e-3)
        self.prior_mlps = nn.ModuleList([self._make_gate_mlp(c) for c in channels])
        self.pet_mlps = nn.ModuleList([self._make_gate_mlp(c) for c in channels])
        self.text_mlps = nn.ModuleList([self._make_gate_mlp(c) for c in channels])
        self._init_conservative_gates()

    def _make_gate_mlp(self, out_channels):
        return nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dim, out_channels),
            nn.Sigmoid(),
        )

    def _init_conservative_gates(self):
        for mlps in (self.prior_mlps, self.pet_mlps, self.text_mlps):
            for mlp in mlps:
                last = mlp[-2]
                nn.init.zeros_(last.weight)
                nn.init.zeros_(last.bias)

    def forward(self, pet_available):
        if pet_available.dim() > 1:
            pet_available = pet_available.view(-1)
        m = pet_available.float().unsqueeze(1)
        text_embed = m * self.embedding_available.unsqueeze(0) + (1.0 - m) * self.embedding_missing.unsqueeze(0)
        prior_gates = [mlp(text_embed).unsqueeze(-1).unsqueeze(-1) for mlp in self.prior_mlps]
        pet_gates = [mlp(text_embed).unsqueeze(-1).unsqueeze(-1) for mlp in self.pet_mlps]
        text_gates = [mlp(text_embed).unsqueeze(-1).unsqueeze(-1) for mlp in self.text_mlps]
        return {
            'text_embed': text_embed,
            'prior_gates': prior_gates,
            'pet_gates': pet_gates,
            'text_gates': text_gates,
        }
