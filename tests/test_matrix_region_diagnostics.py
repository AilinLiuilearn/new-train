import csv
import torch
from torch import nn
from utils.matrix_region_diagnostics import region_correction_report


def test_region_energies_and_no_bank_writes(tmp_path):
    class Scale(nn.Module):
        def forward(self, x):
            return x*2
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.matrix = nn.Module()
            self.matrix.scales = nn.ModuleList([Scale() for _ in range(4)])
            self.matrix_read_mode = 'off'
            self.register_buffer('counter', torch.tensor(0))
        def forward(self, ct, forward_mode):
            assert forward_mode == 'missing'
            assert self.matrix_read_mode == 'normal'
            for scale in self.matrix.scales: scale(ct)
    model = Model().train()
    batch = {'ct':torch.tensor([[[[2., 4.]]]]), 'mask':torch.tensor([[[[1., 0.]]]])}
    path = tmp_path/'regions.csv'
    rows = region_correction_report(model, [batch], 'cpu', path)
    assert len(rows) == 8
    for row in rows:
        assert row['correction_rms_ratio'] == 1
        assert row['ct_rms'] == (2 if row['region']=='foreground' else 4)
    assert model.training and model.matrix_read_mode == 'off'
    assert model.counter == 0
    assert not any(scale._forward_hooks for scale in model.matrix.scales)
    assert len(list(csv.DictReader(path.open()))) == 8
