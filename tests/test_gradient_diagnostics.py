import torch
from types import SimpleNamespace

from models.dual_shared_add_baseline import DualSharedAddPETCTBaseline
from tasks.mdt_seg import MDTSegTeacher


def _config():
    return SimpleNamespace(ct_backbone='convnextv2_nano', pet_backbone='mit_b1', ct_pretrained_path=None, pet_pretrained_path=None, learning_rate=1e-4, weight_decay=1e-4, mixed_precision=False)


def test_gradient_diagnostics_smoke():
    model = DualSharedAddPETCTBaseline(_config())
    task = MDTSegTeacher({'model': model}, _config())
    batch = {
        'ct': torch.randn(1, 1, 64, 64),
        'pet': torch.randn(1, 1, 64, 64),
        'mask': torch.randint(0, 2, (1, 1, 64, 64)).float(),
    }
    stats = task.gradient_diagnostics(batch)
    assert 'shared_grad_cosine_total' in stats
    assert 'diagnostic_full_grad_norm' in stats
    assert 'diagnostic_missing_grad_norm' in stats
