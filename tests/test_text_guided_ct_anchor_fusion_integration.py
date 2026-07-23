import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.dual_shared_add_baseline import DualSharedAddPETCTBaseline
from models.mppc import MPPC
from models.text_guided_ct_anchor_fusion import TextGuidedCTAnchorFusion
from tasks.mdt_seg import MDTSegTeacher


CHANNELS = (64, 128, 320, 512)
TEXT_EMBEDDINGS = torch.nn.functional.normalize(
    (torch.arange(64, dtype=torch.float32).reshape(2, 32) + 1), dim=-1
)


def _pyramid(batch=2, requires_grad=True):
    sizes = ((64, 64), (32, 32), (16, 16), (8, 8))
    return [torch.randn(batch, c, h, w, requires_grad=requires_grad) for c, (h, w) in zip(CHANNELS, sizes)]


def _make_model():
    return DualSharedAddPETCTBaseline(
        fusion_text_embeddings=TEXT_EMBEDDINGS,
        ct_pretrained_path=None,
        pet_pretrained_path=None,
        use_deep_supervision=False,
    )


def test_fusion_shapes_and_finite():
    fusion = TextGuidedCTAnchorFusion(text_embeddings=TEXT_EMBEDDINGS, channels=CHANNELS)
    ct = _pyramid()
    pet = _pyramid()
    full = fusion(ct, pet, mode='full')
    missing = fusion(ct, pet, mode='missing')
    assert len(full) == len(missing) == 4
    for out, ref in zip(full, ct):
        assert out.shape == ref.shape
        assert torch.isfinite(out).all()


def test_mode_selects_different_text_vectors():
    fusion = TextGuidedCTAnchorFusion(text_embeddings=TEXT_EMBEDDINGS, channels=CHANNELS)
    ct = _pyramid(batch=1, requires_grad=False)
    pet = _pyramid(batch=1, requires_grad=False)
    _, aux_full = fusion(ct, pet, mode='full', return_aux=True)
    _, aux_missing = fusion(ct, pet, mode='missing', return_aux=True)
    assert aux_full['mode'] == 'full'
    assert aux_missing['mode'] == 'missing'
    assert aux_full['prompt'] != aux_missing['prompt']


def test_source_text_embeddings_buffer_and_state_dict():
    fusion = TextGuidedCTAnchorFusion(text_embeddings=TEXT_EMBEDDINGS, channels=CHANNELS)
    assert 'source_text_embeddings' in dict(fusion.named_buffers())
    assert not dict(fusion.named_buffers())['source_text_embeddings'].requires_grad
    assert 'source_text_embeddings' in fusion.state_dict()


def test_fusion_parameter_count_and_gradients():
    fusion = TextGuidedCTAnchorFusion(text_embeddings=TEXT_EMBEDDINGS, channels=CHANNELS)
    assert fusion.trainable_parameter_count() < 3_000_000
    ct = _pyramid()
    pet = _pyramid()
    loss = sum(t.mean() for t in fusion(ct, pet, mode='full'))
    loss.backward()
    grads = [p.grad for p in fusion.parameters() if p.requires_grad]
    assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)


def test_missing_backward_has_gradients_and_no_pet_encoder(monkeypatch):
    model = _make_model()
    calls = {'pet': 0}
    orig = model.enc_pet.forward

    def wrapped(*args, **kwargs):
        calls['pet'] += 1
        return orig(*args, **kwargs)

    monkeypatch.setattr(model.enc_pet, 'forward', wrapped)
    out = model(torch.randn(1, 1, 64, 64), None, forward_mode='missing')
    loss = out['logits'].mean()
    loss.backward()
    assert calls['pet'] == 0
    assert any(p.grad is not None for p in model.fusion.parameters() if p.requires_grad)


def test_full_backward_updates_mppc_and_fusion_gradients():
    model = _make_model()
    out = model(torch.randn(1, 1, 64, 64), torch.randn(1, 1, 64, 64), forward_mode='full', mask=torch.zeros(1, 1, 64, 64))
    loss = out['logits'].mean()
    loss.backward()
    assert any(p.grad is not None for p in model.fusion.parameters() if p.requires_grad)


def test_checkpoint_roundtrip_and_state_dict_keys(tmp_path):
    model = _make_model()
    task = MDTSegTeacher({'model': model}, type('Cfg', (), {'learning_rate': 1e-4, 'weight_decay': 1e-4, 'mixed_precision': False, 'loss_smooth': 1.0, 'bce_weight': 1.0, 'dice_weight': 1.0, 'random_state': 2023})())
    ckpt_path = tmp_path / 'ckpt.pth.tar'
    task.save_checkpoint(str(ckpt_path), 1)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    assert any(k.startswith('mppc.') for k in ckpt['model'])
    assert any(k.startswith('fusion.') for k in ckpt['model'])
    assert not any(k.startswith('apsf.') for k in ckpt['model'])
    reloaded = _make_model()
    reloaded.load_state_dict(ckpt['model'], strict=True)


def test_no_auxiliary_fusion_loss_and_decoder_shapes():
    model = _make_model()
    task = MDTSegTeacher({'model': model}, type('Cfg', (), {'learning_rate': 1e-4, 'weight_decay': 1e-4, 'mixed_precision': False, 'loss_smooth': 1.0, 'bce_weight': 1.0, 'dice_weight': 1.0, 'random_state': 2023})())
    batch = {'ct': torch.randn(1, 1, 64, 64), 'pet': torch.randn(1, 1, 64, 64), 'mask': torch.zeros(1, 1, 64, 64)}
    loss, logits, outputs, stats = task.train_step(batch, forward_mode='full')
    assert torch.allclose(stats['loss_total'], stats['loss_seg'])
    assert 'loss_aux_apsf' not in stats
    assert logits.shape[-2:] == batch['mask'].shape[-2:]
    assert outputs['aux'] == {}
