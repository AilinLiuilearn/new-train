import copy

import torch

from models.dual_decoder_pg_mtr_retrieval import DualDecoderPGMTRRetrieval


class DummyFeatureInfo:
    def __init__(self, channels):
        self._channels = channels

    def channels(self):
        return self._channels


class DummyBackbone(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.feature_info = DummyFeatureInfo(channels)

    def forward(self, x):
        b, _, h, w = x.shape
        feats = []
        for i, c in enumerate(self.feature_info.channels(), start=1):
            scale = 2 ** i
            feats.append(torch.randn(b, c, max(1, h // scale), max(1, w // scale), device=x.device, dtype=x.dtype))
        return feats


def make_model(stage_mode='all'):
    model = DualDecoderPGMTRRetrieval.__new__(DualDecoderPGMTRRetrieval)
    torch.nn.Module.__init__(model)
    model.use_deep_supervision = False
    model.pg_mtr_detach_bank_missing = True
    model.enc_ct = DummyBackbone([64, 128, 320, 512])
    model.enc_pet = DummyBackbone([64, 128, 320, 512])
    model.ct_align = torch.nn.Identity()
    model.fusion = torch.nn.Identity()
    model.full_decoder = torch.nn.Identity()
    model.missing_decoder = torch.nn.Identity()
    from models.pg_mtr import PETGroundedMetabolicTokenRetrieval
    model.pg_mtr = PETGroundedMetabolicTokenRetrieval([64, 128, 320, 512], num_tokens=8, temperature=0.07, stage_mode=stage_mode)
    model.retrieval_projs = torch.nn.ModuleDict({str(k): torch.nn.Conv2d(model.pg_mtr.stage_modules[str(k)].latent_dim, [64, 128, 320, 512][k-1], 1, bias=False) for k in model.pg_mtr.active_stage_numbers})
    for p in model.retrieval_projs.parameters():
        torch.nn.init.zeros_(p)
    return model


def test_all_stage_activation():
    model = make_model('all')
    assert model.pg_mtr.active_stage_numbers == (1, 2, 3, 4)
    assert set(model.pg_mtr.stage_modules.keys()) == {'1', '2', '3', '4'}
    assert set(model.retrieval_projs.keys()) == {'1', '2', '3', '4'}


def test_retrieved_memory_shape():
    from models.pg_mtr import PETGroundedMetabolicTokenRetrieval
    pg = PETGroundedMetabolicTokenRetrieval([64, 128, 320, 512], stage_mode='all')
    feats = [torch.randn(2, c, 16 // (2 ** i), 16 // (2 ** i)) for i, c in enumerate([64, 128, 320, 512])]
    retrieved, _, diag = pg(feats, mode='missing')
    assert set(retrieved.keys()) == {1, 2, 3, 4}
    for i, mem in retrieved.items():
        assert mem.shape[0] == 2
        assert mem.shape[1] == pg.stage_modules[str(i)].latent_dim
        assert mem.shape[-2:] == feats[i - 1].shape[-2:]
        assert torch.isfinite(mem).all()
    assert torch.isfinite(torch.stack([v for v in diag.values() if torch.is_tensor(v)])).all()


def test_zero_init_retrieval_degenerates_to_ct():
    model = make_model('all')
    ct = [torch.randn(2, c, 16 // (2 ** i), 16 // (2 ** i)) for i, c in enumerate([64, 128, 320, 512])]
    retrieved, _, _ = model.pg_mtr(ct, mode='missing')
    missing_feats = [ct[i] + model.retrieval_projs[str(i + 1)](retrieved[i + 1]) for i in range(4)]
    for a, b in zip(ct, missing_feats):
        assert torch.allclose(a, b, atol=1e-6)


def test_full_aux_and_missing_routing_grads():
    model = DualDecoderPGMTRRetrieval(
        pg_mtr_stages='all',
        pg_mtr_num_tokens=8,
        pg_mtr_temperature=0.07,
    )
    ct = torch.randn(2, 3, 64, 64, requires_grad=True)
    pet = torch.randn(2, 3, 64, 64, requires_grad=True)
    full_out = model._forward_full(ct, pet, (64, 64))
    loss = full_out['aux_losses']['pg_mtr_route_loss'] + full_out['aux_losses']['pg_mtr_mem_loss']
    loss.backward()
    assert any(p.grad is not None for p in model.full_decoder.parameters())
    assert any(p.grad is not None for p in model.pg_mtr.parameters())


def test_mixed_route_order():
    model = DualDecoderPGMTRRetrieval(pg_mtr_stages='all')
    ct = torch.randn(4, 3, 64, 64)
    pet = torch.randn(4, 3, 64, 64)
    pet_available = torch.tensor([1, 0, 1, 0])
    out = model(ct, pet, pet_available=pet_available, forward_mode='auto')
    assert out['logits'].shape[0] == 4
    assert torch.isfinite(out['logits']).all()
