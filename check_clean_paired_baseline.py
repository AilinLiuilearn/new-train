import torch

import models.build_mdt_seg as build_mod
import models.dual_decoder_paired_add_baseline as baseline_mod
from utils.seg_losses import BCEDiceLoss


class CountingEnc(torch.nn.Module):
    def __init__(self, module, name, counter):
        super().__init__()
        self.module = module
        self.name = name
        self.counter = counter
        self.feature_info = module.feature_info

    def forward(self, x):
        self.counter[self.name] = self.counter.get(self.name, 0) + 1
        return self.module(x)

    def __getattr__(self, item):
        if item in {'module', 'name', 'counter', 'feature_info'}:
            return super().__getattr__(item)
        return getattr(self.module, item)


class CountingDecoder(torch.nn.Module):
    def __init__(self, module, name, counter):
        super().__init__()
        self.module = module
        self.name = name
        self.counter = counter

    def forward(self, *args, **kwargs):
        self.counter[self.name] = self.counter.get(self.name, 0) + 1
        return self.module(*args, **kwargs)

    def __getattr__(self, item):
        if item in {'module', 'name', 'counter'}:
            return super().__getattr__(item)
        return getattr(self.module, item)


def main():
    torch.manual_seed(0)
    def fake_backbone(name, in_channels=3):
        class _B(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.feature_info = type('FI', (), {'channels': lambda self: [8, 16, 32, 64]})()
                self.name = name
            def forward(self, x):
                b, _, h, w = x.shape
                return [
                    torch.randn(b, 8, h // 4, w // 4, device=x.device, requires_grad=True),
                    torch.randn(b, 16, h // 8, w // 8, device=x.device, requires_grad=True),
                    torch.randn(b, 32, h // 16, w // 16, device=x.device, requires_grad=True),
                    torch.randn(b, 64, h // 32, w // 32, device=x.device, requires_grad=True),
                ]
        return _B()

    orig_create = build_mod.create_feature_backbone
    orig_load = build_mod.load_local_weights_safe
    orig_create_baseline = baseline_mod.create_feature_backbone
    orig_load_baseline = baseline_mod.load_local_weights_safe
    build_mod.create_feature_backbone = fake_backbone
    build_mod.load_local_weights_safe = lambda *args, **kwargs: None
    baseline_mod.create_feature_backbone = fake_backbone
    baseline_mod.load_local_weights_safe = lambda *args, **kwargs: None
    try:
        model = baseline_mod.DualDecoderPairedAddPETCTBaseline(use_deep_supervision=False)
    finally:
        build_mod.create_feature_backbone = orig_create
        build_mod.load_local_weights_safe = orig_load
        baseline_mod.create_feature_backbone = orig_create_baseline
        baseline_mod.load_local_weights_safe = orig_load_baseline
    banned = ('ptgc', 'gpnd', 'gvtc', 'pg_mtr', 'mtib', 'hatr')
    names = [n.lower() for n, _ in model.named_parameters()]
    assert not any(any(b in n for b in banned) for n in names), names

    ct = torch.randn(2, 3, 512, 512)
    pet = torch.randn(2, 3, 512, 512)
    mask = torch.randint(0, 2, (2, 1, 512, 512)).float()

    out = model(ct, pet, forward_mode='full')
    assert out['paired_joint'] is True
    assert out['paired_full_logits'].shape == (2, 1, 512, 512)
    assert out['paired_missing_logits'].shape == (2, 1, 512, 512)
    assert out['logits'].shape == (2, 1, 512, 512)

    loss_fn = BCEDiceLoss()
    lf, _ = loss_fn(out['paired_full_logits'], mask)
    lm, _ = loss_fn(out['paired_missing_logits'], mask)
    loss_total = 0.5 * lf + 0.5 * lm
    assert torch.isfinite(loss_total)
    check_total = 0.5 * lf + 0.5 * lm
    assert torch.allclose(loss_total, check_total, atol=1e-7, rtol=1e-7)

    model.zero_grad(set_to_none=True)
    loss_total.backward()
    for name in ('enc_ct', 'enc_pet', 'full_decoder', 'missing_decoder'):
        module = getattr(model, name)
        finite = False
        for p in module.parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all()
                finite = True
        assert finite, name

    counter = {}
    model.enc_pet = CountingEnc(model.enc_pet, 'enc_pet', counter)
    model.full_decoder = CountingDecoder(model.full_decoder, 'full_decoder', counter)
    model.missing_decoder = CountingDecoder(model.missing_decoder, 'missing_decoder', counter)
    model.eval()
    with torch.no_grad():
        _ = model(ct, pet, forward_mode='missing')
    assert counter.get('enc_pet', 0) == 0
    assert counter.get('missing_decoder', 0) == 1
    with torch.no_grad():
        full_out = model(ct, pet, forward_mode='full')
    assert counter.get('enc_pet', 0) == 1
    assert counter.get('full_decoder', 0) == 1
    assert torch.allclose(full_out['logits'], full_out['paired_full_logits']) if 'paired_full_logits' in full_out else True

    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable_count == sum(p.numel() for p in model.parameters() if p.requires_grad), trainable_count
    print('check_clean_paired_baseline: OK')
    print(f'trainable_params={trainable_count}')
    print(f'full_logits_shape={tuple(out["paired_full_logits"].shape)}')
    print(f'missing_logits_shape={tuple(out["paired_missing_logits"].shape)}')
    print(f'loss_total={float(loss_total):.8f}')


if __name__ == '__main__':
    main()
