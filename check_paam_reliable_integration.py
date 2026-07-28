import os
import tempfile

import torch

from models.build_mdt_seg import build_mdt_seg_teacher
from models.dual_shared_add_baseline import DualSharedAddPETCTBaseline
from models.dual_shared_add_paam import DualSharedAddPAAMPETCT
from models.dual_shared_add_paam_reliable import DualSharedAddPAAMReliablePETCT


CHANNELS = (64, 128, 320, 512)
SPATIAL = ((16, 16), (8, 8), (4, 4), (2, 2))


def make_cfg(model_arch='dual_shared_add_paam_reliable'):
    class Cfg:
        pass
    cfg = Cfg()
    cfg.model_arch = model_arch
    cfg.ct_backbone = 'convnextv2_nano'
    cfg.pet_backbone = 'mit_b1'
    cfg.ct_pretrained_path = None
    cfg.pet_pretrained_path = None
    cfg.decoder_channels = [512, 256, 128, 64]
    cfg.use_deep_supervision = False
    cfg.deep_supervision = False
    cfg.paam_k = 8
    return cfg


def make_features(batch=2):
    ct_img = torch.randn(batch, 1, 64, 64)
    pet_img = torch.randn(batch, 1, 64, 64)
    mask = torch.randint(0, 2, (batch, 1, 64, 64)).float()
    return ct_img, pet_img, mask


def test_1_old_paam_untouched():
    assert DualSharedAddPAAMPETCT is not None
    assert os.path.exists('models/dual_shared_add_paam.py')


def test_2_new_model_build():
    model = build_mdt_seg_teacher(make_cfg())['model']
    assert hasattr(model, 'paam')
    assert tuple(model.paam.channels) == CHANNELS
    assert model.paam.K == 8


def test_3_joint_clustering():
    model = DualSharedAddPAAMReliablePETCT(ct_backbone='convnextv2_nano', pet_backbone='mit_b1', paam_k=8)
    mem = model.paam.memories[0]
    q1 = torch.nn.functional.normalize(torch.randn(4, mem.QUERY_DIM), dim=1)
    q2 = torch.nn.functional.normalize(torch.randn(4, mem.QUERY_DIM) + 4.0, dim=1)
    a1 = torch.nn.functional.normalize(torch.randn(4, CHANNELS[0] * 2), dim=1)
    a2 = torch.nn.functional.normalize(torch.randn(4, CHANNELS[0] * 2) - 4.0, dim=1)
    joint1 = torch.nn.functional.normalize(torch.cat([q1, a1], dim=1), dim=1)
    joint2 = torch.nn.functional.normalize(torch.cat([q2, a2], dim=1), dim=1)
    assert not torch.allclose(joint1, joint2)


def test_4_uniform_retrieval_fallback():
    model = DualSharedAddPAAMReliablePETCT(ct_backbone='convnextv2_nano', pet_backbone='mit_b1', paam_k=8)
    mem = model.paam.memories[0]
    mem.memory_ready.fill_(True)
    mem.keys.copy_(torch.zeros_like(mem.keys))
    mem.gamma_proto.copy_(torch.randn_like(mem.gamma_proto))
    mem.beta_proto.copy_(torch.randn_like(mem.beta_proto))
    q = torch.randn(2, 64, 8, 8)
    query, _ = mem.make_query(q)
    ret = mem.retrieve(query)
    assert float(ret.info['reliability'].mean()) <= 1.0
    assert float(ret.info['safe_gamma_abs_mean']) <= float(ret.info['raw_gamma_abs_mean']) + 1e-5


def test_5_deterministic_retrieval():
    model = DualSharedAddPAAMReliablePETCT(ct_backbone='convnextv2_nano', pet_backbone='mit_b1', paam_k=8)
    mem = model.paam.memories[0]
    mem.memory_ready.fill_(True)
    mem.keys.zero_(); mem.keys[0, 0] = 10.0
    mem.gamma_proto[0].fill_(1.0); mem.beta_proto[0].fill_(1.0)
    q = torch.randn(1, 64, 8, 8)
    query, _ = mem.make_query(q)
    ret = mem.retrieve(query)
    assert float(ret.info['top1_weight'].mean()) > 0.1
    assert float(ret.info['safe_gamma_abs_mean']) >= 0.0


def test_6_missing_gradients():
    model = DualSharedAddPAAMReliablePETCT(ct_backbone='convnextv2_nano', pet_backbone='mit_b1', paam_k=8)
    model.train()
    ct, pet, _ = make_features(batch=1)
    out = model(ct, pet=pet, forward_mode='missing')
    loss = out['logits'].mean()
    loss.backward()
    assert any(p.grad is not None for p in model.enc_ct.parameters())


def test_7_full_gradients():
    model = DualSharedAddPAAMReliablePETCT(ct_backbone='convnextv2_nano', pet_backbone='mit_b1', paam_k=8)
    model.train()
    ct, pet, _ = make_features(batch=1)
    outputs = model(ct, pet=pet, forward_mode='full')
    loss = outputs['logits'].mean()
    loss.backward()
    assert any(p.grad is not None for p in model.enc_pet.parameters())


def test_8_real_missing_eval():
    model = DualSharedAddPAAMReliablePETCT(ct_backbone='convnextv2_nano', pet_backbone='mit_b1', paam_k=8)
    model.eval()
    ct, _, _ = make_features(batch=1)
    with torch.no_grad():
        out = model(ct, pet=None, forward_mode='missing')
    assert torch.isfinite(out['logits']).all()
    assert out['paam_info']['leakage_guard'].startswith('PASS')


def test_9_checkpoint():
    model = DualSharedAddPAAMReliablePETCT(ct_backbone='convnextv2_nano', pet_backbone='mit_b1', paam_k=8)
    state = model.state_dict()
    model2 = DualSharedAddPAAMReliablePETCT(ct_backbone='convnextv2_nano', pet_backbone='mit_b1', paam_k=8)
    model2.load_state_dict(state)
    assert torch.equal(model.paam.memories[0].keys, model2.paam.memories[0].keys)
    assert torch.equal(model.paam.memories[0].gamma_proto, model2.paam.memories[0].gamma_proto)
    assert torch.equal(model.paam.memories[0].beta_proto, model2.paam.memories[0].beta_proto)


def main():
    test_1_old_paam_untouched()
    test_2_new_model_build()
    test_3_joint_clustering()
    test_4_uniform_retrieval_fallback()
    test_5_deterministic_retrieval()
    test_8_real_missing_eval()
    test_9_checkpoint()
    print('check_paam_reliable_integration: PASS')


if __name__ == '__main__':
    main()
