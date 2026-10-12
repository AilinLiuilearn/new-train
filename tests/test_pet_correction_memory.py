"""Scientific contracts: no PET leakage, conditional deltas, no self-copy, EMA queues."""
import copy
import unittest
from unittest.mock import patch

import torch
from torch import nn
import torch.nn.functional as F

from models.pet_correction_memory import PairedCorrectionMemory, CorrectionRetriever, audit_regions
from models.pet_correction_model import PETCorrectionModel
from tasks.pet_correction import CorrectionEMA, build_ratio_state, stable_ids


class FakeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 1)
        self.feature_info = type('Info', (), {'channels': lambda _: [8]*4})()
        self.calls = 0
        self.rows = []

    def forward(self, x):
        self.calls += 1
        self.rows.append(x.shape[0])
        return [F.adaptive_avg_pool2d(self.conv(x), (n, n)) for n in (16, 8, 4, 2)]


def model():
    with patch('models.dual_shared_add_baseline.create_feature_backbone', lambda *a, **k: FakeEncoder()):
        return PETCorrectionModel(pretrained=False, decoder_channels=(8, 8, 8, 8),
                                  memory_capacity=16, key_dim=4, topk=2, query_chunk=7,
                                  memory_min_entries=1, memory_max_age=10)


class Contracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)

    def test_ratio_exact_long_term_and_ids_stable(self):
        states = [build_ratio_state(16, i, 2023, .7, 'cpu') for i in range(10)]
        self.assertEqual(sum(int(s.sum()) for s in states), 112)
        self.assertTrue(all(0 < int(s.sum()) < 16 for s in states))
        self.assertTrue(torch.equal(states[2], build_ratio_state(16, 2, 2023, .7, 'cpu')))
        self.assertEqual(stable_ids(['a_1', 'a_1'])[0], stable_ids(['a_1'])[0])

    def test_audit_confident_wrong_ct_can_be_rescued_and_empty_gt_finite(self):
        gt = torch.ones(1, 1, 4, 4)
        audit = audit_regions(torch.full_like(gt, -4.), torch.full_like(gt, 2.), gt,
                              (2, 2), margin=.02, min_support=.6)
        self.assertTrue(audit['eligible'].all())
        self.assertFalse(audit['negative'].any())
        self.assertTrue((audit['reliability'] > 0).all())
        bad = audit_regions(torch.full_like(gt, 2.), torch.full_like(gt, -4.), gt,
                            (2, 2), margin=.02, min_support=.6)
        self.assertTrue(bad['negative'].all())
        self.assertFalse(bad['eligible'].any())
        empty = audit_regions(-torch.ones_like(gt), -2*torch.ones_like(gt), 0*gt,
                              (1, 1), margin=.02, min_support=.6)
        self.assertTrue(torch.isfinite(empty['reliability']).all())

    def test_queue_exclusion_ttl_and_checkpoint(self):
        bank = PairedCorrectionMemory(3, 2, 2, max_age=3)
        bank.enqueue(torch.tensor([[1.,0.],[0.,1.]]), torch.tensor([[2.,3.],[4.,5.]]),
                     torch.ones(2), torch.tensor([10, 11]), torch.tensor([1, 1]), step=2)
        self.assertEqual(bank.allowed(10, 1, step=2, exclude_group=False).sum(), 1)
        self.assertEqual(bank.allowed(10, 1, step=2, exclude_group=True).sum(), 0)
        self.assertEqual(bank.allowed(-1, -1, step=6, exclude_group=False).sum(), 0)
        other = PairedCorrectionMemory(3, 2, 2, max_age=3)
        other.load_state_dict(copy.deepcopy(bank.state_dict()), strict=True)
        for name, val in bank.state_dict().items():
            self.assertTrue(torch.equal(val, other.state_dict()[name]))

    def test_live_key_projection_receives_gradients_but_memory_does_not(self):
        r=CorrectionRetriever(2,2,capacity=4,key_dim=2,topk=2,query_chunk=1,min_entries=1)
        with torch.no_grad():
            r.key_projection.weight.copy_(torch.eye(2))
            r.output.weight.copy_(torch.eye(2).view(2,2,1,1))
        r.memory.enqueue(torch.tensor([[1.,0.],[.7,.7]]),torch.tensor([[2.,-3.],[-1.,2.]]),
                         torch.ones(2),torch.tensor([6,7]),torch.tensor([-1,-1]),step=1)
        h=torch.randn(1,2,1,1,requires_grad=True)
        q=torch.tensor([[[[1.]],[[.2]]]],requires_grad=True)
        out,_=r(h,q,torch.tensor([5]),torch.tensor([-1]),step=1)
        out.square().sum().backward()
        self.assertGreater(r.key_projection.weight.grad.abs().sum(),0)
        self.assertGreater(q.grad.abs().sum(),0)
        self.assertIsNone(r.memory.raw_keys.grad)
        self.assertIsNone(r.memory.values.grad)

    def test_retrieval_real_values_gradients_and_empty_identity(self):
        r = CorrectionRetriever(2, 2, capacity=4, key_dim=2, topk=1, query_chunk=1,
                                min_entries=1, max_age=10)
        with torch.no_grad():
            r.key_projection.weight.copy_(torch.eye(2))
        q = torch.tensor([[[[1.]], [[0.]]]], requires_grad=True)
        h = torch.randn_like(q, requires_grad=True)
        out, hits = r(h, q, torch.tensor([5]), torch.tensor([-1]), step=1)
        self.assertTrue(torch.equal(h, out))
        self.assertEqual(hits.sum(), 0)
        r.memory.enqueue(torch.tensor([[1.,0.]]), torch.tensor([[2.,-3.]]), torch.ones(1),
                         torch.tensor([6]), torch.tensor([-1]), step=1)
        out, hits = r(h, q, torch.tensor([5]), torch.tensor([-1]), step=1)
        self.assertTrue(torch.equal(h, out))  # zero output projection
        out.sum().backward()
        self.assertGreater(r.output.weight.grad.abs().sum(), 0)
        with torch.no_grad():
            r.output.weight.copy_(torch.eye(2).view(2,2,1,1))
        out, _ = r(h, q, torch.tensor([5]), torch.tensor([-1]), step=1)
        self.assertFalse(torch.equal(out, h))
        self.assertTrue(torch.isfinite(out).all())
        excluded, hits = r(h, q, torch.tensor([6]), torch.tensor([-1]), step=1)
        self.assertTrue(torch.equal(excluded, h))

    def test_full_and_missing_baseline_equivalence_zero_init_no_pet_leak(self):
        m = model().eval()
        ct, pet = torch.randn(4, 3, 32, 32), torch.randn(4, 3, 32, 32)
        state = torch.tensor([1,0,1,0])
        with torch.no_grad():
            cf = m._encode_ct(ct)
            ff = m.fusion([c[state.bool()] for c in cf], m._encode_pet(pet[state.bool()]))
            baseline_full = m.decoder(ff, (32,32))['logits']
            baseline_ct = m.decoder([c[~state.bool()] for c in cf], (32,32))['logits']
            out = m(ct, pet, state, sample_ids=torch.arange(4))['logits']
            self.assertTrue(torch.allclose(out[state.bool()], baseline_full, atol=1e-6))
            self.assertTrue(torch.allclose(out[~state.bool()], baseline_ct, atol=1e-6))
            calls = m.enc_pet.calls
            m(ct, pet=None, forward_mode='missing')
            self.assertEqual(calls, m.enc_pet.calls)
            p2 = pet.clone(); p2[~state.bool()] = 1000.
            out2 = m(ct, p2, state, sample_ids=torch.arange(4))['logits']
            self.assertTrue(torch.equal(out[~state.bool()], out2[~state.bool()]))
            self.assertEqual(m.enc_pet.rows[-1], 2)

    def test_conditional_deltas_and_all_scale_backward(self):
        m = model().train()
        ct, pet = torch.randn(3, 3, 32, 32), torch.randn(3, 3, 32, 32)
        out = m(ct, pet, torch.tensor([1,1,0]), training_details=True)
        cf, ff = out['ct_full'], out['fused_full']
        with torch.no_grad():
            projected_ct = m.project(cf)
            projected_full = m.project(ff)
            ref = m.decode_features(projected_ct)
            for s in range(4):
                conditional = m.decode_stage(s, projected_full[s], None if s==3 else ref[s+1])
                self.assertTrue(torch.allclose(out['delta'][s], conditional-ref[s], atol=1e-6))
        loss = out['logits'].square().mean()+out['ct_aux_logits'].square().mean()
        loss += sum((a.square().mean()+b.square().mean()) for a,b in
                    zip(out['reference_logits'],out['plus_logits']))/4
        loss.backward()
        for name in ('enc_ct','enc_pet','ct_align','pet_align','decoder','ct_aux_decoder','stage_heads'):
            grads = [p.grad for p in getattr(m,name).parameters() if p.grad is not None]
            self.assertTrue(grads, name)
            self.assertTrue(all(torch.isfinite(g).all() for g in grads), name)
        self.assertTrue(all(not r.memory.valid.any() for r in m.retrievers))

    def test_ema_copies_queue_without_averaging_and_eval_no_mutation(self):
        m = model().eval()
        ema = CorrectionEMA(m, decay=.99, warmup=False)
        b=m.retrievers[0].memory
        b.enqueue(torch.ones(1,64), 3*torch.ones(1,8), torch.ones(1),
                  torch.tensor([1]),torch.tensor([-1]),step=1)
        m.memory_step.fill_(1); m.retrieval_active.fill_(True)
        ema.update(m)
        for key, val in b.state_dict().items():
            self.assertTrue(torch.equal(val, ema.model.retrievers[0].memory.state_dict()[key]))
        before=copy.deepcopy(ema.model.state_dict())
        ema.model(torch.randn(1,3,32,32), forward_mode='missing', sample_ids=torch.tensor([2]))
        for k,v in before.items():
            self.assertTrue(torch.equal(v, ema.model.state_dict()[k]), k)


if __name__ == '__main__':
    unittest.main()
