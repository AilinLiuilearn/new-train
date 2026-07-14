# -*- coding: utf-8 -*-

import torch

from models.c2maot_w1_loss import C2MAOTHierarchicalW1Loss, one_dimensional_empirical_wasserstein_1


def main():
    torch.manual_seed(7)

    x = torch.randn(2, 3, 4, 5)
    y = x.clone()
    w1 = one_dimensional_empirical_wasserstein_1(x, y)
    print('TEST 1 identity', float(w1))
    assert float(w1) < 1e-7

    y = x + 2.0
    w1 = one_dimensional_empirical_wasserstein_1(x, y)
    print('TEST 2 constant shift', float(w1))
    assert abs(float(w1) - 2.0) < 1e-5

    w1_xy = one_dimensional_empirical_wasserstein_1(x, y)
    w1_yx = one_dimensional_empirical_wasserstein_1(y, x)
    print('TEST 3 symmetry', float(w1_xy), float(w1_yx))
    assert torch.allclose(w1_xy, w1_yx, atol=1e-7, rtol=0)

    x1 = torch.tensor([[3.0, 1.0, 2.0]])
    y1 = torch.tensor([[0.0, 2.0, 4.0]])
    w1 = one_dimensional_empirical_wasserstein_1(x1, y1)
    print('TEST 4 sort correctness', float(w1))
    assert abs(float(w1) - (2.0 / 3.0)) < 1e-6

    x = torch.randn(2, 6, requires_grad=True)
    y = torch.randn(2, 6)
    loss = one_dimensional_empirical_wasserstein_1(x, y)
    loss.backward()
    print('TEST 5 gradient', x.grad is not None, torch.isfinite(x.grad).all().item(), float(x.grad.norm()))
    assert x.grad is not None and torch.isfinite(x.grad).all() and float(x.grad.norm()) > 0

    alpha_loss = C2MAOTHierarchicalW1Loss(alpha=1.5)
    src = {
        1: torch.randn(2, 4, 3, 3),
        2: torch.randn(2, 4, 2, 2),
        3: torch.randn(2, 4, 2, 2),
        4: torch.randn(2, 4, 1, 1),
    }
    tgt = {k: v.clone() for k, v in src.items()}
    total, diag = alpha_loss(src, tgt, (1, 2, 3, 4))
    print('TEST 6 hierarchical weight', {k: float(v) for k, v in diag.items() if 'weight' in k})
    assert float(diag['pg_mtr_ot_s1_weight']) < float(diag['pg_mtr_ot_s2_weight']) < float(diag['pg_mtr_ot_s3_weight']) < float(diag['pg_mtr_ot_s4_weight'])
    assert float(total) < 1e-7

    bad_src = {1: torch.randn(1, 2, 2, 2)}
    bad_tgt = {1: torch.randn(1, 2, 3, 3)}
    try:
        alpha_loss(bad_src, bad_tgt, (1,))
        raise AssertionError('Expected shape mismatch to raise RuntimeError')
    except RuntimeError:
        print('TEST 7 shape mismatch ok')


if __name__ == '__main__':
    main()
