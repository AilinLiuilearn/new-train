# -*- coding: utf-8 -*-
"""Tests for the switchable text encoders."""
import torch

from models.text_encoders import TEXT_ENCODERS, load_text_embeddings


def test_clip_embeddings_shape_and_dim():
    emb, dim = load_text_embeddings('clip')
    assert dim == 512
    assert emb.shape == (2, 512)
    assert torch.isfinite(emb).all()


def test_biomedclip_embeddings_shape_and_dim():
    emb, dim = load_text_embeddings('biomedclip')
    assert dim == 512
    assert emb.shape == (2, 512)
    assert torch.isfinite(emb).all()


def test_biomedbert_embeddings_shape_and_dim():
    emb, dim = load_text_embeddings('biomedbert')
    assert dim == 768
    assert emb.shape == (2, 768)
    assert torch.isfinite(emb).all()


def test_encoders_differ():
    clip, _ = load_text_embeddings('clip')
    bio, _ = load_text_embeddings('biomedclip')
    assert not torch.allclose(clip, bio, atol=1e-3)


def test_invalid_and_beit3_rejected():
    import pytest
    with pytest.raises(ValueError):
        load_text_embeddings('nope')
    with pytest.raises(NotImplementedError):
        load_text_embeddings('beit3')
    assert set(TEXT_ENCODERS) == {'clip', 'biomedclip', 'biomedbert'}


def test_v2_accepts_768_dim_bert_embeddings():
    from models.full_petct_asymmetric_fusion_v2 import FullPETCTAsymmetricFusionV2
    emb, dim = load_text_embeddings('biomedbert')
    m = FullPETCTAsymmetricFusionV2(
        channels=(16, 24, 32, 48), pet_dims=(16, 16, 24, 32), heads=4,
        grid_cap=16, text_dim=dim, text_embeddings=emb, text_encoder='biomedbert').eval()
    cf = [torch.randn(2, c, s, s) for c, s in zip((16, 24, 32, 48), (32, 16, 8, 4))]
    pf = [torch.randn(2, c, s, s) for c, s in zip((16, 24, 32, 48), (32, 16, 8, 4))]
    with torch.no_grad():
        fused, deltas = m.forward_with_delta(cf, pf, state='full')
    assert len(fused) == 4 and all(f.shape == c.shape for f, c in zip(fused, cf))
    assert m.get_extra_state()['text_encoder'] == 'biomedbert'
