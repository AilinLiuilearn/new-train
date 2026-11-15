import torch
import pytest

from models.components.tacs_fusion import TACSFusion, STAGE_CHANNELS


def make_features(batch=2, image_size=64):
    sizes = tuple(image_size // factor for factor in (4, 8, 16, 32))
    ct = [torch.randn(batch, c, s, s, requires_grad=True)
          for c, s in zip(STAGE_CHANNELS, sizes)]
    pet = [torch.randn_like(x, requires_grad=True) for x in ct]
    return ct, pet


@pytest.mark.parametrize("ratio", [0.25, 1.0])
def test_tacs_four_scales_shape_and_exact_add_initialization(ratio):
    fusion = TACSFusion(image_size=64, channel_ratio=ratio)
    ct, pet = make_features()
    outputs = fusion(ct, pet)
    assert len(outputs) == 4
    if ratio == 1.0:
        assert all(stage.channel_attention is None for stage in fusion.stages)
    else:
        assert all(stage.channel_attention is not None for stage in fusion.stages)
    for out, c, p in zip(outputs, ct, pet):
        assert out.shape == c.shape
        assert torch.isfinite(out).all()
        torch.testing.assert_close(out, c + p, atol=0, rtol=0)


def test_tacs_text_embedding_is_optional_and_loaded(tmp_path):
    path = tmp_path / "text.pt"
    torch.save({"embedding": torch.randn(512), "prompt": "test"}, path)
    fusion = TACSFusion(
        image_size=64, channel_ratio=0.25,
        text_embedding_path=str(path), use_text=True,
    )
    ct, pet = make_features()
    outputs = fusion(ct, pet)
    assert len(outputs) == 4
    assert fusion.text_embedding.shape == (512,)


def test_tacs_backward_opens_residual_head_then_router():
    torch.manual_seed(2023)
    fusion = TACSFusion(image_size=64, channel_ratio=0.25)
    ct, pet = make_features()
    optimizer = torch.optim.AdamW(fusion.parameters(), lr=1e-4)
    target = [torch.randn_like(x) for x in ct]

    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        outputs = fusion(ct, pet)
        loss = sum((x - y).square().mean() for x, y in zip(outputs, target))
        loss.backward()
        for stage in fusion.stages:
            assert stage.output_projection.weight.grad is not None
            assert torch.isfinite(stage.output_projection.weight.grad).all()
            if step == 1:
                channel_grad = float(stage.channel_attention.in_proj_weight.grad.abs().sum())
                restore_grad = float(stage.ct_channel_restore.weight.grad.abs().sum())
                assert channel_grad > 0
                assert restore_grad > 0
                if stage.router is not None:
                    router_grad = sum(
                        float(p.grad.abs().sum())
                        for p in stage.router.parameters()
                        if p.grad is not None
                    )
                    assert router_grad > 0
        optimizer.step()


def test_tacs_diagnostics_and_selected_channel_counts():
    fusion = TACSFusion(image_size=64, channel_ratio=0.25)
    ct, pet = make_features()
    fusion.train()
    fusion.reset_diagnostics()
    fusion(ct, pet)
    stats = fusion.pop_diagnostics()
    assert stats["tacs_selected_channels_s1"] == 16
    assert stats["tacs_selected_channels_s2"] == 32
    assert stats["tacs_selected_channels_s3"] == 80
    assert stats["tacs_selected_channels_s4"] == 128
    assert all(torch.isfinite(torch.tensor(v)) for v in stats.values())


def test_tacs_rejects_invalid_text_configuration():
    with pytest.raises(ValueError, match="requires"):
        TACSFusion(image_size=64, use_text=True)
