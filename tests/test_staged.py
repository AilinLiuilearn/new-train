"""Contract tests use REAL timm/HF encoders, aligners and GN decoders, no toy substitutes."""
import gc
import json
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
import torch

import run_staged as runner
from datasets.pclt20k_seg import PCLT20KSegDataset, _records_from_ids
from datasets.staged import HalfMissingBatches
from models.staged_petct import StagedPETCT
from utils.run_common import module_grad_norm
from utils.seg_losses import BCEDiceLoss


torch.set_num_threads(1)


def test_transfer_exact_legacy_ema_and_independent_decoders(tmp_path):
    torch.manual_seed(19)
    stage1 = StagedPETCT(1).eval()
    ct = torch.randn(2, 3, 64, 64)
    with torch.no_grad(): expected = stage1(ct)['logits']
    state = {k: v.clone() for k, v in stage1.state_dict().items()}
    # Deliberately different raw vs EMA: auto must use the selected EMA weights.
    raw = {k: v+1 if v.is_floating_point() else v for k, v in state.items()}
    path = tmp_path/'legacy.pth.tar'
    torch.save({'model': raw, 'model_ema': state, 'eval_weights': 'ema', 'epoch': 17,
                'config': {'baseline': 'ct_only', 'ct_backbone': 'convnextv2_nano',
                           'image_size_2d': 64, 'norm_mode': 'imagenet'}}, path)
    del stage1, raw
    gc.collect()
    stage2 = StagedPETCT(2).eval()
    cfg = runner.parser().parse_args(['--stage', '2', '--image_size_2d', '64'])
    report = runner.import_stage1(stage2, path, cfg, {})
    assert report['weights'] == 'ema' and report['all_parameters_trainable']
    with torch.no_grad(): actual = stage2(ct, mode='missing')['logits']
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    for a, b in zip(stage2.decoder_ct.parameters(), stage2.decoder_full.parameters()):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
        assert a.data_ptr() != b.data_ptr()
    for prefix in ['enc_ct', 'ct_align']:
        for key, value in getattr(stage2, prefix).state_dict().items():
            torch.testing.assert_close(value, state[prefix+'.'+key], atol=0, rtol=0)
    broken = dict(state); broken.pop(next(iter(broken)))
    with pytest.raises(ValueError, match='exact clean'): stage2.import_stage1_state(broken)
    broken = dict(state); broken['decoder.seg_head.weight'] = torch.zeros(2, 64, 1, 1)
    with pytest.raises(ValueError, match='shape mismatch'): stage2.import_stage1_state(broken)
    cfg.norm_mode = 'cipa'
    with pytest.raises(ValueError, match='protocol mismatch'): runner.import_stage1(stage2, path, cfg, {})
    with pytest.raises(ValueError, match='eval_weights'): runner.selected(state)
    torch.testing.assert_close(runner.selected(state, 'raw')[0]['decoder.seg_head.weight'],
                               state['decoder.seg_head.weight'])


def test_real_paths_exact_loss_gradients_and_pet_isolation():
    model = StagedPETCT(2).eval()
    ct = torch.randn(4, 3, 64, 64)
    pet = torch.randn_like(ct)
    flags = torch.tensor([True, False, False, True])
    pet[~flags] = float('nan')
    y = (torch.rand(4, 1, 64, 64) > .97).float()
    batch = {'ct': ct, 'pet': pet, 'mask': y, 'is_full': flags}
    observed = []
    hook = model.enc_pet.register_forward_pre_hook(lambda m, args: observed.append(len(args[0])))
    criterion = BCEDiceLoss()
    loss, stats = runner.forward_loss(model, batch, 2, torch.device('cpu'), criterion)
    assert observed == [2] and stats['ct_samples'] == 4 and stats['full_samples'] == 2
    torch.testing.assert_close(loss.detach(), torch.tensor(.5*(stats['ct_loss']+stats['full_loss'])))
    loss.backward()
    for name, module in model.named_children():
        if list(module.parameters()): assert module_grad_norm(module) > 0, name
    tracked = {name: next(m.parameters()).detach().clone() for name, m in model.named_children() if list(m.parameters())}
    torch.optim.AdamW(model.parameters(), lr=8e-5).step()
    assert all(not torch.equal(tracked[name], next(getattr(model, name).parameters())) for name in tracked)
    hook.remove()
    model.zero_grad(set_to_none=True)
    def forbidden(*unused): raise AssertionError('PET/Full module used on Missing')
    hooks = [m.register_forward_pre_hook(forbidden) for m in
             [model.enc_pet, model.pet_align, model.fusion, model.decoder_full]]
    a = model(ct[:2], mode='missing')['logits']
    b = model(ct[:2], torch.full_like(ct[:2], float('nan')), mode='missing')['logits']
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    a.square().mean().backward()
    assert all(p.grad is None for p in model.enc_pet.parameters())
    assert all(p.grad is None for p in model.decoder_full.parameters())
    for hook in hooks: hook.remove()
    with pytest.raises(ValueError, match='exactly half'):
        model(ct, pet[:1], mode='mixed', full_indices=torch.tensor([0]))


def make_pngs(root, train=4):
    names = []
    for i in range(train+2):
        case = f'case{i}'
        folder = root/case; folder.mkdir(parents=True)
        name = f'{case}_001'; names.append(name)
        rng = np.random.default_rng(i)
        ct = rng.integers(0, 256, (64, 64), dtype=np.uint8)
        mask = np.zeros((64, 64), dtype=np.uint8); mask[20:32, 23:35] = 255
        for suffix, img in [('CT', ct), ('PET', np.flip(ct, 0)), ('mask', mask)]:
            Image.fromarray(img).save(folder/f'{name}_{suffix}.png')
    for split, ids in [('train_original', names[:train]), ('val', [names[-2]]), ('test', [names[-1]])]:
        (root/f'{split}.txt').write_text('\n'.join(ids)+'\n')
    return names


def test_balanced_sampling_missing_io_and_mask_labels(tmp_path):
    batches = HalfMissingBatches(10, 4, 2023)
    first = list(batches)
    batches.set_epoch(2); second = list(batches)
    assert len(first) == 2
    for a, b in zip(first, second):
        assert sum(flag for _, flag in a) == 2
        assert [idx for idx, _ in a] == [idx for idx, _ in b]
        assert all(f != g for (_, f), (_, g) in zip(a, b))
    names = make_pngs(tmp_path)
    records = _records_from_ids(tmp_path, names[:4])
    # Remove PET: simulated Missing must still work; Full must fail.
    Path(records[0]['pet_path']).unlink()
    ds = PCLT20KSegDataset(records, image_size=64, ct_only=False)
    assert not ds[(0, False)]['is_full']
    assert ds[(0, False)]['pet'].count_nonzero() == 0
    with pytest.raises(FileNotFoundError): ds[(0, True)]
    # 0/1 masks must not silently become all background.
    maskpath = Path(records[0]['mask_path'])
    Image.fromarray((np.array(Image.open(maskpath)) > 0).astype(np.uint8)).save(maskpath)
    assert ds[(0, False)]['mask'].sum() == 144
    # Real CIPA augmentations keep a valid binary mask/tensor contract.
    ds.train = True
    for seed in range(8):
        np.random.seed(seed)
        sample = ds[(1, True)]
        assert sample['ct'].shape == (3, 64, 64)
        assert set(sample['mask'].unique().tolist()) <= {0., 1.}
    with pytest.raises(ValueError): HalfMissingBatches(10, 3, 0)


def test_real_two_stage_png_pipeline_resume_ema_and_reevaluation(tmp_path, monkeypatch):
    root = tmp_path/'data'; root.mkdir(); make_pngs(root)
    common = ['--root', str(root), '--image_size_2d', '64', '--batch_size', '2',
              '--num_workers', '0', '--epochs', '2', '--cosine_warmup', '0',
              '--ema_start_epoch', '0', '--device', 'cpu', '--no_pretrained',
              '--aug_mode', 'none', '--log_interval', '20', '--hd95_mode', 'surface']
    stage1 = tmp_path/'ct'
    runner.main(['--stage', '1', '--output', str(stage1), *common])
    full = tmp_path/'joint'
    stage2_args = ['--stage', '2', '--stage1_checkpoint', str(stage1/'best.pt'), *common]
    result = runner.main([*stage2_args, '--output', str(full)])
    rows = [json.loads(x) for x in (full/'train.jsonl').read_text().splitlines()]
    assert len(rows) == 2 and all(r['full_samples'] == r['missing_samples'] == 2 for r in rows)
    assert all(r['ct_samples'] == 4 for r in rows)
    assert result['metrics']['joint_dice'] == .5*(result['metrics']['full']['dice']+result['metrics']['missing']['dice'])
    transfer = json.loads((full/'transfer.json').read_text())
    assert transfer['split_hash_verified'] and transfer['weights'] == 'ema'
    expected = runner.load(full/'last.pt')
    interrupted = tmp_path/'interrupted'
    original_save = runner.atomic_save
    class StopAfterEpoch(Exception): pass
    def stop_after_last(value, path):
        original_save(value, path)
        if Path(path).name == 'last.pt' and value['epoch'] == 1: raise StopAfterEpoch()
    monkeypatch.setattr(runner, 'atomic_save', stop_after_last)
    with pytest.raises(StopAfterEpoch): runner.main([*stage2_args, '--output', str(interrupted)])
    monkeypatch.setattr(runner, 'atomic_save', original_save)
    gc.collect()
    runner.main(['--resume', str(interrupted/'last.pt'), '--device', 'cpu'])
    actual = runner.load(interrupted/'last.pt')
    assert expected['global_step'] == actual['global_step'] == 4
    resumed_rows = [json.loads(x) for x in (interrupted/'train.jsonl').read_text().splitlines()]
    assert [r['epoch'] for r in resumed_rows] == [1, 2]
    for group in ['model', 'model_ema']:
        for key, value in expected[group].items():
            torch.testing.assert_close(value, actual[group][key], atol=0, rtol=0)
    del expected, actual
    gc.collect()
    evaluated = runner.main(['--evaluate', str(full/'best.pt'), '--device', 'cpu'])
    assert evaluated == result
    # Missing-only evaluation must not require PET files anywhere in the dataset.
    for path in root.rglob('*_PET.png'): path.unlink()
    missing = runner.main(['--evaluate', str(full/'best.pt'), '--eval_mode', 'missing', '--device', 'cpu'])
    assert missing['metrics']['missing'] == result['metrics']['missing']
    with pytest.raises(ValueError, match='unsupported overrides'):
        runner.main(['--resume', str(full/'last.pt'), '--learning_rate', '1e-3'])
    # Split content hash detects tampering independently of split file names.
    (root/'test.txt').write_text((root/'test.txt').read_text()+'\n')
    with pytest.raises(ValueError, match='split contents changed'):
        runner.main(['--evaluate', str(full/'best.pt'), '--eval_mode', 'missing', '--device', 'cpu'])


def test_official_metrics_and_protocol_guards():
    from utils.metrics_seg import SegmentationMetricsCIPA
    y = torch.zeros(1, 1, 512, 512); y[:, :, 240:270, 245:265] = 1
    z = y*20-10
    metric = SegmentationMetricsCIPA(hd95_mode='cipa_official')
    metric.update(z, y)
    result = metric.compute()
    assert result['dice'] == 1. and result['hd95'] == 0.
    cfg = runner.parser().parse_args(['--stage', '2', '--output', 'unused', '--batch_size', '3'])
    with pytest.raises(ValueError, match='even batch'): runner.validate(cfg)
    cfg.batch_size = 2; cfg.image_size_2d = 64
    with pytest.raises(ValueError, match='requires image_size'): runner.validate(cfg)
    cfg.hd95_mode = 'surface'; cfg.eval_mode = 'missing'
    with pytest.raises(ValueError, match='evaluation-only'): runner.validate(cfg)
