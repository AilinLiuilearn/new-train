"""Startup errors must not create files or discard any training results."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import run_full_lsconv_fusion as entry


def cfg(tmp_path):
    return SimpleNamespace(checkpoint_root=str(tmp_path), hash='retry', retry_failed_start=True)


def seed_failed_run(tmp_path):
    c = cfg(tmp_path)
    folder = tmp_path / 'MDT' / c.hash
    folder.mkdir(parents=True)
    (folder / 'config_args.json').write_text(json.dumps(dict(
        experiment_type='full_lsconv_fusion', model_arch='full_lsconv_fusion', hash=c.hash)))
    (folder / 'split_summary.json').write_text('{}')
    (folder / 'train_log.csv').write_text('epoch,train_loss\n')
    (folder / 'train_log_readable.txt').write_text('Training Log\n' + '=' * 64 + '\n')
    return c, folder


def test_cuda_failure_before_directory_creation(monkeypatch, tmp_path):
    monkeypatch.setattr(entry.torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(sys, 'argv', ['x', '--hash', 'blocked', '--checkpoint_root', str(tmp_path)])
    with pytest.raises(RuntimeError, match='No experiment files'):
        entry.FullLSConvConfig.parse_arguments()
    assert not (tmp_path / 'MDT').exists()


def test_cpu_smoke_must_explicitly_select_supported_backend(monkeypatch):
    monkeypatch.setattr(entry.torch.cuda, 'is_available', lambda: False)
    entry.check_runtime(False, True, 'torch')
    with pytest.raises(RuntimeError):
        entry.check_runtime(False, True, 'triton')


def test_retry_archives_startup_files_without_deleting(tmp_path):
    c, folder = seed_failed_run(tmp_path)
    before = {p.name: p.read_bytes() for p in folder.iterdir()}
    entry.prepare_run_directory(c)
    assert not folder.exists()
    backups = list(folder.parent.glob('retry.failed-start-*'))
    assert len(backups) == 1
    assert before == {p.name: p.read_bytes() for p in backups[0].iterdir()}


@pytest.mark.parametrize('artifact', ['ckpt.last.pth.tar', 'ckpt.best_full.pth.tar',
                                     'final_test_full.json', 'unexpected.txt', 'epoch'])
def test_retry_refuses_training_artifacts(tmp_path, artifact):
    c, folder = seed_failed_run(tmp_path)
    if artifact == 'epoch':
        with (folder / 'train_log.csv').open('a') as f:
            f.write('1,0.7\n')
    else:
        (folder / artifact).write_text('preserve')
    with pytest.raises(FileExistsError):
        entry.prepare_run_directory(c)
    assert folder.exists()


def test_retry_is_opt_in(tmp_path):
    c, folder = seed_failed_run(tmp_path)
    c.retry_failed_start = False
    with pytest.raises(FileExistsError):
        entry.prepare_run_directory(c)
    assert folder.exists()
