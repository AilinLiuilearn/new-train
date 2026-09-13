# -*- coding: utf-8 -*-
"""train_logger 列对齐合同测试：header 注册、逐 key 取值、错位拒绝。"""
import csv
import os

import pytest

from utils.train_logger import append_epoch_log, get_log_headers, init_train_log


def _val():
    return {'total_loss': 1.0, 'dice': 0.5, 'iou': 0.4, 'acc': 0.9,
            'acc_pixel': 0.9, 'hd95': 5.0}


def test_header_registration_and_column_alignment(tmp_path):
    log = str(tmp_path / 'train_log.csv')
    headers = init_train_log(log, extra_headers=['z_last', 'a_first', 'm_mid'])
    assert headers[:10] == ['epoch', 'train_loss', 'val_loss', 'val_dice', 'val_iou',
                            'val_acc', 'val_acc_pixel', 'val_hd95', 'lr', 'grad_norm']
    assert get_log_headers(log) == headers
    # 插入顺序与 header 顺序故意相反：必须按列名写入，不能错列。
    append_epoch_log(log, 1, 0.5, _val(), lr=1e-4, grad_norm=2.0,
                     extra_metrics={'m_mid': 30.0, 'z_last': 10.0, 'a_first': 20.0})
    with open(log, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert float(rows[0]['z_last']) == pytest.approx(10.0)
    assert float(rows[0]['a_first']) == pytest.approx(20.0)
    assert float(rows[0]['m_mid']) == pytest.approx(30.0)
    assert int(rows[0]['epoch']) == 1


def test_shuffled_dict_order_still_aligned(tmp_path):
    log = str(tmp_path / 'train_log.csv')
    init_train_log(log, extra_headers=['route_ct', 'route_pet', 'counts_s1'])
    for order in ({'counts_s1': '1,2', 'route_pet': 0.2, 'route_ct': 0.8},
                  {'route_ct': 0.8, 'counts_s1': '1,2', 'route_pet': 0.2}):
        append_epoch_log(log, 1, 0.5, _val(), extra_metrics=dict(order))
    with open(log, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    for r in rows:
        assert float(r['route_ct']) == pytest.approx(0.8)
        assert float(r['route_pet']) == pytest.approx(0.2)
        assert r['counts_s1'] == '1,2'


def test_unknown_field_raises(tmp_path):
    log = str(tmp_path / 'train_log.csv')
    init_train_log(log, extra_headers=['a'])
    with pytest.raises(ValueError):
        append_epoch_log(log, 1, 0.5, _val(), extra_metrics={'a': 1.0, 'nope_typo': 2.0})


def test_missing_field_empty_with_warning(tmp_path):
    log = str(tmp_path / 'train_log.csv')
    init_train_log(log, extra_headers=['a', 'b'])
    with pytest.warns(UserWarning):
        append_epoch_log(log, 1, 0.5, _val(), extra_metrics={'a': 1.0})
    with open(log, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    assert float(rows[0]['a']) == pytest.approx(1.0)
    assert rows[0]['b'] == ''


def test_append_without_init_raises(tmp_path):
    log = str(tmp_path / 'never_init.csv')
    with pytest.raises(RuntimeError):
        append_epoch_log(log, 1, 0.5, _val())


def test_readable_log_keeps_format(tmp_path):
    log = str(tmp_path / 'train_log.csv')
    init_train_log(log, extra_headers=['route_ct'])
    append_epoch_log(log, 2, 0.5, _val(), lr=1e-4, grad_norm=1.0,
                     extra_metrics={'route_ct': 0.75})
    readable = os.path.splitext(log)[0] + '_readable.txt'
    text = open(readable, encoding='utf-8').read()
    assert 'Epoch 2' in text and 'route_ct' in text and '0.750000' in text
