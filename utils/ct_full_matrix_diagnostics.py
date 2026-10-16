"""Sampled read statistics and post-commit matrix-bank snapshots."""
import csv
import os
import torch

CONTEXT = ['epoch','batch','global_batch_step','successful_steps','event','scale','bank']
BANK_FIELDS = CONTEXT + ['enabled','valid','pool_size','capacity','accepted','updates',
    'refresh_attempts','refresh_failures','last_status','read_available','gate_mean',
    'correction_rms_ratio','transform_rms_ratio','attention_entropy','class_attention_mass']
CLUSTER_FIELDS = CONTEXT + ['slot','valid','support','key_norm','matrix_norm',
    'fit_mse_in_pool','identity_mse_in_pool','ridge_lambda','applied_matrix_drift',
    'read_attention_mean','read_top1_fraction']


class MatrixCSVLogger:
    def __init__(self, directory, start_epoch=1):
        self.schemas = {'matrix_banks.csv':BANK_FIELDS, 'matrix_slots.csv':CLUSTER_FIELDS}
        self.paths = {}
        os.makedirs(directory, exist_ok=True)
        for name, fields in self.schemas.items():
            path = os.path.join(directory, name); self.paths[name] = path
            retained = []
            if os.path.isfile(path):
                with open(path, newline='') as f:
                    reader = csv.DictReader(f)
                    if reader.fieldnames != fields: raise ValueError('matrix CSV schema mismatch')
                    retained = [r for r in reader if int(r['epoch']) < start_epoch]
            with open(path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader(); writer.writerows(retained)

    @torch.no_grad()
    def snapshot(self, task, epoch, batch, event):
        rows, slots = [], []
        for i, s in enumerate(task.model.matrix.scales):
            read = task.matrix_read_stats[i] if task.matrix_read_stats is not None else {}
            start = 0
            for name, bank in zip(('foreground','background'), s.banks):
                ctx = dict(epoch=epoch, batch=batch, global_batch_step=task.global_batch_step,
                    successful_steps=int(task.model.matrix.update_steps), event=event, scale=f's{i+1}', bank=name)
                valid = bool(bank.valid)
                row = dict(ctx, enabled=task.model.matrix_enabled, valid=valid,
                    pool_size=int(bank.size), capacity=bank.capacity, accepted=int(bank.accepted),
                    updates=int(bank.updates), refresh_attempts=int(bank.refresh_attempts),
                    refresh_failures=int(bank.refresh_failures), last_status=bank.refresh_stats['status'],
                    read_available=read.get('available',False))
                for key in ('gate_mean','correction_rms_ratio','transform_rms_ratio','attention_entropy'):
                    row[key] = read.get(key)
                attention = read.get('attention_mean', [])
                top1 = read.get('top1_fraction', [])
                if valid and attention: row['class_attention_mass'] = sum(attention[start:start+bank.clusters])
                rows.append(row)
                for j in range(bank.clusters):
                    slots.append(dict(ctx, slot=j, valid=valid, support=int(bank.support[j]),
                        key_norm=float(bank.keys[j].norm()), matrix_norm=float(bank.transforms[j].norm()),
                        fit_mse_in_pool=float(bank.fit_error[j]), identity_mse_in_pool=float(bank.identity_error[j]),
                        ridge_lambda=float(bank.ridge_lambda[j]), applied_matrix_drift=float(bank.matrix_drift[j]),
                        read_attention_mean=attention[start+j] if valid and attention else None,
                        read_top1_fraction=top1[start+j] if valid and top1 else None))
                if valid: start += bank.clusters
        for filename, data in (('matrix_banks.csv',rows),('matrix_slots.csv',slots)):
            with open(self.paths[filename], 'a', newline='') as f:
                csv.DictWriter(f, fieldnames=self.schemas[filename]).writerows(data)
        print(f'[matrix epoch {epoch} batch {batch}] post_commit banks; read stats are sampled pre-update', flush=True)
