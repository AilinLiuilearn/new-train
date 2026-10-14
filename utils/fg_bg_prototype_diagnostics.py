"""Read-only per-scale losses, benefit selection and FG/BG library snapshots."""
import csv
import os

import torch

from utils.prototype_diagnostics import CONTEXT, BANK_FIELDS as OLD_BANK_FIELDS, GAIN_COUNTERS, bank_snapshot, ratio


BANK_FIELDS = [field for field in OLD_BANK_FIELDS if not field.startswith('read_')
    and field not in ('spatial_fg_patches_epoch', 'spatial_bg_patches_epoch', 'channel_qk_norm',
                      'channel_logit_scale', 'local_correction', 'correction_alpha')] + [
    'gain_mode', 'adaptive_sampling', 'sampling_decay', 'sampling_updates',
    'sampling_fg_mass', 'sampling_bg_mass', 'sampling_fg_probability',
    'available_patches_epoch', 'allocated_patches_epoch', 'logit_scale', 'correction_alpha',
    'read_sample_global_step', 'read_available', 'read_query_count',
    'read_class_attention_mass', 'read_joint_attention_entropy_normalized',
    'read_correction_rms_ratio', 'read_gate_mean', 'read_q_norm_mean', 'read_k_norm_mean',
    'read_logit_min', 'read_logit_max', 'read_logit_span_mean']
CLUSTER_FIELDS = CONTEXT + ['scale', 'bank', 'enabled', 'valid', 'cluster_id',
    'pool_count', 'pool_fraction', 'mean_member_distance', 'center_norm', 'last_success_count',
    'read_sample_global_step', 'read_attention_mean', 'read_top1_fraction']
GAIN_FIELDS = CONTEXT + ['scale', 'gain_mode', 'prototype_enabled', 'useful_image_fraction'] + GAIN_COUNTERS + [
    f'{label}_{name}' for label in ('fg', 'bg')
    for name in ('selected_fraction', 'mean_selected_gain', 'selected_weight_fraction')
] + ['full_fnr', 'reference_fnr', 'full_fpr', 'reference_fpr']
TRAIN_FIELDS = CONTEXT + ['scale', 'deep_supervision_enabled', 'supervised_batches',
    'loss_full_mean', 'loss_missing_mean', 'loss_ct_reference_mean', 'weighted_loss_mean']


class ForegroundBackgroundCSVLogger:
    def __init__(self, directory, start_epoch=1):
        self.schemas = dict(zip(('prototype_banks.csv', 'prototype_clusters.csv',
                                 'prototype_gain.csv', 'scale_train.csv'),
                                (BANK_FIELDS, CLUSTER_FIELDS, GAIN_FIELDS, TRAIN_FIELDS)))
        self.paths = {}
        os.makedirs(directory, exist_ok=True)
        for filename, fields in self.schemas.items():
            path = os.path.join(directory, filename); self.paths[filename] = path
            rows = []
            if os.path.isfile(path):
                with open(path, newline='') as stream:
                    reader = csv.DictReader(stream)
                    if reader.fieldnames != fields: raise ValueError(f'diagnostics schema mismatch: {path}')
                    rows = [r for r in reader if int(r['epoch']) < start_epoch]
            with open(path, 'w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader(); writer.writerows(rows)

    def _append(self, filename, rows):
        with open(self.paths[filename], 'a', newline='') as stream:
            csv.DictWriter(stream, fieldnames=self.schemas[filename]).writerows(rows)

    @torch.no_grad()
    def snapshot(self, task, epoch, batch, event):
        prototype, enabled = task.model.prototype, task.model.prototype_enabled
        context = dict(epoch=epoch, batch=batch, global_batch_step=task.global_batch_step,
                       successful_steps=int(prototype.update_steps), event=event)
        stats = task.prototype_epoch_stats
        totals = dict(zip(stats, torch.stack(list(stats.values())).cpu().tolist())) if stats else {}
        bank_rows, clusters_rows, gain_rows, train_rows = [], [], [], []
        for i, scale in enumerate(prototype.scales):
            level = f'd{4-i}'
            gain = {name: totals.get(name if name.endswith('_batches') else f'{level}_{name}', 0)
                    for name in GAIN_COUNTERS}
            gain.update(context, scale=level, gain_mode=task.model.gain_mode, prototype_enabled=enabled,
                        useful_image_fraction=ratio(gain['useful_full_images'], gain['full_images']))
            total_weight = gain['fg_weight_sum'] + gain['bg_weight_sum']
            for label in ('fg', 'bg'):
                gain[f'{label}_selected_fraction'] = ratio(gain[f'{label}_selected'], gain[f'{label}_pixels'])
                gain[f'{label}_mean_selected_gain'] = ratio(gain[f'{label}_gain_sum'], gain[f'{label}_selected'])
                gain[f'{label}_selected_weight_fraction'] = ratio(gain[f'{label}_weight_sum'], total_weight)
            for label in ('full', 'reference'):
                gain[f'{label}_fnr'] = ratio(gain[f'{label}_fn'], gain['fg_pixels'])
                gain[f'{label}_fpr'] = ratio(gain[f'{label}_fp'], gain['bg_pixels'])
            gain_rows.append(gain)
            count = totals.get(f'{level}_supervised_batches', 0)
            losses = [ratio(totals.get(f'{level}_loss_{label}', 0), count) for label in ('full', 'missing', 'ct')]
            train_rows.append(dict(context, scale=level, deep_supervision_enabled=task.model.use_deep_supervision,
                supervised_batches=count, loss_full_mean=losses[0], loss_missing_mean=losses[1],
                loss_ct_reference_mean=losses[2], weighted_loss_mean=(.5 * losses[0] + .5 * losses[1]
                    + task.ct_aux_weight * losses[2]) if count else None))
            for label, bank in zip(('foreground', 'background'), scale.banks):
                row, clusters = bank_snapshot(bank, enabled)
                if enabled and int(prototype.update_steps) <= prototype.warmup_steps: row['status'] = 'warmup'
                mass = scale.sampling_mass.cpu().tolist()
                row.update(context, scale=level, bank=label, gain_mode=task.model.gain_mode,
                    generated_epoch=totals.get(f'{level}_{label}_generated', 0),
                    written_epoch=totals.get(f'{level}_{label}_written', 0),
                    available_patches_epoch=totals.get(f'{level}_{label}_available_patches', 0),
                    allocated_patches_epoch=totals.get(f'{level}_{label}_allocated_patches', 0),
                    adaptive_sampling=scale.adaptive_sampling, sampling_decay=scale.sampling_decay,
                    sampling_updates=int(scale.sampling_updates), sampling_fg_mass=mass[0], sampling_bg_mass=mass[1],
                    sampling_fg_probability=float(scale.sampling_probability()[0]),
                    logit_scale=scale.logit_scale, correction_alpha=scale.correction_alpha)
                sampled = task.prototype_read_stats
                if sampled is not None:
                    read = sampled['scales'][i].get(label, {})
                    joint = sampled['scales'][i].get('joint', {})
                    row.update(read_sample_global_step=sampled['global_batch_step'],
                               read_available=read.get('available', False))
                    if read.get('available'):
                        row.update(read_query_count=joint['query_count'],
                            read_class_attention_mass=float(read['class_mass']),
                            read_joint_attention_entropy_normalized=float(joint['attention_entropy']))
                        for name in ('correction_rms_ratio', 'gate_mean', 'q_norm_mean', 'k_norm_mean',
                                     'logit_min', 'logit_max', 'logit_span_mean'):
                            row[f'read_{name}'] = float(joint[name])
                        attention, top1 = read['attention_mean'].cpu().tolist(), read['top1_fraction'].cpu().tolist()
                        for k, cluster in enumerate(clusters):
                            cluster.update(read_sample_global_step=sampled['global_batch_step'],
                                           read_attention_mean=attention[k], read_top1_fraction=top1[k])
                bank_rows.append(row)
                for cluster in clusters:
                    cluster.update(context, scale=level, bank=label); clusters_rows.append(cluster)
        self._append('prototype_banks.csv', bank_rows)
        self._append('prototype_clusters.csv', clusters_rows)
        self._append('prototype_gain.csv', gain_rows)
        self._append('scale_train.csv', train_rows)
        print(f'[fg_bg prototype epoch {epoch} batch {batch}] epoch_cumulative '
              f'successful_steps={context["successful_steps"]}', flush=True)
        for gain, rows in zip(gain_rows, (bank_rows[j:j+2] for j in range(0, 8, 2))):
            descriptions = [f'{r["bank"]}={r["status"]} pool={r["pool_size"]}/{r["capacity"]} '
                            f'updates={r["updates_total"]}' for r in rows]
            print(f'  {gain["scale"]} selected FG={gain["fg_selected"]:.0f} BG={gain["bg_selected"]:.0f}; '
                  + '; '.join(descriptions), flush=True)
