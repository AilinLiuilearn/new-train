"""Read-only, bounded-pool training diagnostics; never used by the model.

Snapshots are cumulative within the current epoch, not interval deltas.
Cluster assignments describe the current FIFO against the deployed EMA centers.
No feature tensors or patient identifiers are written to disk.
"""
import csv
import math
import os

import torch


CONTEXT = ['epoch', 'batch', 'global_batch_step', 'successful_steps', 'event']
BANK_FIELDS = CONTEXT + [
    'scale', 'bank', 'enabled', 'valid', 'status', 'pool_size', 'capacity', 'pool_fill',
    'min_candidates', 'clusters', 'accepted_total', 'generated_epoch', 'written_epoch',
    'spatial_fg_patches_epoch', 'spatial_bg_patches_epoch', 'updates_total',
    'refresh_attempts', 'refresh_insufficient_pool', 'refresh_insufficient_unique',
    'refresh_empty_cluster', 'last_attempt_status', 'last_success_iterations',
    'last_success_pool_size', 'last_success_retained', 'ema_decay',
    'last_raw_center_drift', 'last_applied_center_drift',
    'occupied_clusters', 'assignment_entropy_normalized', 'max_cluster_fraction',
    'mean_nearest_distance', 'mean_nearest_margin', 'mean_center_norm', 'min_center_distance',
    'read_sample_global_step', 'read_available', 'read_query_count',
    'read_attention_entropy_normalized', 'read_correction_rms_ratio', 'read_spatial_gate_mean',
    'read_channel_gate_mean', 'read_q_norm_mean', 'read_k_norm_mean',
    'read_logit_min', 'read_logit_max', 'read_logit_span_mean',
    'read_total_correction_rms_ratio', 'channel_qk_norm', 'channel_logit_scale',
    'local_correction', 'correction_alpha']
CLUSTER_FIELDS = CONTEXT + [
    'scale', 'bank', 'enabled', 'valid', 'cluster_id', 'pool_count', 'pool_fraction',
    'mean_member_distance', 'center_norm', 'last_success_count',
    'read_sample_global_step', 'read_attention_mean', 'read_top1_fraction']
GAIN_COUNTERS = ['attempted_batches', 'successful_batches', 'skipped_batches', 'warmup_batches',
                 'full_images', 'useful_full_images', 'fg_pixels', 'bg_pixels',
                 'fg_selected', 'bg_selected', 'fg_full_wrong', 'bg_full_wrong',
                 'fg_below_gain', 'bg_below_gain', 'fg_gain_sum', 'bg_gain_sum',
                 'fg_weight_sum', 'bg_weight_sum', 'full_fn', 'reference_fn',
                 'full_fp', 'reference_fp']
GAIN_FIELDS = CONTEXT + ['prototype_enabled', 'useful_image_fraction'] + GAIN_COUNTERS + [
    f'{label}_{name}' for label in ('fg', 'bg')
    for name in ('selected_fraction', 'mean_selected_gain', 'selected_weight_fraction')
] + ['full_fnr', 'reference_fnr', 'full_fpr', 'reference_fpr']


def ratio(numerator, denominator):
    # Undefined statistics are blank in CSV, rather than misleading zeroes.
    return numerator / denominator if denominator else None


@torch.no_grad()
def bank_snapshot(bank, enabled):
    """Compute O(capacity*K*D) distances on CPU, at logging times only."""
    size, valid = int(bank.size), bool(bank.valid)
    refresh = bank.refresh_stats
    status = ('disabled' if not enabled else 'ready' if valid else
              'collecting' if size < bank.min_candidates else 'waiting_for_valid_clusters')
    row = dict(enabled=enabled, valid=valid, status=status, pool_size=size,
               capacity=bank.capacity, pool_fill=size / bank.capacity,
               min_candidates=bank.min_candidates, clusters=bank.clusters,
               accepted_total=int(bank.accepted), updates_total=int(bank.updates),
               refresh_attempts=refresh['attempts'],
               refresh_insufficient_pool=refresh['insufficient_pool'],
               refresh_insufficient_unique=refresh['insufficient_unique'],
               refresh_empty_cluster=refresh['empty_cluster'],
               last_attempt_status=refresh['last_status'],
               last_success_iterations=refresh.get('last_iterations'),
               last_success_pool_size=refresh.get('last_pool_size'),
               last_success_retained=refresh.get('last_retained'), ema_decay=bank.decay,
               last_raw_center_drift=refresh.get('last_raw_drift'),
               last_applied_center_drift=refresh.get('last_applied_drift'))
    clusters = [dict(cluster_id=k, enabled=enabled, valid=valid) for k in range(bank.clusters)]
    counts_at_refresh = refresh.get('last_cluster_counts')
    for k, cluster in enumerate(clusters):
        cluster['last_success_count'] = counts_at_refresh[k] if counts_at_refresh is not None else None
    if not enabled or not valid:
        return row, clusters
    centers = bank.centers.detach().cpu().double()
    norms = centers.norm(dim=1)
    row['mean_center_norm'] = float(norms.mean())
    # Direct pdist avoids cancellation when centers are nearly equal.
    row['min_center_distance'] = float(torch.pdist(centers).min())
    for k, cluster in enumerate(clusters):
        cluster['center_norm'] = float(norms[k])
    if not size:
        return row, clusters
    values = bank.pool[:size].detach().cpu().double()
    distances = bank._distance(values, centers).sqrt()
    nearest, assignment = distances.min(1)
    counts = torch.bincount(assignment, minlength=bank.clusters)
    fractions = counts.double() / size
    positive = fractions[fractions > 0]
    row.update(occupied_clusters=int((counts > 0).sum()),
               assignment_entropy_normalized=float(-(positive * positive.log()).sum() / math.log(bank.clusters)),
               max_cluster_fraction=float(fractions.max()),
               mean_nearest_distance=float(nearest.mean()),
               mean_nearest_margin=float((distances.topk(2, largest=False).values.diff(dim=1)).mean()))
    for k, cluster in enumerate(clusters):
        count = int(counts[k])
        cluster.update(pool_count=count, pool_fraction=float(fractions[k]),
                       mean_member_distance=float(nearest[assignment == k].mean()) if count else None)
    return row, clusters


class PrototypeCSVLogger:
    def __init__(self, directory, start_epoch=1):
        self.paths = {}
        for filename, fields in (('prototype_banks.csv', BANK_FIELDS),
                                 ('prototype_clusters.csv', CLUSTER_FIELDS),
                                 ('prototype_gain.csv', GAIN_FIELDS)):
            path = os.path.join(directory, filename)
            self.paths[filename] = path
            if os.path.isfile(path):
                with open(path, newline='') as stream:
                    reader = csv.DictReader(stream)
                    # Additive read diagnostics can upgrade old CSVs in place;
                    # missing historical values stay blank, never invented.
                    legacy_bank_fields = BANK_FIELDS[:BANK_FIELDS.index('read_channel_gate_mean')]
                    compatible = (reader.fieldnames == fields or
                                  (filename == 'prototype_banks.csv'
                                   and reader.fieldnames == legacy_bank_fields))
                    if not compatible:
                        raise ValueError(f'prototype diagnostics schema mismatch: {path}')
                    # Remove snapshots from epochs interrupted before checkpoint.
                    rows = [row for row in reader if int(row['epoch']) < start_epoch]
            else:
                rows = []
            with open(path, 'w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)

    def _append(self, filename, fields, rows):
        with open(self.paths[filename], 'a', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writerows(rows)

    @torch.no_grad()
    def snapshot(self, task, epoch, batch, event):
        prototype = task.model.prototype
        enabled = task.model.prototype_enabled
        context = dict(epoch=epoch, batch=batch, global_batch_step=task.global_batch_step,
                       successful_steps=int(prototype.update_steps), event=event)
        # One device-to-host transfer for all epoch reductions.
        epoch_stats = task.prototype_epoch_stats
        totals = (dict(zip(epoch_stats, torch.stack(list(epoch_stats.values())).cpu().tolist()))
                  if epoch_stats else {})
        gain = {name: totals.get(name, 0) for name in GAIN_COUNTERS}
        gain.update(context, prototype_enabled=enabled,
                    useful_image_fraction=ratio(gain['useful_full_images'], gain['full_images']))
        weight_sum = gain['fg_weight_sum'] + gain['bg_weight_sum']
        for label in ('fg', 'bg'):
            gain[f'{label}_selected_fraction'] = ratio(gain[f'{label}_selected'], gain[f'{label}_pixels'])
            gain[f'{label}_mean_selected_gain'] = ratio(gain[f'{label}_gain_sum'], gain[f'{label}_selected'])
            gain[f'{label}_selected_weight_fraction'] = ratio(gain[f'{label}_weight_sum'], weight_sum)
        for label in ('full', 'reference'):
            gain[f'{label}_fnr'] = ratio(gain[f'{label}_fn'], gain['fg_pixels'])
            gain[f'{label}_fpr'] = ratio(gain[f'{label}_fp'], gain['bg_pixels'])
        self._append('prototype_gain.csv', GAIN_FIELDS, [gain])
        bank_rows, cluster_rows, summary = [], [], []
        for i, scale in enumerate(prototype.scales):
            level = f'd{4-i}'
            for label, bank, active in (('spatial', scale.spatial_bank, scale.spatial_enabled),
                                         ('channel', scale.channel_bank, scale.channel_enabled)):
                row, clusters = bank_snapshot(bank, enabled and active)
                if enabled and active and int(prototype.update_steps) <= prototype.warmup_steps:
                    row['status'] = 'warmup'
                row.update(context, scale=level, bank=label,
                           channel_qk_norm=scale.channel_qk_norm,
                           channel_logit_scale=scale.channel_logit_scale,
                           local_correction=scale.local_correction,
                           correction_alpha=scale.correction_alpha,
                           generated_epoch=totals.get(f'{level}_{label}_generated', 0),
                           written_epoch=totals.get(f'{level}_{label}_written', 0),
                           spatial_fg_patches_epoch=totals.get(f'{level}_spatial_fg_patches', 0) if label == 'spatial' else None,
                           spatial_bg_patches_epoch=totals.get(f'{level}_spatial_bg_patches', 0) if label == 'spatial' else None)
                sampled = task.prototype_read_stats
                if sampled is not None:
                    read = sampled['scales'][i].get(label, {})
                    row['read_sample_global_step'] = sampled['global_batch_step']
                    row['read_available'] = read.get('available', False)
                    total_ratio = sampled['scales'][i].get('total_correction_rms_ratio')
                    row['read_total_correction_rms_ratio'] = float(total_ratio) if total_ratio is not None else None
                    if read.get('available'):
                        row.update(read_query_count=read['query_count'],
                                   read_attention_entropy_normalized=float(read['attention_entropy']),
                                   read_correction_rms_ratio=float(read['correction_rms_ratio']),
                                   read_spatial_gate_mean=float(read['gate_mean']) if label == 'spatial' and 'gate_mean' in read else None,
                                   read_channel_gate_mean=float(read['gate_mean']) if label == 'channel' and 'gate_mean' in read else None)
                        for name in ('q_norm_mean', 'k_norm_mean', 'logit_min', 'logit_max', 'logit_span_mean'):
                            row[f'read_{name}'] = float(read[name]) if name in read else None
                        attention = read['attention_mean'].cpu().tolist()
                        top1 = read['top1_fraction'].cpu().tolist()
                        for k, cluster in enumerate(clusters):
                            cluster.update(read_sample_global_step=sampled['global_batch_step'],
                                           read_attention_mean=attention[k], read_top1_fraction=top1[k])
                bank_rows.append(row)
                for cluster in clusters:
                    cluster.update(context, scale=level, bank=label)
                    cluster_rows.append(cluster)
                occupied = row.get('occupied_clusters', '-')
                entropy = row.get('assignment_entropy_normalized')
                entropy_text = '-' if entropy is None else f'{entropy:.2f}'
                summary.append(f'{level}/{label}: {row["status"]} pool={row["pool_size"]}/{row["capacity"]} '
                               f'updates={row["updates_total"]} occupied={occupied}/{bank.clusters} H={entropy_text}')
        self._append('prototype_banks.csv', BANK_FIELDS, bank_rows)
        self._append('prototype_clusters.csv', CLUSTER_FIELDS, cluster_rows)
        selected = gain['fg_selected'] + gain['bg_selected']
        print(f'[prototype epoch {epoch} batch {batch}] epoch_cumulative '
              f'useful_images={gain["useful_full_images"]:.0f}/{gain["full_images"]:.0f} '
              f'selected_pixels={selected:.0f} FG={gain["fg_selected"]:.0f} BG={gain["bg_selected"]:.0f} '
              f'successful_steps={context["successful_steps"]}', flush=True)
        for line in summary:
            print(f'  {line}', flush=True)
