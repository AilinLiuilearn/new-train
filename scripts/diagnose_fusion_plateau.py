#!/usr/bin/env python
"""Paired Full-path diagnosis for Add versus CMX/SKFF. Never proves a model ceiling."""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from scipy import ndimage as ndi

EPS = 1e-12


def counts(pred, gt):
    p, g = np.asarray(pred, bool), np.asarray(gt, bool)
    if p.shape != g.shape:
        raise ValueError("Prediction/GT shapes differ")
    return np.array([np.count_nonzero(p & g), np.count_nonzero(p & ~g),
                     np.count_nonzero(~p & g), np.count_nonzero(~p & ~g)], dtype=np.int64)


def scores(c):
    tp, fp, fn, tn = map(float, c)
    den = 2 * tp + fp + fn
    return dict(dice=2 * tp / den if den else 1.0,
                iou=tp / (tp + fp + fn) if tp + fp + fn else 1.0,
                precision=tp / (tp + fp) if tp + fp else 0.0,
                recall=tp / (tp + fn) if tp + fn else 0.0,
                tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn))


def json_write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def csv_write(path, rows):
    if not rows:
        Path(path).write_text("")
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_ids(path):
    ids = [s.strip() for s in Path(path).read_text().splitlines() if s.strip()]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Empty split or duplicate IDs: " + str(path))
    return ids


def select_records(records, max_cases, slices_per_case, seed):
    groups = {}
    for r in records:
        groups.setdefault(r["case_id"], []).append(r)
    rng = np.random.default_rng(seed)
    cases = sorted(groups)
    if max_cases and len(cases) > max_cases:
        cases = sorted(rng.choice(cases, max_cases, replace=False).tolist())
    chosen = []
    for case in cases:
        rows = sorted(groups[case], key=lambda x: x["image_id"])
        if slices_per_case and len(rows) > slices_per_case:
            indices = sorted(rng.choice(len(rows), slices_per_case, replace=False).tolist())
            rows = [rows[i] for i in indices]
        chosen.extend(rows)
    return chosen


def donor_indices(records, seed):
    """Random case cycle; donors always come from a different patient, not a bijection."""
    groups = {}
    for i, r in enumerate(records):
        groups.setdefault(r["case_id"], []).append(i)
    if len(groups) < 2:
        return None
    rng = np.random.default_rng(seed)
    cases = list(rng.permutation(sorted(groups)))
    successor = {c: cases[(i + 1) % len(cases)] for i, c in enumerate(cases)}
    return [int(rng.choice(groups[successor[r["case_id"]]])) for r in records]


class PairedDiagnosticDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, donors=None):
        self.dataset, self.donors = dataset, donors

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        item = dict(self.dataset[i])
        if self.donors is not None:
            item["donor_pet"] = self.dataset[self.donors[i]]["pet"]
        return item


def load_model(path, device, weights):
    from tasks.mdt_seg import MDTSegTeacher
    payload = MDTSegTeacher.load_state_dicts(path)
    saved = dict(payload["config"])
    use_ema = (payload.get("eval_weights", "raw") == "ema") if weights == "recorded" else weights == "ema"
    tag = "ema" if use_ema else "raw"
    if saved.get("experiment_type") == "full_cmx_fusion":
        # Existing evaluator checks architecture/source metadata and loads strictly.
        del payload
        from eval_full_cmx_fusion import load_full_checkpoint
        model, cfg, payload, actual = load_full_checkpoint(path, device, use_ema=use_ema)
        assert actual == tag
        saved = dict(payload["config"])
    elif (saved.get("experiment_type") == "full_add_baseline"
          and saved.get("fusion_type") == "AddFusion"):
        if saved.get("use_deep_supervision") or saved.get("deep_supervision"):
            raise ValueError("This diagnostic expects the non-auxiliary Full Add baseline")
        from models.dual_shared_add_baseline import DualSharedAddPETCTBaseline
        model = DualSharedAddPETCTBaseline(
            ct_backbone=saved["ct_backbone"], pet_backbone=saved["pet_backbone"],
            pretrained=False, decoder_channels=tuple(saved["decoder_channels"]))
        selected = payload.get("model_ema") if use_ema else payload.get("model")
        if selected is None:
            raise ValueError("Requested weights missing: " + tag)
        model.load_state_dict(selected, strict=True)
        model.to(device)
    else:
        raise ValueError("Supported checkpoints: Full Add baseline and this branch's Full CMX/SKFF; "
                         "unsupported experiment: " + str(saved.get("experiment_type")))
    model.eval()
    meta = dict(path=str(Path(path).resolve()), sha256=sha256_file(path), weights=tag,
                epoch=payload.get("epoch"), best_epoch=payload.get("best_epoch"),
                config=saved)
    del payload
    return model, saved, meta


def validate_protocol(a, b):
    required = ("ct_backbone", "pet_backbone", "decoder_channels", "image_size_2d",
                "norm_mode", "hd95_mode", "train_split_file", "val_split_file", "test_split_file")
    problems = [k for k in required if k not in a or k not in b
                or json.dumps(a[k]) != json.dumps(b[k])]
    if problems:
        raise ValueError("Checkpoint protocols differ or metadata is missing: " + ", ".join(problems))


def make_loaders(cfg, args):
    from datasets.pclt20k_seg import (PCLT20KSegDataset, _records_from_ids,
                                      _check_patient_disjoint)
    root = args.root or cfg["root"]
    names = {k: cfg[k + "_split_file"] for k in ("train", "val", "test")}
    all_records = {k: _records_from_ids(root, read_ids(Path(root) / name))
                   for k, name in names.items()}
    _check_patient_disjoint(all_records["train"], all_records["val"], all_records["test"],
        allow_val_equals_test=bool(cfg.get("allow_val_equals_test", False) and names["val"] == names["test"]))
    selected = select_records(all_records[args.split], args.max_cases, args.slices_per_case, args.seed)
    train = select_records(all_records["train"], args.train_cases, args.train_slices_per_case, args.seed + 1)
    donors = donor_indices(selected, args.seed + 2)
    def dataset(rows):
        return PCLT20KSegDataset(rows, image_size=cfg["image_size_2d"], train=False,
                                aug_mode="none", norm_mode=cfg["norm_mode"], ct_only=False)
    def loader(ds):
        return torch.utils.data.DataLoader(ds, batch_size=args.batch_size,
            shuffle=False, drop_last=False, num_workers=args.num_workers,
            pin_memory=args.device == "cuda")
    manifest = dict(root=str(Path(root).resolve()), splits=names,
                    selected_split=args.split,
                    full_selected_split=len(selected) == len(all_records[args.split]),
                    evaluation_ids=[r["image_id"] for r in selected],
                    training_ids=[r["image_id"] for r in train],
                    cross_patient_donors=([selected[j]["image_id"] for j in donors] if donors else None))
    return loader(PairedDiagnosticDataset(dataset(selected), donors)), loader(dataset(train)), manifest


def lesion_info(gt, pred, image_id, mode, small_area, coverage):
    labels, n = ndi.label(gt, structure=np.ones((3, 3), dtype=bool))
    areas = np.bincount(labels.ravel(), minlength=n + 1)[1:]
    overlaps = np.bincount(labels[pred], minlength=n + 1)[1:]
    rows = []
    for i, (area, overlap) in enumerate(zip(areas, overlaps), 1):
        rows.append(dict(image_id=image_id, mode=mode, gt_component=i,
                         area_pixels=int(area), small=bool(area <= small_area),
                         predicted_coverage=float(overlap / area),
                         detected=bool(overlap / area >= coverage)))
    return rows


class Accumulator:
    def __init__(self, hd95_mode):
        self.hd95_mode = hd95_mode
        self.c = np.zeros(4, dtype=np.int64)
        self.cases = {}
        self.hd, self.undefined_hd = [], 0
        self.loss_sum, self.n, self.slice_dice_sum = 0.0, 0, 0.0
        self.lesions, self.detected, self.small, self.small_detected = 0, 0, 0, 0

    def add(self, pred, gt, case, loss, lesions, compute_hd=True):
        from utils.metrics_seg import compute_hd95_pair
        c = counts(pred, gt)
        self.c += c
        self.cases[case] = self.cases.get(case, np.zeros(4, dtype=np.int64)) + c
        self.n += 1
        self.loss_sum += float(loss)
        self.slice_dice_sum += scores(c)["dice"]
        for row in lesions:
            self.lesions += 1
            self.detected += row["detected"]
            self.small += row["small"]
            self.small_detected += row["small"] and row["detected"]
        hd = None
        if compute_hd:
            # No invented fallback for the official metric's undefined empty-GT case.
            if self.hd95_mode == "cipa_official" and not gt.any() and pred.any():
                self.undefined_hd += 1
            else:
                hd = float(compute_hd95_pair(pred, gt, self.hd95_mode))
                if not math.isfinite(hd):
                    raise RuntimeError("Nonfinite HD95")
                self.hd.append(hd)
        return c, hd

    def summary(self):
        return dict(scores(self.c), samples=self.n,
            macro_slice_dice=self.slice_dice_sum / self.n if self.n else None,
            macro_case_dice=float(np.mean([scores(c)["dice"] for c in self.cases.values()])) if self.cases else None,
            loss=self.loss_sum / self.n if self.n else None,
            hd95=float(np.mean(self.hd)) if self.hd else None,
            hd95_defined_samples=len(self.hd), hd95_undefined_samples=self.undefined_hd,
            gt_components=self.lesions,
            gt_component_recall=self.detected / self.lesions if self.lesions else None,
            small_gt_components=self.small,
            small_component_recall=self.small_detected / self.small if self.small else None)


def paired_bootstrap(base, fusion, repeats, seed):
    cases = sorted(base)
    if set(cases) != set(fusion):
        raise ValueError("Bootstrap requires identical paired patients")
    if len(cases) < 2:
        return dict(patients=len(cases), ci95=None, reason="Fewer than two patients")
    a, b = np.stack([base[k] for k in cases]), np.stack([fusion[k] for k in cases])
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(repeats):
        idx = rng.integers(0, len(cases), len(cases))
        deltas.append(scores(b[idx].sum(0))["dice"] - scores(a[idx].sum(0))["dice"])
    return dict(patients=len(cases), repeats=repeats,
                delta_global_dice=scores(b.sum(0))["dice"] - scores(a.sum(0))["dice"],
                ci95=[float(x) for x in np.percentile(deltas, [2.5, 97.5])],
                scope="Patient-resampled interval for these fixed checkpoints and selected slices; not seed variance")


def log_diagnostics(path):
    path = Path(path)
    if not path.is_file():
        return dict(available=False, path=str(path))
    sample = path.read_text()
    try:
        dialect = csv.Sniffer().sniff(sample[:8192], delimiters=",\t")
    except csv.Error:
        dialect = csv.excel
    rows = list(csv.DictReader(sample.splitlines(), dialect=dialect))
    def number(row, key):
        try:
            v = float(row.get(key, ""))
            return v if math.isfinite(v) else None
        except (ValueError, TypeError):
            return None
    valid = [r for r in rows if number(r, "epoch") is not None]
    invalid = {}
    for key in ("grad_norm", "grad_enc_ct", "grad_enc_pet", "grad_fusion"):
        invalid[key] = sum(r.get(key, "") != "" and number(r, key) is None for r in valid)
    best = max((r for r in valid if number(r, "val_dice") is not None),
               key=lambda r: number(r, "val_dice"), default=None)
    tail = valid[-5:]
    skipped = sum(number(r, "skipped_updates") or 0 for r in valid)
    batches = sum(number(r, "train_batches") or 0 for r in valid)
    def values(row):
        return {k: number(row, k) for k in ("epoch", "train_loss", "val_loss", "val_dice")}
    out = dict(available=True, path=str(path), epochs=len(valid),
               best_dice_row=values(best) if best else None,
               last_row=values(valid[-1]) if valid else None,
               nonfinite_or_invalid_epoch_grad_entries=invalid,
               skipped_updates=skipped, logged_batches=batches,
               skipped_fraction=skipped / batches if batches else None)
    if len(tail) >= 2:
        out["last_window_change"] = {k: number(tail[-1], k) - number(tail[0], k)
            if number(tail[-1], k) is not None and number(tail[0], k) is not None else None
            for k in ("train_loss", "val_loss", "val_dice")}
    return out


def feature_rows(cs, ps, fused, mask, ids, gates):
    import torch.nn.functional as F
    rows = []
    for stage, (c, p, f) in enumerate(zip(cs, ps, fused)):
        c, p, f = c.float(), p.float(), f.float()
        add = c + p
        flat_a, flat_f = add.flatten(1), f.flatten(1)
        alpha = (flat_a * flat_f).sum(1) / flat_a.square().sum(1).clamp_min(EPS)
        residual = f - alpha[:, None, None, None] * add
        cosine = F.cosine_similarity(c.flatten(1), p.flatten(1), dim=1)
        area = F.interpolate(mask.float(), size=f.shape[-2:], mode="area")[:, 0]
        for i, image_id in enumerate(ids):
            raw = float((f[i] - add[i]).norm() / add[i].norm().clamp_min(EPS))
            shape_change = float(residual[i].norm() / f[i].norm().clamp_min(EPS))
            energy = residual[i].square().mean(0)
            fg = float((energy * area[i]).sum() / area[i].sum().clamp_min(EPS))
            bg = float((energy * (1 - area[i])).sum() / (1 - area[i]).sum().clamp_min(EPS))
            row = dict(image_id=image_id, stage=stage + 1,
                       ct_pet_cosine=float(cosine[i]), pet_ct_norm_ratio=float(p[i].norm() / c[i].norm().clamp_min(EPS)),
                       native_vs_add_relative_l2=raw, fitted_native_add_scale=float(alpha[i]),
                       native_vs_scaled_add_relative_l2=shape_change,
                       residual_fg_rms=math.sqrt(max(fg, 0)) if float(area[i].sum()) > 0 else None,
                       residual_bg_rms=math.sqrt(max(bg, 0)) if float((1-area[i]).sum()) > 0 else None)
            if stage in gates:
                w = gates[stage][i].numpy().reshape(2, -1)
                row.update(ct_weight_mean=float(w[0].mean()), pet_weight_mean=float(w[1].mean()),
                           pet_weight_std=float(w[1].std()),
                           normalized_gate_entropy=float(-(w * np.log(np.maximum(w, EPS))).sum(0).mean() / math.log(2)),
                           gate_saturated_fraction=float(((w[1] < .05) | (w[1] > .95)).mean()))
            rows.append(row)
    return rows


@torch.no_grad()
def predict_variants(model, batch, device, criterion, prefix, include_stages, verify_forward=False):
    import torch.nn.functional as F
    ct, pet, target = (batch[k].to(device, non_blocking=True) for k in ("ct", "pet", "mask"))
    cs, ps = model._encode_ct(ct), model._encode_pet(pet)
    ps = [F.interpolate(p, size=c.shape[-2:], mode="bilinear", align_corners=False)
          if p.shape[-2:] != c.shape[-2:] else p for c, p in zip(cs, ps)]
    gates, handles = {}, []
    if include_stages and hasattr(model.fusion, "skffs"):
        for i, block in enumerate(model.fusion.skffs):
            handles.append(block.softmax.register_forward_hook(
                lambda m, inp, out, i=i: gates.__setitem__(i, out.detach().float().cpu())))
    try:
        native = model.fusion(cs, ps)
    finally:
        for handle in handles:
            handle.remove()
    full_logits = model._decode(native, ct.shape[-2:])["logits"]
    if verify_forward:
        official = model(ct, pet=pet, forward_mode="full")["logits"]
        torch.testing.assert_close(full_logits, official, atol=1e-5, rtol=1e-5)
    predictions = {}
    def add(name, feats=None, logits=None):
        if logits is None:
            logits = model._decode(feats, ct.shape[-2:])["logits"]
        if not torch.isfinite(logits).all():
            raise RuntimeError("Nonfinite FP32 logits: " + name)
        loss, _ = criterion(logits.float(), target.float())
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite FP32 loss: " + name)
        predictions[prefix + "/" + name] = (logits.float().sigmoid()[:, 0].cpu().numpy(), float(loss))
    add("full", logits=full_logits)
    add("zero_pet_features", model.fusion(cs, [torch.zeros_like(p) for p in ps]))
    if "donor_pet" in batch:
        donor_ps = model._encode_pet(batch["donor_pet"].to(device, non_blocking=True))
        add("cross_patient_pet", model.fusion(cs, donor_ps))
    features = []
    if include_stages:
        features = feature_rows(cs, ps, native, target, batch["image_id"], gates)
        additions = [c + p for c, p in zip(cs, ps)]
        add("add_all", additions)
        for i in range(4):
            altered = list(native)
            altered[i] = additions[i]
            add("add_s" + str(i + 1), altered)
    return predictions, features


@torch.no_grad()
def evaluate_pair(baseline, fusion, loader, args):
    from utils.seg_losses import BCEDiceLoss
    criterion = BCEDiceLoss()
    accum, slice_rows, lesion_rows, feat_rows = {}, [], [], []
    sweep = {(mode, t): np.zeros(4, dtype=np.int64)
             for mode in ("baseline/full", "fusion/full") for t in args.thresholds}
    oracle_c = np.zeros(4, dtype=np.int64)
    oracle_slice_sum, n, gained, harmed, shared = 0.0, 0, 0, 0, 0
    for batch_idx, batch in enumerate(loader):
        pa, _ = predict_variants(baseline, batch, args.device, criterion, "baseline", False, batch_idx == 0)
        pb, feats = predict_variants(fusion, batch, args.device, criterion, "fusion", True, batch_idx == 0)
        feat_rows.extend(feats)
        predictions = dict(pa, **pb)
        masks = batch["mask"][:, 0].numpy() >= .5
        for i, (gt, image_id, case) in enumerate(zip(masks, batch["image_id"], batch["case_id"])):
            base = pa["baseline/full"][0][i] > .5
            fused = pb["fusion/full"][0][i] > .5
            base_ok, fused_ok = base == gt, fused == gt
            corrected = int(np.count_nonzero(~base_ok & fused_ok))
            damaged = int(np.count_nonzero(base_ok & ~fused_ok))
            oracle_c += counts(np.where(base_ok, base, fused), gt)
            oracle_slice_sum += max(scores(counts(base, gt))["dice"], scores(counts(fused, gt))["dice"])
            gained += corrected
            harmed += damaged
            shared += int(np.count_nonzero(~base_ok & ~fused_ok))
            n += 1
            band = ndi.binary_dilation(gt, iterations=args.boundary_radius) & ~ndi.binary_erosion(gt, iterations=args.boundary_radius)
            for mode, (probs, loss) in predictions.items():
                prob, pred = probs[i], probs[i] > .5
                components = lesion_info(gt, pred, image_id, mode, args.small_area, args.lesion_coverage)
                lesion_rows.extend(components)
                stats = accum.setdefault(mode, Accumulator(args.hd95_mode))
                c, hd = stats.add(pred, gt, case, loss, components)
                native_prob = pa["baseline/full"][0][i] if mode.startswith("baseline/") else pb["fusion/full"][0][i]
                native_pred = native_prob > .5
                diff = np.abs(prob - native_prob)
                row = dict(image_id=image_id, case_id=case, mode=mode, **scores(c), hd95=hd,
                    batch_loss=loss, gt_area=int(gt.sum()), boundary_fp=int(np.count_nonzero(pred & ~gt & band)),
                    boundary_fn=int(np.count_nonzero(~pred & gt & band)),
                    changed_pixels_vs_own_full=int(np.count_nonzero(pred != native_pred)),
                    mean_probability_change_vs_own_full=float(diff.mean()),
                    fg_probability_change=float(diff[gt].mean()) if gt.any() else None,
                    bg_probability_change=float(diff[~gt].mean()) if (~gt).any() else None)
                if mode == "fusion/full":
                    row.update(corrected_vs_baseline=corrected, damaged_vs_baseline=damaged)
                slice_rows.append(row)
                if mode in ("baseline/full", "fusion/full"):
                    for t in args.thresholds:
                        sweep[(mode, t)] += counts(prob > t, gt)
        if batch_idx % 10 == 0 or batch_idx + 1 == len(loader):
            print("[EVAL] batches=" + str(batch_idx + 1) + " slices=" + str(n), flush=True)
    summaries = {mode: a.summary() for mode, a in accum.items()}
    case_rows = []
    for case in sorted(accum["baseline/full"].cases):
        a = scores(accum["baseline/full"].cases[case])
        b = scores(accum["fusion/full"].cases[case])
        case_rows.append(dict(case_id=case, baseline_dice=a["dice"], fusion_dice=b["dice"],
                              delta_dice=b["dice"] - a["dice"], baseline_fp=a["fp"], fusion_fp=b["fp"],
                              baseline_fn=a["fn"], fusion_fn=b["fn"]))
    sweep_rows = [dict(mode=mode, threshold=t, **scores(c)) for (mode, t), c in sweep.items()]
    result = dict(modes=summaries,
        paired_patient_bootstrap=paired_bootstrap(accum["baseline/full"].cases, accum["fusion/full"].cases, args.bootstrap, args.seed),
        corrected_pixels=gained, damaged_pixels=harmed, shared_error_pixels=shared,
        pixel_oracle=dict(scores(oracle_c), definition="GT picks a correct prediction from the two fixed models at each pixel; unattainable diagnostic oracle, not a model ceiling"),
        slice_oracle_macro_dice=oracle_slice_sum / n if n else None)
    return result, dict(per_slice=slice_rows, per_case=case_rows, per_lesion=lesion_rows,
                        feature_scales=feat_rows, threshold_sweep=sweep_rows)


@torch.no_grad()
def evaluate_train_subset(models, loader, args):
    from utils.seg_losses import BCEDiceLoss
    criterion, result = BCEDiceLoss(), {}
    for name, model in models.items():
        stat = Accumulator(args.hd95_mode)
        for batch in loader:
            ct, pet, target = (batch[k].to(args.device) for k in ("ct", "pet", "mask"))
            logits = model(ct, pet=pet, forward_mode="full")["logits"]
            if not torch.isfinite(logits).all():
                raise RuntimeError("Nonfinite training-subset FP32 logits")
            loss, _ = criterion(logits.float(), target.float())
            prob = logits.sigmoid()[:, 0].cpu().numpy()
            for p, g, case in zip(prob, batch["mask"][:, 0].numpy() >= .5, batch["case_id"]):
                stat.add(p > .5, g, case, float(loss), [], compute_hd=False)
        result[name] = stat.summary()
    return result


def tiny_fit_probe(model, train_loader, args):
    """Extra in-memory optimization on fixed TRAIN samples; not a benchmark continuation."""
    from utils.seg_losses import BCEDiceLoss
    items, ids = [], []
    dataset = train_loader.dataset
    indices = np.linspace(0, len(dataset) - 1, min(len(dataset), args.overfit_samples)).astype(int).tolist()
    probe_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, indices),
                                               batch_size=args.batch_size, shuffle=False, num_workers=0)
    for batch in probe_loader:
        for i, image_id in enumerate(batch["image_id"]):
            if len(items) < args.overfit_samples:
                items.append({k: batch[k][i].clone() for k in ("ct", "pet", "mask")})
                ids.append(image_id)
        if len(items) >= args.overfit_samples:
            break
    if not items:
        raise ValueError("No training examples for tiny-fit probe")
    probe = copy.deepcopy(model).to(args.device).float().eval()
    # eval() intentionally freezes BN buffers/dropout, while weights still receive gradients.
    for parameter in probe.parameters():
        parameter.requires_grad_(True)
    batches = []
    for start in range(0, len(items), args.batch_size):
        subset = items[start:start + args.batch_size]
        batches.append({k: torch.stack([x[k] for x in subset]).to(args.device)
                        for k in ("ct", "pet", "mask")})
    criterion = BCEDiceLoss()
    optimizer = torch.optim.AdamW(probe.parameters(), lr=args.overfit_lr, weight_decay=0)
    @torch.no_grad()
    def assess(step):
        c, loss_sum, samples = np.zeros(4, dtype=np.int64), 0.0, 0
        image_dice = []
        for batch in batches:
            logits = probe(batch["ct"], pet=batch["pet"], forward_mode="full")["logits"]
            loss, _ = criterion(logits.float(), batch["mask"].float())
            p = logits.sigmoid()[:, 0].cpu().numpy() > .5
            g = batch["mask"][:, 0].cpu().numpy() >= .5
            c += counts(p, g)
            image_dice.extend(scores(counts(pi, gi))["dice"] for pi, gi in zip(p, g))
            loss_sum += float(loss) * len(p)
            samples += len(p)
        return dict(step=step, loss=loss_sum / samples, macro_slice_dice=float(np.mean(image_dice)),
                    min_slice_dice=min(image_dice), **scores(c))
    history, status = [assess(0)], "completed"
    for step in range(1, args.overfit_steps + 1):
        batch = batches[(step - 1) % len(batches)]
        optimizer.zero_grad(set_to_none=True)
        logits = probe(batch["ct"], pet=batch["pet"], forward_mode="full")["logits"]
        loss, _ = criterion(logits.float(), batch["mask"].float())
        if not torch.isfinite(loss):
            status = "nonfinite_loss"
            break
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in probe.parameters()):
            status = "nonfinite_gradient"
            break
        torch.nn.utils.clip_grad_norm_(probe.parameters(), 5.0, error_if_nonfinite=True)
        optimizer.step()
        if step % 25 == 0 or step == args.overfit_steps:
            history.append(assess(step))
            print("[TINY FIT] " + json.dumps(history[-1]), flush=True)
    return dict(status=status, training_ids=ids, history=history, lr=args.overfit_lr,
                batch_size=args.batch_size, requested_steps=args.overfit_steps,
                target_dice=args.overfit_target_dice,
                reached_target=history[-1]["min_slice_dice"] >= args.overfit_target_dice,
                positive_samples=sum(bool(item["mask"].any()) for item in items),
                scope="Fixed TRAIN examples, no augmentation, eval-mode BN/dropout, FP32, AdamW weight_decay=0; in-memory clone only, no checkpoint saved")


def make_report(summary, tables):
    modes = summary["evaluation"]["modes"]
    base, fused = modes["baseline/full"], modes["fusion/full"]
    ci = summary["evaluation"]["paired_patient_bootstrap"].get("ci95")
    lines = [
        "# 融合停滞诊断",
        "",
        "**本报告不能证明模型达到理论上限。它只描述两个 checkpoint 在指定样本上的表现与敏感性。**",
        "",
        "- 完整评估所选划分：" + str(summary["manifest"]["full_selected_split"]),
        "- 固定阈值：0.5；FP32、eval 模式；HD95 单位为输入网格像素。",
        "- 置信区间按患者重采样，仅覆盖这些固定模型/样本，不覆盖随机种子和训练过程的不确定性。",
        "- 重用验证/测试集：" + str(summary["manifest"]["splits"]["val"] == summary["manifest"]["splits"]["test"]),
        "",
        "| 路径 | 全局 Dice | 患者平均 Dice | HD95 | FP | FN |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    def fmt(v):
        return "NA" if v is None else format(v, ".6f")
    for mode, m in modes.items():
        lines.append("| " + mode + " | " + " | ".join([
            fmt(m["dice"]), fmt(m["macro_case_dice"]), fmt(m["hd95"]), str(m["fp"]), str(m["fn"])]) + " |")
    lines += ["", "- 训练设置差异：" + json.dumps(summary.get("training_protocol_differences", {}), ensure_ascii=False),
        "- 实际权重标签：" + json.dumps(summary.get("comparison_weight_labels", {}), ensure_ascii=False),
        "", "## 已观察到的差异", "",
        "- 融合相对 baseline 的全局 Dice 差（百分点）：" + fmt(100 * (fused["dice"] - base["dice"])),
        "- 患者配对 bootstrap 95% 区间（百分点）：" + str([100*x for x in ci] if ci else None),
        "- 融合纠正的像素数：" + str(summary["evaluation"]["corrected_pixels"]),
        "- 融合损害的像素数：" + str(summary["evaluation"]["damaged_pixels"]),
        "- 两模型共同错误像素数：" + str(summary["evaluation"]["shared_error_pixels"]),
        "- GT 像素选择 oracle Dice：" + fmt(summary["evaluation"]["pixel_oracle"]["dice"]),
        "",
        "Oracle 使用了 GT，只衡量这两个固定预测的互补空间，不能作为可部署分数或模型上限。",
        "",
        "## 如何判读", "",
        "1. 先比较关闭增强的训练子集分数与评估分数。训练拟合很高而评估停滞，优先考虑泛化；"
        "训练子集也低时，再看可选的小样本拟合探针。两者都不直接证明容量上限。",
        "2. 查看零 PET 特征、跨患者 PET 的分数和 per_slice.csv 中相对自身 full 的概率/掩码变化。"
        "变化小表示当前模型对这些干预不敏感；变化大仍需结合 Dice 判断是否利用了有用信息。"
        "这些干预会改变输入分布，不能代替独立训练的 CT-only 对照。",
        "3. add_all/add_s1..4 保留该融合模型自身的编码器、对齐器和解码器，只在推理时改为相加。"
        "因此属于压力测试，不能用它直接替代重训消融或认定某尺度融合无价值。",
        "4. 查看 per_case.csv 的正负差异与 per_lesion.csv。病灶在此定义为 2D GT 的 8 连通分量，"
        "检测定义为预测覆盖该 GT 分量至少指定比例；不是 3D 病灶匹配指标。",
        "5. feature_scales.csv 同时报告原始差异和去除最佳标量缩放后的差异。"
        "单纯幅值变化不等于新信息；通道权重、余弦相似度和梯度大小也不能被直接解释为可靠性。",
        "6. threshold_sweep.csv 是同一集合上的事后阈值扫描，仅检查校准敏感性。"
        "不能把其中最优值作为独立测试成绩；正式阈值需在独立验证集确定。",
        "7. HD95 的 undefined 计数必须一并检查。官方口径对 GT 空、预测非空的情况未定义，"
        "脚本保留像素 FP 并报告未定义数，不伪造距离；不同有效样本数的均值不能直接横比。",
        "",
        "## 关闭增强的训练子集", ""]
    for name, m in summary["training_subset"].items():
        lines.append("- " + name + "：Dice=" + fmt(m["dice"]) + "，loss=" + fmt(m["loss"]) + "，n=" + str(m["samples"]))
    lines += ["", "## 可选的小样本拟合探针", ""]
    probe = summary["tiny_fit"]
    if probe.get("enabled"):
        lines += ["- 状态：" + probe["status"], "- 正样本数：" + str(probe["positive_samples"]),
            "- 初始：" + json.dumps(probe["history"][0], ensure_ascii=False),
            "- 最后：" + json.dumps(probe["history"][-1], ensure_ascii=False),
            "- 每张训练样本均达到预设 Dice 门槛：" + str(probe["reached_target"]),
            "能够拟合少数样本只说明该配置可以记住这些训练样本；未达到门槛也可能来自优化、"
            "冻结 BN、学习率、步数或样本难度，不能据此宣布网络达到上限。"]
    else:
        lines.append("未执行。可通过 --overfit_steps 200 对固定训练样本进行独立的内存拟合探针。")
    lines += ["", "## 训练日志", "",
        "完整数值见 training_logs.json。非有限的 epoch 平均梯度不等于整轮都发生梯度爆炸，"
        "需要结合跳过步数和逐步记录。训练日志中的增强训练损失与本报告的无增强 FP32 评估损失不直接混用。",
        "", "## 当前结论的边界", "",
        "**是否达到模型上限：无法由本次检查确认。** 若需判定改进是否稳定，还需要相同协议的多种子对照、"
        "独立患者验证，以及受控改变解码器/训练方式的实验。请先依据上述证据定位主要失败形式，再决定结构修改。",
        ""]
    worst = sorted(tables["per_case"], key=lambda r: r["delta_dice"])[:10]
    lines += ["## 下降最多的患者", "", "| 患者 | Baseline Dice | Fusion Dice | 差值 |",
              "| --- | ---: | ---: | ---: |"]
    for r in worst:
        lines.append("| " + str(r["case_id"]) + " | " + fmt(r["baseline_dice"]) + " | " +
                     fmt(r["fusion_dice"]) + " | " + fmt(r["delta_dice"]) + " |")
    return "\n".join(lines) + "\n"


def self_test():
    """Synthetic invariants only: does not validate real checkpoint/GPU integration."""
    gt = np.array([[1, 1], [0, 0]], bool)
    a = np.array([[1, 0], [1, 0]], bool)
    b = np.array([[0, 1], [0, 0]], bool)
    assert counts(a, gt).tolist() == [1, 1, 1, 1]
    assert scores(counts(gt, gt))["dice"] == 1
    oracle = np.where(a == gt, a, b)
    assert np.array_equal(oracle, gt)
    base_cases = {"a": counts(a, gt), "b": counts(b, gt)}
    same = paired_bootstrap(base_cases, base_cases, 50, 1)
    assert same["ci95"] == [0.0, 0.0]
    assert paired_bootstrap({"a": counts(a, gt)}, {"a": counts(b, gt)}, 50, 1)["ci95"] is None
    rows = [dict(case_id=c, image_id=c + str(i)) for c in ("a", "b", "c") for i in range(3)]
    donors = donor_indices(rows, 1)
    assert all(rows[i]["case_id"] != rows[j]["case_id"] for i, j in enumerate(donors))
    assert donor_indices(rows[:3], 1) is None
    chosen = select_records(rows, 2, 2, 1)
    assert len(chosen) == 4 and chosen == select_records(rows, 2, 2, 1)
    components = lesion_info(gt, gt, "x", "full", 4, .1)
    assert len(components) == 1 and components[0]["detected"]
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "log.csv"
        path.write_text("epoch,train_loss,val_loss,val_dice,grad_norm,skipped_updates,train_batches\n"
                        "1,1,1,0.5,inf,1,100\n2,0.5,1.1,0.49,1,0,100\n")
        result = log_diagnostics(path)
        assert result["best_dice_row"]["epoch"] == 1
        assert result["nonfinite_or_invalid_epoch_grad_entries"]["grad_norm"] == 1
        assert result["skipped_fraction"] == .005
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head = torch.nn.Conv2d(6, 1, 1)
        def forward(self, ct, pet=None, forward_mode="full"):
            return {"logits": self.head(torch.cat([ct, pet], dim=1))}
    examples = [{"ct": torch.ones(3, 4, 4), "pet": torch.zeros(3, 4, 4),
                 "mask": torch.ones(1, 4, 4), "image_id": str(i)} for i in range(2)]
    args = argparse.Namespace(device="cpu", overfit_samples=2, batch_size=2,
                              overfit_steps=2, overfit_lr=.01, overfit_target_dice=.95)
    model = Toy().eval()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    fit = tiny_fit_probe(model, torch.utils.data.DataLoader(examples, batch_size=2), args)
    assert fit["status"] == "completed" and len(fit["history"]) == 2
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())
    class PairFusion(torch.nn.Module):
        def __init__(self, scale):
            super().__init__()
            self.scale = scale
        def forward(self, cs, ps):
            return [c + self.scale * p for c, p in zip(cs, ps)]
    class CachedToy(torch.nn.Module):
        def __init__(self, scale):
            super().__init__()
            self.fusion = PairFusion(scale)
        def _encode_ct(self, x):
            return [x.mean(1, keepdim=True)] * 4
        def _encode_pet(self, x):
            return [x.mean(1, keepdim=True)] * 4
        def _decode(self, feats, target_size):
            return {"logits": sum(feats) / 4}
        def forward(self, ct, pet=None, forward_mode="full"):
            return self._decode(self.fusion(self._encode_ct(ct), self._encode_pet(pet)), ct.shape[-2:])
    examples = [{"ct": torch.ones(3, 4, 4), "pet": torch.zeros(3, 4, 4),
                 "donor_pet": torch.ones(3, 4, 4), "mask": torch.ones(1, 4, 4),
                 "image_id": str(i), "case_id": str(i)} for i in range(2)]
    loader = torch.utils.data.DataLoader(examples, batch_size=2)
    opts = argparse.Namespace(device="cpu", hd95_mode="surface", thresholds=[.5], boundary_radius=1,
                              small_area=4, lesion_coverage=.1, bootstrap=10, seed=1)
    evaluation, tables = evaluate_pair(CachedToy(1).eval(), CachedToy(.3).eval(), loader, opts)
    assert evaluation["modes"]["fusion/full"]["dice"] == 1
    assert len(evaluation["modes"]) == 11 and len(tables["feature_scales"]) == 8
    training = evaluate_train_subset({"baseline": CachedToy(1).eval()}, loader, opts)
    summary = dict(evaluation=evaluation, training_subset=training, tiny_fit={"enabled": False},
                   manifest={"full_selected_split": False, "splits": {"val": "val.txt", "test": "test.txt"}})
    assert "融合停滞诊断" in make_report(summary, tables)
    json.dumps(evaluation, allow_nan=False)
    print("[PASS] synthetic metrics/oracle/bootstrap/donor/subset/lesion/log/tiny-fit/cached-evaluation/report invariants", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self_test", action="store_true")
    parser.add_argument("--baseline_checkpoint")
    parser.add_argument("--fusion_checkpoint")
    parser.add_argument("--output_dir")
    parser.add_argument("--root", default=None)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--weights", choices=("recorded", "raw", "ema"), default="recorded")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_cases", type=int, default=20)
    parser.add_argument("--slices_per_case", type=int, default=8)
    parser.add_argument("--train_cases", type=int, default=20)
    parser.add_argument("--train_slices_per_case", type=int, default=8)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[.2, .3, .4, .5, .6, .7, .8])
    parser.add_argument("--hd95_mode", choices=("cipa_official", "surface", "legacy"), default=None)
    parser.add_argument("--small_area", type=int, default=256)
    parser.add_argument("--lesion_coverage", type=float, default=.1)
    parser.add_argument("--boundary_radius", type=int, default=3)
    parser.add_argument("--overfit_steps", type=int, default=0)
    parser.add_argument("--overfit_samples", type=int, default=8)
    parser.add_argument("--overfit_lr", type=float, default=1e-4)
    parser.add_argument("--overfit_target_dice", type=float, default=.95)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.self_test:
        self_test()
        return
    if not all((args.baseline_checkpoint, args.fusion_checkpoint, args.output_dir)):
        parser.error("--baseline_checkpoint, --fusion_checkpoint and --output_dir are required")
    if (min(args.max_cases, args.slices_per_case, args.train_cases, args.train_slices_per_case,
            args.num_workers, args.overfit_steps) < 0
            or min(args.batch_size, args.bootstrap, args.small_area, args.boundary_radius, args.overfit_samples) < 1
            or not 0 < args.lesion_coverage <= 1 or not 0 < args.overfit_target_dice <= 1
            or args.overfit_lr <= 0 or not args.thresholds or any(not 0 < t < 1 for t in args.thresholds)):
        parser.error("Invalid diagnostic limits/thresholds")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu explicitly")
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if Path(args.output_dir).exists():
        raise FileExistsError("Use a new output directory: " + args.output_dir)
    baseline, base_cfg, base_meta = load_model(args.baseline_checkpoint, args.device, args.weights)
    fusion, fusion_cfg, fusion_meta = load_model(args.fusion_checkpoint, args.device, args.weights)
    if base_cfg.get("fusion_type") != "AddFusion":
        raise ValueError("--baseline_checkpoint must be the independently trained Full Add baseline")
    validate_protocol(base_cfg, fusion_cfg)
    args.hd95_mode = args.hd95_mode or base_cfg["hd95_mode"]
    if args.hd95_mode == "cipa_official":
        from medpy.metric.binary import hd95
        if base_cfg["image_size_2d"] != 512:
            raise ValueError("Official HD95 requires 512 input")
    eval_loader, train_loader, manifest = make_loaders(base_cfg, args)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    json_write(out / "manifest.json", dict(args=vars(args), data=manifest,
               baseline=base_meta, fusion=fusion_meta, torch_version=str(torch.__version__)))
    try:
        evaluation, tables = evaluate_pair(baseline, fusion, eval_loader, args)
        training = evaluate_train_subset({"baseline": baseline, "fusion": fusion}, train_loader, args)
        tiny = dict(enabled=False)
        if args.overfit_steps:
            baseline.cpu()
            if args.device == "cuda":
                torch.cuda.empty_cache()
            tiny = dict(enabled=True, **tiny_fit_probe(fusion, train_loader, args))
        logs = {name: log_diagnostics(Path(path).parent / "train_log.csv")
                for name, path in (("baseline", args.baseline_checkpoint), ("fusion", args.fusion_checkpoint))}
        protocol_keys = ("random_state", "epochs", "batch_size", "learning_rate", "weight_decay",
                         "cosine_warmup", "cosine_min_lr", "lr_flat_ratio", "aug_mode",
                         "ema_enabled", "ema_decay", "ema_start_epoch", "mixed_precision", "grad_clip")
        protocol_differences = {k: {"baseline": base_cfg.get(k), "fusion": fusion_cfg.get(k)}
                                for k in protocol_keys if base_cfg.get(k) != fusion_cfg.get(k)}
        summary = dict(theoretical_model_ceiling="not_identifiable_from_this_diagnostic",
                       training_protocol_differences=protocol_differences,
                       comparison_weight_labels={"baseline": base_meta["weights"], "fusion": fusion_meta["weights"]},
                       manifest=manifest, evaluation=evaluation, training_subset=training,
                       training_logs=logs, tiny_fit=tiny,
                       metric_definitions=dict(threshold=.5, hd95_mode=args.hd95_mode,
                           small_component_max_area=args.small_area, component_connectivity=8,
                           component_detection_min_gt_coverage=args.lesion_coverage),
                       limitations=["Ablations are inference-time distribution interventions, not retrained models",
                           "Patient bootstrap does not measure seed variability",
                           "Threshold sweep and GT oracle are diagnostic only",
                           "Train subset uses eval-mode, no augmentation, unlike training-log losses"])
        for name, rows in tables.items():
            csv_write(out / (name + ".csv"), rows)
        json_write(out / "summary.json", summary)
        json_write(out / "training_logs.json", logs)
        if args.overfit_steps:
            json_write(out / "tiny_fit.json", tiny)
        (out / "report.md").write_text(make_report(summary, tables))
        print("[DONE] " + str(out.resolve() / "report.md"), flush=True)
    except Exception as exc:
        json_write(out / "error.json", dict(type=type(exc).__name__, message=str(exc),
                                          completed=False))
        raise


if __name__ == "__main__":
    main()
