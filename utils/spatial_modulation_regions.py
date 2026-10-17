"""GT is used only for post-hoc region statistics, never retrieval or bank writes."""

import csv
import torch
import torch.nn.functional as F


@torch.no_grad()
def region_correction_report(model, loader, device, path):
    totals = {}
    context = {}
    handles = []

    def hook(scale):
        def collect(module, inputs, output):
            x = inputs[0].float()
            delta = output.float() - x
            q = F.interpolate(context["mask"].float(), size=x.shape[-2:], mode="area")
            for label, weight in [("foreground", q), ("background", 1 - q)]:
                key = (scale, label)
                acc = totals.setdefault(key, [0.0, 0.0, 0.0])
                acc[0] += float(weight.sum())
                acc[1] += float((x.square().mean(1, keepdim=True) * weight).sum())
                acc[2] += float((delta.square().mean(1, keepdim=True) * weight).sum())

        return collect

    was_training, mode = model.training, model.matrix_read_mode
    model.eval()
    model.matrix_read_mode = "normal"
    try:
        for i, scale in enumerate(model.matrix.scales, 1):
            handles.append(scale.register_forward_hook(hook(i)))
        for index, batch in enumerate(loader, 1):
            ct = batch["ct"].to(device)
            mask = batch["mask"].to(device)
            if mask.ndim == 3:
                mask = mask[:, None]
            context["mask"] = mask
            model(ct, forward_mode="missing")
            if index % 100 == 0:
                print(f"[region diagnosis] batch {index}/{len(loader)}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
        model.matrix_read_mode = mode
        model.train(was_training)
    rows = []
    for (scale, label), (mass, original, correction) in sorted(totals.items()):
        rows.append(
            dict(
                scale=f"s{scale}",
                region=label,
                occupancy_mass=mass,
                ct_rms=(original / mass) ** 0.5 if mass else None,
                correction_rms=(correction / mass) ** 0.5 if mass else None,
                correction_rms_ratio=(
                    (correction / original) ** 0.5 if original else None
                ),
            )
        )
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows
