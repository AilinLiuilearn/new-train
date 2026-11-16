"""Descriptive 2D small-component coverage; not clinical lesion detection."""
import numpy as np
from scipy import ndimage
import torch


class SmallTargetMetrics:
    def __init__(self, max_area=256):
        self.max_area = max_area
        self.components = self.any_overlap = self.half_covered = 0
        self.true_pixels = self.covered_pixels = 0
        self.coverage_sum = 0.

    @torch.no_grad()
    def update(self, logits, target):
        if logits.shape != target.shape or logits.ndim != 4 or logits.shape[1] != 1:
            raise ValueError('Same-grid B,1,H,W required for component diagnostics')
        pred = (logits.float().sigmoid() > .5).cpu().numpy()[:, 0]
        gt = (target >= .5).cpu().numpy()[:, 0]
        for p, y in zip(pred, gt):
            labels, count = ndimage.label(y, structure=np.ones((3, 3), dtype=np.uint8))
            areas = np.bincount(labels.ravel(), minlength=count+1)
            hits = np.bincount(labels[p].ravel(), minlength=count+1)
            selected = (areas > 0) & (areas <= self.max_area)
            selected[0] = False
            a, h = areas[selected], hits[selected]
            coverage = h / a
            self.components += len(a)
            self.any_overlap += int((h > 0).sum())
            self.half_covered += int((coverage >= .5).sum())
            self.true_pixels += int(a.sum())
            self.covered_pixels += int(h.sum())
            self.coverage_sum += float(coverage.sum())

    def compute(self):
        def ratio(a, b):
            return a / b if b else None
        return {'small_2d_component_max_area_px': self.max_area,
                'small_2d_component_count': self.components,
                'small_2d_any_overlap_recall': ratio(self.any_overlap, self.components),
                'small_2d_half_coverage_recall': ratio(self.half_covered, self.components),
                'small_2d_mean_coverage': ratio(self.coverage_sum, self.components),
                'small_2d_pixel_sensitivity': ratio(self.covered_pixels, self.true_pixels),
                'scope': 'GT 8-connected components <=256 pixels on each 512 image; '
                         'not patient-level or physical-size lesion detection; not used for model selection'}
