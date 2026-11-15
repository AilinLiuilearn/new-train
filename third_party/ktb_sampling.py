"""Adapted sampling pattern from imcjx/KTB, Apache-2.0 (see KTB_LICENSE.txt).

Source: semseg/models/backbones/swin.py, commit
75370dd584fb9d8571bced1aabade198dfbdc80f, DSCF grid_sample/union block.
Changes: plain B,C,H,W instead of groups; (x,y) input; cell-centered
align_corners=False; border padding; returns B,N,C tokens. Not full DSCF.
"""
import torch
from torch.nn import functional as F


def sample_union(feature,ct_points,pet_points):
    x=F.grid_sample(feature,ct_points,mode='bilinear',padding_mode='border',align_corners=False)
    y=F.grid_sample(feature,pet_points,mode='bilinear',padding_mode='border',align_corners=False)
    return torch.cat([x.flatten(2),y.flatten(2)],dim=2).transpose(1,2)
