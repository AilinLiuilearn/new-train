import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sparsemax(logits, dim=-1):
    orig_dtype = logits.dtype
    z = logits.float()
    z = z - z.max(dim=dim, keepdim=True).values
    z_sorted, _ = torch.sort(z, dim=dim, descending=True)
    z_cumsum = z_sorted.cumsum(dim)
    r = torch.arange(1, z.size(dim) + 1, device=z.device, dtype=z.dtype)
    shape = [1] * z.dim()
    shape[dim] = -1
    r = r.view(shape)
    support = 1 + r * z_sorted > z_cumsum
    k = support.sum(dim=dim, keepdim=True).clamp_min(1)
    tau = (z_cumsum.gather(dim, k - 1) - 1) / k.to(z.dtype)
    out = torch.clamp(z - tau, min=0.0)
    return out.to(orig_dtype)


class GainVerifiedVirtualTaskCompensation(nn.Module):
    def __init__(self, task_channels=512):
        super().__init__()
        self.task_channels = int(task_channels)
        self.route_dim = 128
        self.operator_rank = 16
        self.num_nodes = 9
        self.num_active_operators = 8
        self.route_hw = (16, 16)
        self.region_hw = (8, 8)

        self.semantic_ln = nn.LayerNorm(self.task_channels)
        self.semantic_projection = nn.Linear(self.task_channels, self.route_dim)
        self.decision_projection = nn.Linear(1, self.route_dim)
        self.local_query_projection = nn.Linear(self.route_dim, self.route_dim, bias=False)
        self.region_query_projection = nn.Linear(self.route_dim, self.route_dim, bias=False)
        self.local_keys = nn.Parameter(torch.randn(self.num_nodes, self.route_dim) * 0.02)
        self.region_keys = nn.Parameter(torch.randn(self.num_nodes, self.route_dim) * 0.02)
        self.local_keys.data[0].zero_()
        self.region_keys.data[0].zero_()
        self.operator_down = nn.Parameter(torch.empty(self.num_active_operators, self.task_channels, self.operator_rank))
        self.operator_up = nn.Parameter(torch.empty(self.num_active_operators, self.operator_rank, self.task_channels))
        nn.init.xavier_uniform_(self.operator_down)
        nn.init.xavier_uniform_(self.operator_up)
        self.value_norm = nn.LayerNorm(self.task_channels)
        self.output_projection = nn.Linear(self.task_channels, self.task_channels)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, d4_ct, ct_logits):
        x = d4_ct.detach().flatten(2).transpose(1, 2)
        q_sem = self.semantic_projection(self.semantic_ln(x))
        ct_prob = torch.sigmoid(ct_logits.detach().float())
        prob_patch = F.adaptive_avg_pool2d(ct_prob, output_size=self.route_hw)
        prob_token = prob_patch.flatten(2).transpose(1, 2)
        q_dec = self.decision_projection(prob_token)
        q_local = F.layer_norm(q_sem + q_dec, (self.route_dim,))
        q_map = q_local.transpose(1, 2).reshape(d4_ct.size(0), self.route_dim, *self.route_hw)
        region_map = F.avg_pool2d(q_map, kernel_size=2, stride=2)
        region_map = F.interpolate(region_map, size=self.route_hw, mode='nearest')
        q_region = region_map.flatten(2).transpose(1, 2)
        q_local_p = self.local_query_projection(q_local)
        q_region_p = self.region_query_projection(q_region)
        local_score = torch.einsum('bnc,nc->bn', q_local_p, self.local_keys) / math.sqrt(self.route_dim)
        region_score = torch.einsum('bnc,nc->bn', q_region_p, self.region_keys) / math.sqrt(self.route_dim)
        routing_logits = local_score + region_score
        routing_weights = sparsemax(routing_logits, dim=-1)
        x_operator = self.value_norm(d4_ct.detach().flatten(2).transpose(1, 2))
        hidden = torch.einsum('bnc,mcr->bnmr', x_operator, self.operator_down)
        hidden = F.gelu(hidden)
        operator_outputs = torch.einsum('bnmr,mrc->bnmc', hidden, self.operator_up)
        active_weights = routing_weights[:, :, 1:]
        raw_delta = (active_weights.unsqueeze(-1) * operator_outputs).sum(dim=2)
        delta_token = self.output_projection(raw_delta)
        delta_d4 = delta_token.transpose(1, 2).reshape_as(d4_ct)
        diagnostics = {
            'gvtc_null_weight_mean': routing_weights[:, :, 0].mean(),
            'gvtc_null_selection_ratio': (routing_weights.argmax(dim=-1) == 0).float().mean(),
            'gvtc_active_operator_count': (routing_weights[:, :, 1:] > 1e-6).float().sum(dim=-1).mean(),
            'gvtc_delta_rms': delta_d4.float().pow(2).mean().sqrt(),
            'gvtc_routing_max_weight': routing_weights.max(dim=-1).values.mean(),
        }
        for i in range(8):
            diagnostics[f'gvtc_operator_{i+1}_usage'] = routing_weights[:, :, i + 1].mean()
        return {'delta_d4': delta_d4, 'routing_weights': routing_weights, 'prob_patch': prob_patch, 'diagnostics': diagnostics}
