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
    view_shape = [1] * z.dim()
    view_shape[dim] = -1
    r = r.view(view_shape)
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

        self.semantic_ln = nn.LayerNorm(self.task_channels)
        self.semantic_projection = nn.Linear(self.task_channels, self.route_dim)
        self.decision_projection = nn.Linear(1, self.route_dim)
        self.local_query_projection = nn.Linear(self.route_dim, self.route_dim, bias=False)
        self.region_query_projection = nn.Linear(self.route_dim, self.route_dim, bias=False)

        self.local_keys = nn.Parameter(torch.randn(self.num_nodes, self.route_dim) * 0.02)
        self.region_keys = nn.Parameter(torch.randn(self.num_nodes, self.route_dim) * 0.02)

        self.operator_down = nn.Parameter(torch.empty(self.num_active_operators, self.task_channels, self.operator_rank))
        self.operator_up = nn.Parameter(torch.empty(self.num_active_operators, self.operator_rank, self.task_channels))
        for m in range(self.num_active_operators):
            nn.init.xavier_uniform_(self.operator_down[m])
            nn.init.xavier_uniform_(self.operator_up[m])

        self.value_norm = nn.LayerNorm(self.task_channels)
        self.output_projection = nn.Linear(self.task_channels, self.task_channels)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def _build_task_state(self, d4_ct, ct_logits):
        B, C, H, W = d4_ct.shape
        x = d4_ct.detach().flatten(2).transpose(1, 2)
        q_sem = self.semantic_projection(self.semantic_ln(x))
        ct_prob = torch.sigmoid(ct_logits.detach().float())
        prob_patch = F.adaptive_avg_pool2d(ct_prob, output_size=(H, W))
        q_dec = self.decision_projection(prob_patch.flatten(2).transpose(1, 2))
        q_local = F.layer_norm(q_sem + q_dec, (self.route_dim,))
        q_map = q_local.transpose(1, 2).reshape(B, self.route_dim, H, W)
        region_map = F.avg_pool2d(q_map, kernel_size=2, stride=2)
        region_map = F.interpolate(region_map, size=(H, W), mode='nearest')
        q_region = region_map.flatten(2).transpose(1, 2)
        return x, q_local, q_region, prob_patch

    def _compute_routing(self, q_local, q_region):
        local_q = F.normalize(self.local_query_projection(q_local).float(), dim=-1, eps=1e-6)
        region_q = F.normalize(self.region_query_projection(q_region).float(), dim=-1, eps=1e-6)
        local_k = F.normalize(self.local_keys.float(), dim=-1, eps=1e-6)
        region_k = F.normalize(self.region_keys.float(), dim=-1, eps=1e-6)
        local_score = torch.einsum('bnd,md->bnm', local_q, local_k)
        region_score = torch.einsum('bnd,md->bnm', region_q, region_k)
        routing_logits = local_score + region_score
        routing_weights = sparsemax(routing_logits, dim=-1)
        return routing_logits, routing_weights

    def _apply_operators(self, x, routing_weights):
        x_operator = self.value_norm(x)
        hidden = torch.einsum('bpc,mcr->bpmr', x_operator, self.operator_down)
        hidden = F.gelu(hidden)
        operator_outputs = torch.einsum('bpmr,mrd->bpmd', hidden, self.operator_up)
        raw_delta = (routing_weights[:, :, 1:].unsqueeze(-1) * operator_outputs).sum(dim=2)
        return raw_delta

    def forward(self, d4_ct, ct_logits):
        B, C, H, W = d4_ct.shape
        x, q_local, q_region, prob_patch = self._build_task_state(d4_ct, ct_logits)
        routing_logits, routing_weights = self._compute_routing(q_local, q_region)
        raw_delta = self._apply_operators(x, routing_weights)
        delta_token = self.output_projection(raw_delta)
        delta_d4 = delta_token.transpose(1, 2).reshape(B, C, H, W)
        diagnostics = {
            'gvtc_null_weight_mean': routing_weights[:, :, 0].mean(),
            'gvtc_null_selection_ratio': (routing_weights.argmax(dim=-1) == 0).float().mean(),
            'gvtc_active_operator_count': (routing_weights[:, :, 1:] > 1e-6).float().sum(dim=-1).mean(),
            'gvtc_operator_1_usage': routing_weights[:, :, 1].mean(),
            'gvtc_operator_2_usage': routing_weights[:, :, 2].mean(),
            'gvtc_operator_3_usage': routing_weights[:, :, 3].mean(),
            'gvtc_operator_4_usage': routing_weights[:, :, 4].mean(),
            'gvtc_operator_5_usage': routing_weights[:, :, 5].mean(),
            'gvtc_operator_6_usage': routing_weights[:, :, 6].mean(),
            'gvtc_operator_7_usage': routing_weights[:, :, 7].mean(),
            'gvtc_operator_8_usage': routing_weights[:, :, 8].mean(),
            'gvtc_delta_rms': delta_d4.float().pow(2).mean().sqrt(),
            'gvtc_routing_max_weight': routing_weights.max(dim=-1).values.mean(),
            'gvtc_routing_zero_ratio': (routing_weights == 0).float().mean(),
            'gvtc_non_null_weight_mean': routing_weights[:, :, 1:].sum(dim=-1).mean(),
            'gvtc_route_h': torch.tensor(float(H), device=d4_ct.device),
            'gvtc_route_w': torch.tensor(float(W), device=d4_ct.device),
        }
        return {'delta_d4': delta_d4, 'routing_weights': routing_weights, 'prob_patch': prob_patch, 'diagnostics': diagnostics}
