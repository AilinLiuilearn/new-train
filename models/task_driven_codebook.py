import torch
import torch.nn as nn
import torch.nn.functional as F

from models.baseline_petct_unet import _check_tensor


_STAGE_MODE_MAP = {
    's4': (4,),
    's34': (3, 4),
    'deep': (3, 4),
    's234': (2, 3, 4),
    'all': (1, 2, 3, 4),
}


class StageCTCodebookQuery(nn.Module):
    def __init__(self, in_channels, latent_dim):
        super().__init__()
        self.norm = nn.GroupNorm(_valid_group_count(in_channels), in_channels, affine=True)
        self.proj = nn.Conv2d(in_channels, latent_dim, kernel_size=1, bias=False)
        nn.init.kaiming_normal_(self.proj.weight, mode='fan_out', nonlinearity='linear')

    def forward(self, x):
        return self.proj(self.norm(x))


class TaskDrivenLearnableCodebook(nn.Module):
    def __init__(self, channels_list, num_tokens=8, temperature=0.07, stage_mode='all'):
        super().__init__()
        if stage_mode not in _STAGE_MODE_MAP:
            raise ValueError(f'Unsupported stage_mode={stage_mode}. Use one of {tuple(_STAGE_MODE_MAP)}')
        self.channels_list = tuple(int(c) for c in channels_list)
        self.num_tokens = int(num_tokens)
        self.temperature = float(temperature)
        self.stage_mode = stage_mode
        self.active_stage_numbers = _STAGE_MODE_MAP[stage_mode]
        deepest = self.active_stage_numbers[-1] - 1
        self.latent_dim = min(max(self.channels_list[deepest] // 4, 32), 128)

        self.shared_codebook_tokens = nn.Parameter(torch.empty(self.num_tokens, self.latent_dim))
        self.codebook_key = nn.Linear(self.latent_dim, self.latent_dim, bias=False)
        self.codebook_value = nn.Linear(self.latent_dim, self.latent_dim, bias=False)

        self.stage_queries = nn.ModuleDict()
        for stage in self.active_stage_numbers:
            self.stage_queries[str(stage)] = StageCTCodebookQuery(self.channels_list[stage - 1], self.latent_dim)

        nn.init.trunc_normal_(self.shared_codebook_tokens, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.codebook_key.weight)
        nn.init.xavier_uniform_(self.codebook_value.weight)

    def _diagnostics_for_stage(self, stage, q, assignment, retrieved):
        assignment_f = assignment.float()
        entropy = -(assignment_f * (assignment_f.clamp_min(1e-8).log())).sum(dim=1).mean()
        peak = assignment_f.max(dim=1).values.mean()
        retrieved_rms = retrieved.float().pow(2).mean().add(1e-12).sqrt()
        return {
            f'task_codebook_s{stage}_route_entropy': entropy.detach(),
            f'task_codebook_s{stage}_route_peak': peak.detach(),
            f'task_codebook_s{stage}_retrieved_rms': retrieved_rms.detach(),
        }

    def forward(self, aligned_ct_feats):
        k = self.codebook_key(self.shared_codebook_tokens)
        v = self.codebook_value(self.shared_codebook_tokens)
        _check_tensor('task_codebook_tokens', self.shared_codebook_tokens)
        _check_tensor('task_codebook_key', k)
        _check_tensor('task_codebook_value', v)

        diagnostics = {}
        retrieved_by_stage = {}
        key_norm = F.normalize(k.float(), dim=-1, eps=1e-6)
        token_rms = self.shared_codebook_tokens.float().pow(2).mean().add(1e-12).sqrt()
        key_cosine = (key_norm @ key_norm.t()).triu(1).mean()
        value_norm = F.normalize(v.float(), dim=-1, eps=1e-6)
        value_cosine = (value_norm @ value_norm.t()).triu(1).mean()
        diagnostics['task_codebook_token_rms'] = token_rms.detach()
        diagnostics['task_codebook_key_cosine'] = key_cosine.detach()
        diagnostics['task_codebook_value_cosine'] = value_cosine.detach()

        for stage in self.active_stage_numbers:
            q = self.stage_queries[str(stage)](aligned_ct_feats[stage - 1])
            q = F.normalize(q.float(), dim=1, eps=1e-6)
            logits = torch.einsum('bdhw,kd->bkhw', q, key_norm) / self.temperature
            _check_tensor(f'task_codebook_logits_s{stage}', logits)
            assignment = F.softmax(logits, dim=1)
            _check_tensor(f'task_codebook_assignment_s{stage}', assignment)
            retrieved = torch.einsum('bkhw,kd->bdhw', assignment.float(), v.float())
            retrieved = retrieved.to(dtype=aligned_ct_feats[stage - 1].dtype)
            _check_tensor(f'task_codebook_retrieved_s{stage}', retrieved)
            diagnostics.update(self._diagnostics_for_stage(stage, q, assignment, retrieved))
            retrieved_by_stage[stage] = retrieved

        return retrieved_by_stage, diagnostics


from models.pg_mtr import _valid_group_count
