import torch
import torch.nn as nn
import torch.nn.functional as F



class PatchTaskGainCompensator(nn.Module):
    def __init__(
        self,
        task_channels=512,
        hidden_channels=256,
        gain_embed_channels=32,
        norm_groups=8,
    ):
        super().__init__()
        self.task_channels = int(task_channels)
        self.hidden_channels = int(hidden_channels)
        self.gain_embed_channels = int(gain_embed_channels)
        self.norm_groups = int(norm_groups)

        self.gain_proj = nn.Sequential(
            nn.Conv2d(self.task_channels + 1, self.hidden_channels, kernel_size=1, bias=False),
            nn.GroupNorm(self.norm_groups, self.hidden_channels),
            nn.GELU(),
        )
        self.gain_local = nn.Conv2d(
            self.hidden_channels,
            self.hidden_channels,
            kernel_size=3,
            padding=1,
            groups=self.hidden_channels,
            bias=False,
        )
        self.gain_regional = nn.Conv2d(
            self.hidden_channels,
            self.hidden_channels,
            kernel_size=3,
            padding=2,
            dilation=2,
            groups=self.hidden_channels,
            bias=False,
        )
        self.gain_fuse = nn.Sequential(
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=1, bias=False),
            nn.GroupNorm(self.norm_groups, self.hidden_channels),
            nn.GELU(),
        )
        self.gain_head = nn.Sequential(
            nn.Conv2d(self.hidden_channels, 1, kernel_size=1),
            nn.Tanh(),
        )

        self.embed = nn.Sequential(
            nn.Conv2d(1, self.gain_embed_channels, kernel_size=1, bias=False),
            nn.GroupNorm(self._valid_gn_groups(self.gain_embed_channels), self.gain_embed_channels),
            nn.GELU(),
        )
        self.correction_proj = nn.Sequential(
            nn.Conv2d(self.task_channels + self.gain_embed_channels, self.hidden_channels, kernel_size=1, bias=False),
            nn.GroupNorm(self.norm_groups, self.hidden_channels),
            nn.GELU(),
        )
        self.corr_local = nn.Conv2d(
            self.hidden_channels,
            self.hidden_channels,
            kernel_size=3,
            padding=1,
            groups=self.hidden_channels,
            bias=False,
        )
        self.corr_regional = nn.Conv2d(
            self.hidden_channels,
            self.hidden_channels,
            kernel_size=3,
            padding=2,
            dilation=2,
            groups=self.hidden_channels,
            bias=False,
        )
        self.corr_fuse = nn.Sequential(
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=1, bias=False),
            nn.GroupNorm(self.norm_groups, self.hidden_channels),
            nn.GELU(),
        )
        self.delta_head = nn.Conv2d(self.hidden_channels, self.task_channels, kernel_size=1)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

    @staticmethod
    def _valid_gn_groups(channels):
        for g in (8, 4, 2, 1):
            if channels % g == 0:
                return g
        return 1

    def forward(self, d4_ct, ct_logits):
        prob = torch.sigmoid(ct_logits.detach().float())
        entropy = -prob * torch.log(prob + 1e-6) - (1.0 - prob) * torch.log(1.0 - prob + 1e-6)
        entropy_patch = F.adaptive_avg_pool2d(entropy, output_size=d4_ct.shape[-2:])

        gain_input = torch.cat([d4_ct.detach(), entropy_patch.detach()], dim=1)
        gain_feature = self.gain_proj(gain_input)
        gain_local = self.gain_local(gain_feature)
        gain_regional = self.gain_regional(gain_feature)
        gain_feature = self.gain_fuse(gain_local + gain_regional)
        gain_pred_signed = self.gain_head(gain_feature)
        benefit_pred = F.relu(gain_pred_signed)

        gain_embed = self.embed(benefit_pred)
        correction_input = torch.cat([d4_ct.detach(), gain_embed], dim=1)
        correction_feature = self.correction_proj(correction_input)
        corr_local = self.corr_local(correction_feature)
        corr_regional = self.corr_regional(correction_feature)
        correction_feature = self.corr_fuse(corr_local + corr_regional)
        delta_d4 = self.delta_head(correction_feature)

        return {
            'gain_pred_signed': gain_pred_signed,
            'benefit_pred': benefit_pred,
            'delta_d4': delta_d4,
            'entropy_patch': entropy_patch,
        }
