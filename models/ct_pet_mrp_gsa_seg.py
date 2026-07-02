"""CT ConvNeXt encoder with stage-wise PET Metabolic Relation Prior Guided Self-Attention."""

from typing import Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from models.baseline_petct_unet import PETCTBaselineUNet, UNetStyleDecoder, _check_tensor, _check_tensor_list, _sanitize
from models.build_mdt_seg import create_feature_backbone, load_local_weights_safe
from models.pet_mrp_gsa import PETMRPGSABlock, PET_MRP_GSA_LOG_KEYS


VALID_PET_MRP_STAGES = ('all', 'c34', 'c4')
STAGE_NAMES = ('c1', 'c2', 'c3', 'c4')
PET_GUIDE_SPECS = (
    dict(dim=96, num_heads=4, split_or_not=True, mlp_ratio=4.0),
    dict(dim=192, num_heads=4, split_or_not=True, mlp_ratio=4.0),
    dict(dim=384, num_heads=8, split_or_not=True, mlp_ratio=3.0),
    dict(dim=768, num_heads=16, split_or_not=False, mlp_ratio=3.0),
)


def parse_pet_mrp_stages(stages: Union[str, Iterable[int], None]) -> Tuple[int, ...]:
    if stages is None or str(stages).strip().lower() in ('all',):
        return (0, 1, 2, 3)
    key = str(stages).strip().lower()
    if key in ('none', 'off', 'disabled'):
        return ()
    if key == 'c34':
        return (2, 3)
    if key == 'c4':
        return (3,)
    parsed = tuple(int(x.strip()) for x in key.split(',') if x.strip())
    invalid = [idx for idx in parsed if idx not in (0, 1, 2, 3)]
    if invalid:
        raise ValueError(
            f'Invalid pet_mrp_stages={stages}. Use all, c34, c4, or comma-separated stage indices 0-3.'
        )
    return parsed


class ConvNextStageWiseEncoder(nn.Module):
    """Expose HuggingFace ConvNeXt stem and stages for interleaved PET guidance."""

    def __init__(self, variant='convnext_tiny', in_channels=3):
        super().__init__()
        backbone = create_feature_backbone(variant, in_channels=in_channels)
        if not hasattr(backbone, 'model') or not hasattr(backbone.model, 'embeddings'):
            raise ValueError(
                f'PET-MRP-GSA requires HuggingFace ConvNeXt backbone, got {type(backbone).__name__}.'
            )
        self.model = backbone.model
        self.feature_info = backbone.feature_info

    def forward_stages(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.model.embeddings(x)
        stage_feats = []
        for stage in self.model.encoder.stages:
            x = stage(x)
            stage_feats.append(x)
        return stage_feats


class CTPETMRPGSASegmentation(nn.Module):
    """
    CT-only ConvNeXt-tiny encoder with PET metabolic relation prior guided self-attention.

    PET is used only as a prior map; no PET encoder is introduced.
    """

    def __init__(
        self,
        ct_backbone='convnext_tiny',
        ct_pretrained_path=None,
        in_channels=3,
        out_channels=1,
        decoder_channels=(512, 256, 128, 64),
        use_deep_supervision=False,
        use_pet_mrp_gsa=True,
        pet_mrp_stages='all',
        **kwargs,
    ):
        super().__init__()
        self.use_pet_mrp_gsa = bool(use_pet_mrp_gsa)
        self.pet_mrp_stage_indices = parse_pet_mrp_stages(pet_mrp_stages)
        self.use_deep_supervision = bool(use_deep_supervision)
        self._log_shapes = bool(kwargs.get('log_pet_mrp_shapes', False))
        self._shape_logged = False

        self.enc_ct = ConvNextStageWiseEncoder(variant=ct_backbone, in_channels=in_channels)
        load_local_weights_safe(self.enc_ct.model, ct_pretrained_path, name='CT_Encoder')
        encoder_channels = self.enc_ct.feature_info.channels()

        self.pet_guides = nn.ModuleList([
            PETMRPGSABlock(**spec) if self.use_pet_mrp_gsa and idx in self.pet_mrp_stage_indices else nn.Identity()
            for idx, spec in enumerate(PET_GUIDE_SPECS)
        ])

        self.decoder = UNetStyleDecoder(
            encoder_channels,
            decoder_channels=decoder_channels,
            out_channels=out_channels,
            use_deep_supervision=self.use_deep_supervision,
        )
        self._print_init_summary(encoder_channels)

    def _print_init_summary(self, encoder_channels: Sequence[int]):
        active = [STAGE_NAMES[i] for i in self.pet_mrp_stage_indices] if self.use_pet_mrp_gsa else []
        guide_params = sum(
            p.numel()
            for idx, block in enumerate(self.pet_guides)
            if idx in self.pet_mrp_stage_indices and isinstance(block, PETMRPGSABlock)
            for p in block.parameters()
        )
        total_params = sum(p.numel() for p in self.parameters())
        print('[PET-MRP-GSA] enabled=' + str(self.use_pet_mrp_gsa))
        print(f'[PET-MRP-GSA] active_stages={",".join(active) if active else "none"}')
        print(f'[PET-MRP-GSA] encoder_channels={list(encoder_channels)}')
        print(f'[PET-MRP-GSA] guide_params={guide_params / 1e6:.3f}M total_params={total_params / 1e6:.3f}M')
        for idx, block in enumerate(self.pet_guides):
            block_type = 'PETMRPGSABlock' if isinstance(block, PETMRPGSABlock) else 'Identity'
            split_or_not = getattr(block, 'split_or_not', None)
            print(
                f'[PET-MRP-GSA] stage={STAGE_NAMES[idx]} block={block_type} '
                f'split_or_not={split_or_not} enabled={idx in self.pet_mrp_stage_indices and self.use_pet_mrp_gsa}'
            )

    @staticmethod
    def _to_3ch(x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            return x.repeat(1, 3, 1, 1)
        return x

    @staticmethod
    def _resolve_pet_available(
        pet_available: Optional[Union[bool, torch.Tensor]],
        batch_size: int,
        device,
    ) -> Optional[Union[bool, torch.Tensor]]:
        if pet_available is None:
            return True
        if isinstance(pet_available, bool):
            return pet_available
        return pet_available.to(device=device).long().view(-1)

    def _apply_pet_guide(
        self,
        stage_idx: int,
        feat: torch.Tensor,
        pet: Optional[torch.Tensor],
        pet_available: Optional[Union[bool, torch.Tensor]],
    ) -> torch.Tensor:
        block = self.pet_guides[stage_idx]
        if isinstance(block, nn.Identity) or pet is None:
            return feat
        if isinstance(pet_available, bool) and not pet_available:
            return feat
        if torch.is_tensor(pet_available) and float(pet_available.float().sum().detach().cpu()) <= 0.0:
            return feat
        out = block(feat, pet, pet_available=pet_available)
        if self._log_shapes or not self._shape_logged:
            print(
                f'[PET-MRP-GSA] stage={STAGE_NAMES[stage_idx]} '
                f'in={tuple(feat.shape)} out={tuple(out.shape)} prior_skipped=False'
            )
            self._shape_logged = True
        return out

    def forward(
        self,
        ct: torch.Tensor,
        pet: Optional[torch.Tensor] = None,
        pet_available: Optional[Union[bool, torch.Tensor]] = True,
        target_size=None,
        return_aux=False,
    ):
        if target_size is None:
            target_size = ct.shape[-2:]

        ct_in = self._to_3ch(ct)
        pet_available = self._resolve_pet_available(pet_available, ct.shape[0], ct.device)
        use_pet = (
            self.use_pet_mrp_gsa
            and pet is not None
            and not (isinstance(pet_available, bool) and not pet_available)
            and not (
                torch.is_tensor(pet_available)
                and float(pet_available.float().sum().detach().cpu()) <= 0.0
            )
        )
        if not self._shape_logged:
            print(
                f'[PET-MRP-GSA] forward pet_prior_skipped={not use_pet} '
                f'pet_available={pet_available}'
            )

        stage_feats = self.enc_ct.forward_stages(ct_in)
        _check_tensor_list('ct_stage_feats', stage_feats)
        if not self._shape_logged:
            print(
                f'[PET-MRP-GSA] CT stem/stage shapes: '
                + ', '.join(f'{STAGE_NAMES[i]}={tuple(f.shape)}' for i, f in enumerate(stage_feats))
            )

        guided_feats = []
        for idx, feat in enumerate(stage_feats):
            guided = self._apply_pet_guide(idx, feat, pet if use_pet else None, pet_available)
            guided_feats.append(_sanitize(guided))
            _check_tensor(f'guided_{STAGE_NAMES[idx]}', guided_feats[-1])

        dec_out = self.decoder(guided_feats, target_size)
        outputs = PETCTBaselineUNet._finalize_decoder_output(dec_out)
        _check_tensor('logits', outputs['logits'])

        active_stage_count = sum(
            1 for idx in self.pet_mrp_stage_indices
            if self.use_pet_mrp_gsa and idx < len(stage_feats)
        )
        outputs['pred'] = outputs['logits']
        outputs['aux'] = {
            'pet_mrp_gsa_enabled': float(self.use_pet_mrp_gsa and use_pet),
            'pet_mrp_prior_skipped': float(not use_pet),
            'pet_mrp_active_stages': float(active_stage_count),
        }
        return outputs
