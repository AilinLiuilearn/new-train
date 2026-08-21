import torch
import torch.nn as nn
import torch.nn.functional as F

from models.baseline_blocks import PrototypeReferencedPETAffineCalibration, StateAwareWeightedAddFusion, UNetStyleDecoder, _check_tensor, _check_tensor_list, _sanitize
from models.build_mdt_seg import create_feature_backbone, load_local_weights_safe
from models.ct_pet_prototype_imputation import CrossScaleCTPETPrototypeMemory


DP_SCALE_NAMES = ('s1', 's2', 's3', 's4')
DP_SCALE_CHANNELS = {'s1': 64, 's2': 128, 's3': 320, 's4': 512}
DP_SCALE_HEADS = {'s1': 4, 's2': 4, 's3': 8, 's4': 8}
DP_SCALE_SPATIAL = {'s1': (128, 128), 's2': (64, 64), 's3': (32, 32), 's4': (16, 16)}


def parse_dp_pgfa_scales(scales):
    if scales is None:
        return ()
    text = str(scales).strip().lower()
    if not text:
        return ()
    if text in ('all', 's1,s2,s3,s4'):
        return DP_SCALE_NAMES
    parts = tuple(p.strip() for p in text.split(',') if p.strip())
    invalid = [p for p in parts if p not in DP_SCALE_NAMES]
    if invalid:
        raise ValueError(f'Unsupported dp_pgfa_scales entries: {invalid}')
    # preserve user order but uniquify
    ordered = []
    for p in parts:
        if p not in ordered:
            ordered.append(p)
    return tuple(ordered)


class StageChannelAlign(nn.Module):
    def __init__(self, in_channels_list, out_channels_list):
        super().__init__()
        self.proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
            ) for c_in, c_out in zip(in_channels_list, out_channels_list)
        ])

    def forward(self, feats):
        return [proj(feat) for proj, feat in zip(self.proj, feats)]


class DualSharedAddPETCTBaseline(nn.Module):
    def __init__(
        self,
        ct_backbone='convnextv2_nano',
        pet_backbone='mit_b1',
        ct_pretrained_path=None,
        pet_pretrained_path=None,
        in_channels=3,
        out_channels=1,
        decoder_channels=(512, 256, 128, 64),
        use_deep_supervision=False,
        cppi_num_clusters=6,
        cppi_build_stage=3,
        cppi_output_dir=None,
        dp_pgfa_enabled=False,
        dp_pgfa_scales='s4',
        dp_text_tower_path=None,
        dp_biomedclip_model_path=None,
        dp_window_size=8,
        dp_depth=2,
        dp_prompt_len=128,
        dp_compress_ratio=8,
        dp_use_task_prompt=True,
        dp_use_text_prompt=True,
    ):
        super().__init__()
        self.use_deep_supervision = bool(use_deep_supervision)
        self.enc_ct = create_feature_backbone(ct_backbone, in_channels=in_channels)
        self.enc_pet = create_feature_backbone(pet_backbone, in_channels=in_channels)
        load_local_weights_safe(self.enc_ct, ct_pretrained_path, name='CT_Encoder')
        load_local_weights_safe(self.enc_pet, pet_pretrained_path, name='PET_Encoder')
        ct_channels = list(self.enc_ct.feature_info.channels())
        pet_channels = list(self.enc_pet.feature_info.channels())
        self.ct_align = StageChannelAlign(ct_channels, pet_channels)
        self.pet_calibration = PrototypeReferencedPETAffineCalibration(channels=pet_channels)
        self.fusion = StateAwareWeightedAddFusion(num_scales=len(pet_channels))
        self.decoder = UNetStyleDecoder(pet_channels, decoder_channels=decoder_channels, out_channels=out_channels, use_deep_supervision=self.use_deep_supervision)
        self.prototype_memory = CrossScaleCTPETPrototypeMemory(
            channels=pet_channels,
            num_clusters=cppi_num_clusters,
            build_stage=cppi_build_stage,
            output_dir=cppi_output_dir,
        )

        self.dp_pgfa_enabled = bool(dp_pgfa_enabled)
        self.dp_pgfa_scales = parse_dp_pgfa_scales(dp_pgfa_scales) if self.dp_pgfa_enabled else ()
        self.dp_text_tower_path = dp_text_tower_path
        self.dp_biomedclip_model_path = dp_biomedclip_model_path
        self.dp_window_size = int(dp_window_size)
        self.dp_depth = int(dp_depth)
        self.dp_prompt_len = int(dp_prompt_len)
        self.dp_compress_ratio = int(dp_compress_ratio)
        self.dp_use_task_prompt = bool(dp_use_task_prompt)
        self.dp_use_text_prompt = bool(dp_use_text_prompt)
        self.stage2_dp_only = False
        self._last_missing_used_real_pet = False
        self._last_missing_cppi_collect = False
        self._last_missing_cppi_retrieve = False
        self._last_dp_route = None
        self.dp_text_embedding_dim = 0
        self.dp_full_missing_text_cosine = None

        self.dp_pgfa_s1 = None
        self.dp_pgfa_s2 = None
        self.dp_pgfa_s3 = None
        self.dp_pgfa_s4 = None

        if self.dp_pgfa_enabled:
            self._build_dp_pgfa_adapters(pet_channels)

    def _build_dp_pgfa_adapters(self, pet_channels):
        from models.dual_prompt_pgfa import DualPromptPGFAAdapter, encode_fixed_biomedical_text_prompts

        if len(pet_channels) != 4:
            raise ValueError(f'DP-PGFA expects 4 PET scales, got {len(pet_channels)}')
        for scale_name, expected_c, actual_c in zip(DP_SCALE_NAMES, [DP_SCALE_CHANNELS[s] for s in DP_SCALE_NAMES], pet_channels):
            if int(actual_c) != int(expected_c):
                raise ValueError(
                    f'DP-PGFA channel mismatch at {scale_name}: expected {expected_c}, got {actual_c}'
                )
        if not self.dp_pgfa_scales:
            raise ValueError('dp_pgfa_enabled=True but dp_pgfa_scales is empty')

        full_emb = None
        missing_emb = None
        text_dim = 768
        if self.dp_use_text_prompt:
            if not self.dp_text_tower_path:
                raise ValueError('dp_use_text_prompt=True requires dp_text_tower_path')
            full_emb, missing_emb = encode_fixed_biomedical_text_prompts(self.dp_text_tower_path)
            text_dim = int(full_emb.numel())
            self.dp_text_embedding_dim = text_dim
            with torch.no_grad():
                self.dp_full_missing_text_cosine = float(
                    F.cosine_similarity(
                        full_emb.float().view(1, -1),
                        missing_emb.float().view(1, -1),
                        dim=-1,
                        eps=1e-6,
                    ).item()
                )

        for scale_name in self.dp_pgfa_scales:
            idx = DP_SCALE_NAMES.index(scale_name)
            channels = int(pet_channels[idx])
            adapter = DualPromptPGFAAdapter(
                in_channels=channels,
                num_heads=DP_SCALE_HEADS[scale_name],
                window_size=self.dp_window_size,
                depth=self.dp_depth,
                mlp_ratio=2.66,
                compress_ratio=self.dp_compress_ratio,
                prompt_len=self.dp_prompt_len,
                task_atom_num=32,
                task_prompt_dim=256,
                task_prompt_hidden=64,
                text_dim=text_dim,
                use_task_prompt=self.dp_use_task_prompt,
                use_text_prompt=self.dp_use_text_prompt,
                full_text_embedding=full_emb,
                missing_text_embedding=missing_emb,
                text_tower_path=None,
                qkv_bias=True,
                bias=False,
                drop=0.0,
                attn_drop=0.0,
                drop_path=0.0,
                zero_init_output=True,
            )
            setattr(self, f'dp_pgfa_{scale_name}', adapter)

    def _iter_dp_adapters(self, active_only=False):
        for scale_name in DP_SCALE_NAMES:
            adapter = getattr(self, f'dp_pgfa_{scale_name}', None)
            if adapter is None:
                continue
            if active_only and scale_name not in self.dp_pgfa_scales:
                continue
            yield scale_name, adapter

    def enable_stage2_dp_only(self):
        if not self.dp_pgfa_enabled:
            raise RuntimeError('enable_stage2_dp_only() called but dp_pgfa_enabled=False')
        active = list(self._iter_dp_adapters(active_only=True))
        if not active:
            raise RuntimeError('No active DP-PGFA adapters to train')

        for p in self.parameters():
            p.requires_grad = False
        for _, adapter in active:
            for p in adapter.parameters():
                p.requires_grad = True
        self.stage2_dp_only = True
        return self

    def train(self, mode=True):
        super().train(mode)
        if self.stage2_dp_only:
            self.enc_ct.eval()
            self.enc_pet.eval()
            self.ct_align.eval()
            self.pet_calibration.eval()
            self.fusion.eval()
            self.prototype_memory.eval()
            self.decoder.eval()
            active_names = set(self.dp_pgfa_scales)
            for scale_name in DP_SCALE_NAMES:
                adapter = getattr(self, f'dp_pgfa_{scale_name}', None)
                if adapter is None:
                    continue
                if scale_name in active_names:
                    adapter.train(mode)
                else:
                    adapter.eval()
        return self

    def cppi_status_snapshot(self):
        ready = self.prototype_memory.prototype_ready
        return {
            'bank_ready': bool(self.prototype_memory.bank_ready),
            'bank_version': int(self.prototype_memory.bank_version.item()),
            'ready_slots': int(ready.flatten().sum().item()),
            'total_slots': int(ready.numel()),
        }

    @staticmethod
    def _to_3ch(x):
        return x.repeat(1, 3, 1, 1) if x.shape[1] == 1 else x

    def _encode_ct(self, ct):
        ct_feats = self.enc_ct(self._to_3ch(ct))
        _check_tensor_list('ct_feats', ct_feats)
        return self.ct_align(ct_feats)

    def _encode_pet(self, pet):
        if pet is None:
            raise ValueError('API-style baseline requires PET input before fusion-time masking')
        pet_feats = self.enc_pet(self._to_3ch(pet))
        _check_tensor_list('pet_feats', pet_feats)
        return pet_feats

    def _decode(self, fused_feats, target_size, aux=None):
        out = self.decoder(fused_feats, target_size)
        _check_tensor('logits', out['logits'])
        out['pred'] = out['logits']
        out['aux'] = {} if aux is None else aux
        return out

    def _collect_cppi(self, ct_feats, pet_feats_real, mask):
        if self.stage2_dp_only:
            return None
        if self.training and mask is not None:
            return self.prototype_memory.collect(
                ct_feats=ct_feats,
                pet_feats=pet_feats_real,
                mask=mask,
                print_info=False,
                compute_report=False,
            )
        return None

    def _retrieve_cppi(self, ct_feats, compute_report=False, save_diagnostics=False, print_info=False, return_ct_reference=False):
        return self.prototype_memory.retrieve(
            ct_feats,
            compute_report=compute_report,
            save_diagnostics=save_diagnostics,
            print_info=print_info,
            return_ct_reference=return_ct_reference,
        )

    def _refine_dp_pgfa(self, fused_feats, route, pet_available=None):
        if not self.dp_pgfa_enabled:
            return fused_feats, {}

        if len(fused_feats) != 4:
            raise ValueError(f'DP-PGFA expects 4 fused scales, got {len(fused_feats)}')

        if self.stage2_dp_only:
            fused_feats = [x.detach() for x in fused_feats]

        refined = list(fused_feats)
        dp_stats = {}
        self._last_dp_route = str(route).lower().strip()

        for scale_name, adapter in self._iter_dp_adapters(active_only=True):
            idx = DP_SCALE_NAMES.index(scale_name)
            feat = refined[idx]
            expected_c = DP_SCALE_CHANNELS[scale_name]
            if feat.shape[1] != expected_c:
                raise AssertionError(
                    f'{scale_name} channel mismatch: expected {expected_c}, got {feat.shape[1]}'
                )
            result = adapter(feat, route=route, pet_available=pet_available)
            out = result.feature
            if out.shape != feat.shape:
                raise AssertionError(
                    f'{scale_name} DP-PGFA shape mismatch: in={tuple(feat.shape)} out={tuple(out.shape)}'
                )
            refined[idx] = out
            for key, value in result.stats.items():
                if key == 'prompt_weights_mean':
                    dp_stats[f'dp_{scale_name}_prompt_weights_mean'] = value
                elif key == 'raw_residual_l2_ratio':
                    dp_stats[f'dp_{scale_name}_raw_residual_l2_ratio'] = value
                elif key == 'delta_l2_ratio':
                    dp_stats[f'dp_{scale_name}_delta_l2_ratio'] = value
                elif key == 'prompt_weights_entropy':
                    dp_stats[f'dp_{scale_name}_prompt_weights_entropy'] = value
                else:
                    dp_stats[f'dp_{scale_name}_{key}'] = value
        return refined, dp_stats

    def _forward_full(self, ct, pet, target_size, mask=None):
        ct_feats = self._encode_ct(ct)
        pet_feats_real = self._encode_pet(pet)
        self._collect_cppi(ct_feats, pet_feats_real, mask)
        if self.prototype_memory.bank_ready:
            _, ct_reference_feats, _ = self._retrieve_cppi(
                ct_feats,
                compute_report=False,
                save_diagnostics=False,
                print_info=False,
                return_ct_reference=True,
            )
            ct_reference_feats = [x.detach() for x in ct_reference_feats]
            pet_feats_cal = self.pet_calibration(
                ct_feats,
                pet_feats_real,
                ct_reference_feats,
                reference_valid=True,
            )
        else:
            pet_feats_cal = self.pet_calibration(
                ct_feats,
                pet_feats_real,
                None,
                reference_valid=False,
            )
        fused_feats = self.fusion(ct_feats, pet_feats_cal, mode='full')
        fused_feats, dp_stats = self._refine_dp_pgfa(fused_feats, route='full')
        return self._decode(fused_feats, target_size, aux={'dp_pgfa_stats': dp_stats})

    def _forward_missing(self, ct, pet, target_size, mask=None):
        ct_feats = self._encode_ct(ct)
        if self.stage2_dp_only:
            self._last_missing_used_real_pet = False
            self._last_missing_cppi_collect = False
            pet_feats_proxy, ct_reference_feats, _ = self._retrieve_cppi(
                ct_feats,
                compute_report=False,
                save_diagnostics=False,
                print_info=False,
                return_ct_reference=True,
            )
            self._last_missing_cppi_retrieve = True
            ct_reference_feats = [x.detach() for x in ct_reference_feats]
            pet_feats_cal = self.pet_calibration(
                ct_feats,
                pet_feats_proxy,
                ct_reference_feats,
                reference_valid=self.prototype_memory.bank_ready,
            )
            fused_feats = self.fusion(ct_feats, pet_feats_cal, mode='missing')
            fused_feats, dp_stats = self._refine_dp_pgfa(fused_feats, route='missing')
            return self._decode(fused_feats, target_size, aux={'dp_pgfa_stats': dp_stats})

        self._last_missing_used_real_pet = True
        pet_feats_real = self._encode_pet(pet)
        collect_out = self._collect_cppi(ct_feats, pet_feats_real, mask)
        self._last_missing_cppi_collect = collect_out is not None
        pet_feats_proxy, ct_reference_feats, _ = self._retrieve_cppi(
            ct_feats,
            compute_report=False,
            save_diagnostics=False,
            print_info=False,
            return_ct_reference=True,
        )
        self._last_missing_cppi_retrieve = True
        ct_reference_feats = [x.detach() for x in ct_reference_feats]
        pet_feats_cal = self.pet_calibration(
            ct_feats,
            pet_feats_proxy,
            ct_reference_feats,
            reference_valid=self.prototype_memory.bank_ready,
        )
        fused_feats = self.fusion(ct_feats, pet_feats_cal, mode='missing')
        fused_feats, dp_stats = self._refine_dp_pgfa(fused_feats, route='missing')
        return self._decode(fused_feats, target_size, aux={'dp_pgfa_stats': dp_stats})

    def _forward_auto(self, ct, pet, pet_available, target_size, mask=None):
        ct_feats = self._encode_ct(ct)
        availability = pet_available.to(device=ct.device).long().view(-1)
        if availability.numel() != ct.shape[0]:
            raise ValueError('pet_available must contain one state per sample')
        if not torch.all((availability == 0) | (availability == 1)):
            raise ValueError('pet_available values must be 0 or 1')

        if self.stage2_dp_only:
            # Stage2 auto: encode real PET only for available samples' fusion path,
            # but never collect/update CPPI. Proxy is always retrieved for missing slots.
            pet_feats_proxy, ct_reference_feats, _ = self._retrieve_cppi(
                ct_feats,
                compute_report=False,
                save_diagnostics=False,
                print_info=False,
                return_ct_reference=True,
            )
            ct_reference_feats = [x.detach() for x in ct_reference_feats]
            if bool((availability == 1).any()):
                pet_feats_real = self._encode_pet(pet)
            else:
                pet_feats_real = pet_feats_proxy
            pet_selected = []
            avail_view = availability.view(-1, 1, 1, 1)
            for feat_real, feat_proxy in zip(pet_feats_real, pet_feats_proxy):
                availability_mask = avail_view.to(device=feat_real.device, dtype=feat_real.dtype)
                pet_selected.append(feat_real * availability_mask + feat_proxy * (1.0 - availability_mask))
            pet_selected = self.pet_calibration(
                ct_feats,
                pet_selected,
                ct_reference_feats,
                reference_valid=self.prototype_memory.bank_ready,
            )
            fused_feats = self.fusion(
                ct_feats,
                pet_selected,
                mode='auto',
                pet_available=pet_available,
            )
            fused_feats, dp_stats = self._refine_dp_pgfa(
                fused_feats,
                route='auto',
                pet_available=pet_available,
            )
            return self._decode(fused_feats, target_size, aux={'dp_pgfa_stats': dp_stats})

        pet_feats_real = self._encode_pet(pet)
        self._collect_cppi(ct_feats, pet_feats_real, mask)
        pet_feats_proxy, ct_reference_feats, _ = self._retrieve_cppi(
            ct_feats,
            compute_report=False,
            save_diagnostics=False,
            print_info=False,
            return_ct_reference=True,
        )
        ct_reference_feats = [x.detach() for x in ct_reference_feats]
        pet_selected = []
        avail_view = availability.view(-1, 1, 1, 1)
        for feat_real, feat_proxy in zip(pet_feats_real, pet_feats_proxy):
            availability_mask = avail_view.to(device=feat_real.device, dtype=feat_real.dtype)
            pet_selected.append(feat_real * availability_mask + feat_proxy * (1.0 - availability_mask))
        pet_selected = self.pet_calibration(
            ct_feats,
            pet_selected,
            ct_reference_feats,
            reference_valid=self.prototype_memory.bank_ready,
        )
        fused_feats = self.fusion(
            ct_feats,
            pet_selected,
            mode='auto',
            pet_available=pet_available,
        )
        fused_feats, dp_stats = self._refine_dp_pgfa(
            fused_feats,
            route='auto',
            pet_available=pet_available,
        )
        return self._decode(fused_feats, target_size, aux={'dp_pgfa_stats': dp_stats})

    @torch.no_grad()
    def finalize_cppi_epoch(self, epoch, save_json=True, save_visualizations=False, print_info=True):
        if self.stage2_dp_only:
            status = self.cppi_status_snapshot()
            if print_info:
                print(
                    f"[CPPI STAGE2 READONLY] epoch={epoch} "
                    f"bank_ready={status['bank_ready']} "
                    f"bank_version={status['bank_version']} "
                    f"ready_slots={status['ready_slots']}/{status['total_slots']}",
                    flush=True,
                )
            return {
                'bank_version_before': status['bank_version'],
                'bank_version_after': status['bank_version'],
                'ready_count': status['ready_slots'],
                'ready_slots': status['ready_slots'],
                'classes': {
                    'background': {'num_candidates': 0},
                    'foreground': {'num_candidates': 0},
                },
                'stage2_retrieve_only': True,
            }
        return self.prototype_memory.finalize_epoch(
            epoch=epoch,
            save_json=save_json,
            save_visualizations=save_visualizations,
            print_info=print_info,
        )

    def forward(self, ct, pet, pet_available=None, target_size=None, forward_mode='auto', mask=None):
        if target_size is None:
            target_size = ct.shape[-2:]
        if forward_mode == 'full':
            return self._forward_full(ct, pet, target_size, mask=mask)
        if forward_mode == 'missing':
            return self._forward_missing(ct, pet, target_size, mask=mask)
        if forward_mode == 'auto':
            if pet_available is None:
                pet_available = torch.ones(ct.shape[0], device=ct.device, dtype=torch.long)
            return self._forward_auto(ct, pet, pet_available, target_size, mask=mask)
        raise ValueError(f'Unsupported forward_mode={forward_mode!r}')
