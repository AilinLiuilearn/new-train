import torch
import torch.nn as nn
import torch.nn.functional as F

from models.baseline_petct_unet import UNetStyleDecoder, _check_tensor, _check_tensor_list, _sanitize
from models.build_mdt_seg import ConvBNAct, create_feature_backbone


class FrequencyDecouple(nn.Module):
    """Frequency-domain low/high decomposition for 2D PET/CT tensors.

    ``fft`` uses a differentiable radial Gaussian low-pass mask in the Fourier
    domain and defines the high-frequency residual as ``x - x_low``. This keeps
    the spatial tensor shape and input intensity distribution stable while using
    an explicit frequency-domain separation instead of a local smoothing-only
    approximation. ``avgpool`` and ``blur`` are retained for ablation.

    Args:
        method: ``fft`` / ``fft_gaussian`` for frequency-domain filtering, or
            ``avgpool`` / ``blur`` for ablations.
        kernel_size: Odd spatial low-pass kernel size for avgpool/blur modes.
        cutoff_ratio: Normalized radial cut-off in [0, 0.5] for FFT modes.
        transition_width: Smoothness of the Gaussian radial roll-off.

    Input shape:
        x: [B, C, H, W]

    Output shapes:
        x_low: [B, C, H, W]
        x_high: [B, C, H, W]
    """

    def __init__(self, method="fft", kernel_size=5, cutoff_ratio=0.12, transition_width=0.08):
        super().__init__()
        self.method = str(method).lower()
        self.kernel_size = int(kernel_size)
        self.cutoff_ratio = float(cutoff_ratio)
        self.transition_width = float(transition_width)
        if self.kernel_size % 2 == 0:
            raise ValueError("kernel_size for FrequencyDecouple must be odd.")
        if self.method not in ("fft", "fft_gaussian", "avgpool", "blur"):
            raise ValueError(f"Unsupported freq_method={method}. Use fft, fft_gaussian, avgpool, or blur.")
        if not (0.0 < self.cutoff_ratio <= 0.5):
            raise ValueError("cutoff_ratio must be in (0, 0.5].")
        kernel = torch.tensor(
            [
                [1.0, 4.0, 6.0, 4.0, 1.0],
                [4.0, 16.0, 24.0, 16.0, 4.0],
                [6.0, 24.0, 36.0, 24.0, 6.0],
                [4.0, 16.0, 24.0, 16.0, 4.0],
                [1.0, 4.0, 6.0, 4.0, 1.0],
            ],
            dtype=torch.float32,
        )
        kernel = kernel / kernel.sum()
        self.register_buffer("blur_kernel", kernel.view(1, 1, 5, 5), persistent=False)
        self._mask_cache = {}

    def _fft_lowpass_mask(self, h, w, device, dtype):
        key = (h, w, str(device), dtype)
        cached = self._mask_cache.get(key)
        if cached is not None:
            return cached
        fy = torch.fft.fftfreq(h, d=1.0, device=device).view(h, 1)
        fx = torch.fft.fftfreq(w, d=1.0, device=device).view(1, w)
        radius = torch.sqrt(fx.square() + fy.square())
        transition = max(self.transition_width, 1e-6)
        mask = torch.sigmoid((self.cutoff_ratio - radius) / transition)
        mask = mask / mask.amax().clamp_min(1e-6)
        mask = mask.clamp(0.0, 1.0).to(dtype=dtype).view(1, 1, h, w)
        self._mask_cache[key] = mask
        return mask

    def _fft_decouple(self, x):
        orig_dtype = x.dtype
        x_float = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        _, _, h, w = x_float.shape
        spectrum = torch.fft.fft2(x_float, dim=(-2, -1), norm="ortho")
        mask = self._fft_lowpass_mask(h, w, x_float.device, x_float.dtype)
        low = torch.fft.ifft2(spectrum * mask, dim=(-2, -1), norm="ortho").real
        low = low.to(dtype=orig_dtype)
        high = x - low
        return low, high

    def forward(self, x):
        if self.method in ("fft", "fft_gaussian"):
            return self._fft_decouple(x)
        if self.method == "avgpool":
            pad = self.kernel_size // 2
            x_pad = F.pad(x, (pad, pad, pad, pad), mode="reflect")
            x_low = F.avg_pool2d(x_pad, kernel_size=self.kernel_size, stride=1, padding=0)
        else:
            c = x.shape[1]
            weight = self.blur_kernel.to(dtype=x.dtype, device=x.device).repeat(c, 1, 1, 1)
            x_pad = F.pad(x, (2, 2, 2, 2), mode="reflect")
            x_low = F.conv2d(x_pad, weight, groups=c)
        x_high = x - x_low
        return x_low, x_high


class PETFrequencyProxyGenerator(nn.Module):
    """Generate CT-conditioned PET low/high-frequency proxies.

    The learnable missing token is used only as conditional information and is
    concatenated with CT low/high components before convolutional prediction.

    Input shapes:
        ct_low: [B, C, H, W]
        ct_high: [B, C, H, W]
        pet_available: [B] or [B, 1], 1 for available and 0 for missing

    Output shapes:
        pet_low_proxy: [B, C, H, W]
        pet_high_proxy: [B, C, H, W]
    """

    def __init__(self, in_channels, hidden_channels=32, token_channels=8):
        super().__init__()
        self.in_channels = int(in_channels)
        self.token_channels = int(token_channels)
        self.missing_token = nn.Parameter(torch.zeros(1, self.token_channels, 1, 1))
        self.available_token = nn.Parameter(torch.zeros(1, self.token_channels, 1, 1))
        self.net = nn.Sequential(
            ConvBNAct(self.in_channels * 2 + self.token_channels, hidden_channels, kernel_size=3),
            ConvBNAct(hidden_channels, hidden_channels, kernel_size=3),
            nn.Conv2d(hidden_channels, self.in_channels * 2, kernel_size=1),
        )
        nn.init.normal_(self.missing_token, mean=0.0, std=0.02)
        nn.init.normal_(self.available_token, mean=0.0, std=0.02)

    def forward(self, ct_low, ct_high, pet_available):
        b, _, h, w = ct_low.shape
        if pet_available is None:
            pet_available = torch.ones(b, device=ct_low.device, dtype=ct_low.dtype)
        pet_available = pet_available.to(device=ct_low.device, dtype=ct_low.dtype).view(b, 1, 1, 1)
        cond_token = pet_available * self.available_token + (1.0 - pet_available) * self.missing_token
        cond_token = cond_token.expand(b, -1, h, w)
        proxy = self.net(torch.cat([ct_low, ct_high, cond_token], dim=1))
        pet_low_proxy, pet_high_proxy = torch.chunk(proxy, 2, dim=1)
        return pet_low_proxy, pet_high_proxy


def _unwrap_state_dict(state_dict):
    if isinstance(state_dict, dict):
        for key in ('state_dict', 'model', 'module'):
            if key in state_dict and isinstance(state_dict[key], dict):
                state_dict = state_dict[key]
                break
    return state_dict


def _sanitize_state_key(key):
    for prefix in ('module.', 'backbone.', 'visual.'):
        if key.startswith(prefix):
            key = key[len(prefix):]
    return key


def _load_local_weights_expand_first_conv(model, path, name='MAFD_Encoder'):
    if not path:
        print(f'[-] {name}: pretrained path not provided; training from scratch')
        return
    import os
    if os.path.isdir(path):
        for cand in ('pytorch_model.bin', 'model.safetensors', 'mit_b1.pth', 'mit-b1.pth', 'mit_b0.pth', 'mit-b0.pth'):
            full = os.path.join(path, cand)
            if os.path.exists(full):
                path = full
                break
    if not os.path.exists(path):
        print(f'[-] {name}: pretrained path not found: {path}; training from scratch')
        return
    if str(path).endswith('.safetensors'):
        from safetensors.torch import load_file
        state_dict = load_file(path, device='cpu')
    else:
        state_dict = torch.load(path, map_location='cpu', weights_only=False)
    state_dict = {_sanitize_state_key(k): v for k, v in _unwrap_state_dict(state_dict).items()}
    model_state = model.state_dict()
    loadable = {}
    expanded = []
    for key, value in state_dict.items():
        candidates = [key]
        if key.startswith('segformer.'):
            suffix = key[len('segformer.'):]
            candidates.extend([suffix, 'model.' + suffix])
        if key.startswith('model.'):
            candidates.append(key[len('model.'):])
        matched = None
        for cand in candidates:
            if cand not in model_state:
                continue
            target = model_state[cand]
            if target.shape == value.shape:
                matched = value
                break
            if value.ndim == 4 and target.ndim == 4 and value.shape[0] == target.shape[0] and value.shape[2:] == target.shape[2:]:
                if target.shape[1] % value.shape[1] == 0:
                    repeat = target.shape[1] // value.shape[1]
                    matched = value.repeat(1, repeat, 1, 1) / float(repeat)
                    expanded.append(cand)
                    break
                if target.shape[1] > value.shape[1]:
                    matched = value.mean(dim=1, keepdim=True).repeat(1, target.shape[1], 1, 1)
                    expanded.append(cand)
                    break
        if matched is not None:
            loadable[cand] = matched
    if loadable:
        msg = model.load_state_dict(loadable, strict=False)
        print(
            f'[PRETRAIN] {name}: loaded_tensors={len(loadable)} '
            f'expanded_first_conv={expanded[:4]} missing_after_load={len(msg.missing_keys)}'
        )
    else:
        print(f'[-] {name}: no compatible tensors were loaded; training from scratch')


class FrequencyFeatureFusion(nn.Module):
    """Fuse low/high encoder features with 1x1 Conv + BN + ReLU at each scale."""

    def __init__(self, low_channels, high_channels, out_channels=None):
        super().__init__()
        out_channels = list(out_channels or low_channels)
        self.fuse = nn.ModuleList([
            ConvBNAct(c_low + c_high, c_out, kernel_size=1)
            for c_low, c_high, c_out in zip(low_channels, high_channels, out_channels)
        ])

    def forward(self, low_feats, high_feats):
        fused = []
        for low_feat, high_feat, fuse in zip(low_feats, high_feats, self.fuse):
            if high_feat.shape[-2:] != low_feat.shape[-2:]:
                high_feat = F.interpolate(high_feat, size=low_feat.shape[-2:], mode="bilinear", align_corners=False)
            fused.append(_sanitize(fuse(torch.cat([low_feat, high_feat], dim=1))))
        return fused


class MAFDNet(nn.Module):
    """Modality-Availability-aware Frequency Decoupling Network.

    CT and PET are decomposed into low/high components. PET-missing samples use
    CT-conditioned PET frequency proxies, keeping the low/high encoder input
    structure identical for full-PET and missing-PET inference.
    """

    def __init__(
        self,
        img_channels=3,
        num_classes=1,
        encoder_name="mit_b1",
        pretrained=True,
        freq_method="fft",
        use_pet_proxy=True,
        proxy_loss_weight=0.05,
        consistency_loss_weight=0.0,
        ct_pretrained_path=None,
        pet_pretrained_path=None,
        decoder_channels=(512, 256, 128, 64),
        use_deep_supervision=False,
        **kwargs,
    ):
        super().__init__()
        self.img_channels = int(img_channels)
        self.num_classes = int(num_classes)
        self.encoder_name = encoder_name
        self.use_pet_proxy = bool(use_pet_proxy)
        self.proxy_loss_weight = float(proxy_loss_weight)
        self.consistency_loss_weight = float(consistency_loss_weight)
        self.use_deep_supervision = bool(use_deep_supervision)

        self.freq_decouple = FrequencyDecouple(method=freq_method, kernel_size=5)
        self.proxy_generator = PETFrequencyProxyGenerator(self.img_channels, hidden_channels=32)

        encoder_in_channels = self.img_channels * 2
        self.low_encoder = create_feature_backbone(encoder_name, in_channels=encoder_in_channels)
        self.high_encoder = create_feature_backbone(encoder_name, in_channels=encoder_in_channels)

        if pretrained:
            _load_local_weights_expand_first_conv(self.low_encoder, ct_pretrained_path or pet_pretrained_path, name="MAFD_Low_Encoder")
            _load_local_weights_expand_first_conv(self.high_encoder, pet_pretrained_path or ct_pretrained_path, name="MAFD_High_Encoder")

        low_channels = self.low_encoder.feature_info.channels()
        high_channels = self.high_encoder.feature_info.channels()
        self.freq_fusion = FrequencyFeatureFusion(low_channels, high_channels, out_channels=low_channels)
        self.shared_decoder = UNetStyleDecoder(
            low_channels,
            decoder_channels=decoder_channels,
            out_channels=self.num_classes,
            use_deep_supervision=self.use_deep_supervision,
        )

    @staticmethod
    def _to_nch(x, channels):
        if x is None:
            return None
        if x.shape[1] == channels:
            return x
        if x.shape[1] == 1 and channels == 3:
            return x.repeat(1, 3, 1, 1)
        if channels == 1 and x.shape[1] > 1:
            return x[:, :1]
        raise ValueError(f"Expected {channels} input channels, got {x.shape[1]}.")

    @staticmethod
    def _normalize_pet_available(pet_available, batch_size, device):
        if pet_available is None:
            return torch.ones(batch_size, device=device, dtype=torch.float32)
        return pet_available.to(device=device, dtype=torch.float32).view(batch_size).clamp(0.0, 1.0)

    def forward(self, ct, pet=None, pet_available=None, return_aux=False, target_size=None):
        """
        Args:
            ct: [B, C, H, W]
            pet: [B, C, H, W] or None
            pet_available: [B] or [B, 1], 1 means PET available, 0 means PET missing
            return_aux: return frequency tensors for optional proxy loss
            target_size: optional decoder output size, default CT spatial size
        """
        if target_size is None:
            target_size = ct.shape[-2:]
        ct = self._to_nch(ct, self.img_channels)
        pet = self._to_nch(pet, self.img_channels)
        b = ct.shape[0]
        pet_available = self._normalize_pet_available(pet_available, b, ct.device)

        ct_low, ct_high = self.freq_decouple(ct)
        pet_low_proxy, pet_high_proxy = self.proxy_generator(ct_low, ct_high, pet_available)

        if pet is not None:
            pet_low_real, pet_high_real = self.freq_decouple(pet)
        else:
            pet_low_real = torch.zeros_like(pet_low_proxy)
            pet_high_real = torch.zeros_like(pet_high_proxy)

        if self.use_pet_proxy:
            mask = pet_available.view(b, 1, 1, 1)
            pet_low_used = mask * pet_low_real + (1.0 - mask) * pet_low_proxy
            pet_high_used = mask * pet_high_real + (1.0 - mask) * pet_high_proxy
        else:
            pet_low_used = pet_low_real
            pet_high_used = pet_high_real

        low_input = torch.cat([ct_low, pet_low_used], dim=1)
        high_input = torch.cat([ct_high, pet_high_used], dim=1)
        low_feats = self.low_encoder(low_input)
        high_feats = self.high_encoder(high_input)
        _check_tensor_list("mafd_low_feats", low_feats)
        _check_tensor_list("mafd_high_feats", high_feats)

        fused_feats = self.freq_fusion(low_feats, high_feats)
        dec_out = self.shared_decoder(fused_feats, target_size)
        outputs = self._finalize_decoder_output(dec_out)
        _check_tensor("mafd_logits", outputs["logits"])
        outputs["pred"] = outputs["logits"]

        if return_aux:
            outputs.update({
                "ct_low": ct_low,
                "ct_high": ct_high,
                "pet_low_used": pet_low_used,
                "pet_high_used": pet_high_used,
                "pet_low_proxy": pet_low_proxy,
                "pet_high_proxy": pet_high_proxy,
                "pet_low_real": pet_low_real,
                "pet_high_real": pet_high_real,
                "pet_available": pet_available,
                "aux": {},
            })
            return outputs
        return outputs["logits"]

    @staticmethod
    def _finalize_decoder_output(dec_out):
        if isinstance(dec_out, dict):
            out = {"logits": _sanitize(dec_out["logits"])}
            if "aux_logits" in dec_out:
                out["aux_logits"] = [_sanitize(x) for x in dec_out["aux_logits"]]
            return out
        return {"logits": _sanitize(dec_out)}
