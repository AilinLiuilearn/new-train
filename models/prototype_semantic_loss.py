import torch
import torch.nn as nn
import torch.nn.functional as F


class PETPrototypeSemanticContrastiveLoss(nn.Module):
    """
    MCPL-inspired class-aware PET prototype semantic constraint.

    Classes:
        0 = background
        1 = foreground

    feature:
        [B, C, H, W]

    pet_prototypes:
        [2, K, C]

    prototype_ready:
        [2, K] bool

    The prototype bank is treated as a stop-gradient semantic anchor.
    """

    def __init__(self, temperature=0.5, eps=1e-8):
        super().__init__()
        self.temperature = float(temperature)
        self.eps = float(eps)

    def _masked_pool(self, feature, weight):
        """
        feature: [B,C,H,W]
        weight:  [B,1,H,W]

        return:
            descriptor: [B,C]
            valid:      [B] bool
        """
        denom = weight.sum(dim=(2, 3))  # [B,1]

        valid = denom[:, 0] > self.eps

        descriptor = (
            feature * weight
        ).sum(dim=(2, 3)) / denom.clamp_min(self.eps)

        descriptor = torch.where(
            valid[:, None],
            descriptor,
            torch.zeros_like(descriptor),
        )

        return descriptor, valid

    def _class_scores(
        self,
        descriptor,
        prototypes,
        ready,
    ):
        """
        descriptor: [N,C]
        prototypes: [2,K,C]
        ready:      [2,K]

        return:
            logits: [N,2]

        Each class score is aggregated over all ready prototypes
        of that class using logsumexp.
        """

        descriptor = F.normalize(
            descriptor.float(),
            p=2,
            dim=-1,
            eps=self.eps,
        )

        # Prototype bank is anchor only.
        prototypes = F.normalize(
            prototypes.detach().float(),
            p=2,
            dim=-1,
            eps=self.eps,
        )

        class_scores = []

        for class_idx in range(2):

            class_ready = ready[class_idx]

            class_proto = prototypes[
                class_idx,
                class_ready,
            ]  # [K_ready,C]

            if class_proto.shape[0] == 0:
                raise RuntimeError(
                    f"No ready prototype for class {class_idx}"
                )

            # [N,K_ready]
            similarity = torch.matmul(
                descriptor,
                class_proto.transpose(0, 1),
            )

            similarity = (
                similarity / self.temperature
            )

            # Aggregate the K prototypes belonging
            # to this semantic class.
            score = torch.logsumexp(
                similarity,
                dim=1,
            )  # [N]

            class_scores.append(score)

        # [N,2]
        logits = torch.stack(
            class_scores,
            dim=1,
        )

        return logits

    def forward(
        self,
        feature,
        mask,
        pet_prototypes,
        prototype_ready,
    ):
        """
        feature:
            Full    -> real PET S4 feature
            Missing -> retrieved PET proxy S4 feature

        mask:
            [B,1,H0,W0], GT tumor mask

        returns:
            loss
            diagnostics
        """

        if feature.ndim != 4:
            raise ValueError(
                f"feature must be [B,C,H,W], "
                f"got {tuple(feature.shape)}"
            )

        if mask.ndim != 4 or mask.shape[1] != 1:
            raise ValueError(
                f"mask must be [B,1,H,W], "
                f"got {tuple(mask.shape)}"
            )

        if pet_prototypes.ndim != 3:
            raise ValueError(
                "pet_prototypes must be [2,K,C]"
            )

        if prototype_ready.ndim != 2:
            raise ValueError(
                "prototype_ready must be [2,K]"
            )

        # Need at least one valid BG and one valid FG prototype.
        if (
            not bool(prototype_ready[0].any())
            or not bool(prototype_ready[1].any())
        ):
            zero = feature.new_zeros(())

            return zero, {
                "loss_fg": zero.detach(),
                "loss_bg": zero.detach(),
                "margin_fg": zero.detach(),
                "margin_bg": zero.detach(),
            }

        _, _, h, w = feature.shape

        # Resize GT mask to S4 resolution.
        # Soft mask is intentionally preserved.
        fg_mask = F.adaptive_avg_pool2d(
            mask.float(),
            output_size=(h, w),
        ).clamp(0.0, 1.0)

        bg_mask = 1.0 - fg_mask

        # Patient-level foreground/background descriptors.
        z_fg, valid_fg = self._masked_pool(
            feature,
            fg_mask,
        )

        z_bg, valid_bg = self._masked_pool(
            feature,
            bg_mask,
        )

        losses = []

        zero = feature.new_zeros(())

        loss_fg = zero
        loss_bg = zero

        margin_fg = zero
        margin_bg = zero

        # ---------------------------------
        # Foreground descriptor
        # target class = 1
        # ---------------------------------
        if bool(valid_fg.any()):

            fg_logits = self._class_scores(
                z_fg[valid_fg],
                pet_prototypes,
                prototype_ready,
            )

            fg_target = torch.ones(
                fg_logits.shape[0],
                dtype=torch.long,
                device=fg_logits.device,
            )

            loss_fg = F.cross_entropy(
                fg_logits,
                fg_target,
            )

            losses.append(loss_fg)

            # FG score - BG score
            margin_fg = (
                fg_logits[:, 1]
                - fg_logits[:, 0]
            ).mean()

        # ---------------------------------
        # Background descriptor
        # target class = 0
        # ---------------------------------
        if bool(valid_bg.any()):

            bg_logits = self._class_scores(
                z_bg[valid_bg],
                pet_prototypes,
                prototype_ready,
            )

            bg_target = torch.zeros(
                bg_logits.shape[0],
                dtype=torch.long,
                device=bg_logits.device,
            )

            loss_bg = F.cross_entropy(
                bg_logits,
                bg_target,
            )

            losses.append(loss_bg)

            # BG score - FG score
            margin_bg = (
                bg_logits[:, 0]
                - bg_logits[:, 1]
            ).mean()

        if len(losses) == 0:
            loss = zero
        else:
            loss = torch.stack(losses).mean()

        if not torch.isfinite(loss):
            raise RuntimeError(
                "PET prototype semantic loss "
                "contains NaN/Inf"
            )

        diagnostics = {
            "loss_fg": loss_fg.detach(),
            "loss_bg": loss_bg.detach(),
            "margin_fg": margin_fg.detach(),
            "margin_bg": margin_bg.detach(),
        }

        return loss, diagnostics