"""PET-CT model using four-scale TACS sparse channel-spatial fusion."""
import torch

from models.dual_shared_add_baseline import DualSharedAddPETCTBaseline
from models.components.tacs_fusion import TACSFusion


class FullTACSPETCT(DualSharedAddPETCTBaseline):
    """Keep the heterogeneous encoders and GN decoder; replace only Full fusion.

    The inherited Missing path never calls the fusion module and remains
    CT-only. The TACS module is initialized inside fork_rng so it does not
    shift the RNG stream used by the shared baseline model.
    """

    def __init__(
        self,
        image_size=512,
        tacs_channel_ratio=0.25,
        tacs_text_embedding_path=None,
        tacs_use_text=False,
        **kwargs,
    ):
        kwargs["baseline"] = "full_add"
        super().__init__(**kwargs)
        rng_state = torch.get_rng_state()
        with torch.random.fork_rng(devices=[]):
            self.fusion = TACSFusion(
                image_size=image_size,
                channel_ratio=tacs_channel_ratio,
                text_embedding_path=tacs_text_embedding_path,
                use_text=tacs_use_text,
            )
        torch.set_rng_state(rng_state)


def build_tacs_model(cfg):
    if cfg.baseline != "full_add":
        raise ValueError("TACS is a Full CT+PET fusion module; use --baseline full_add")
    model = FullTACSPETCT(
        image_size=cfg.image_size_2d,
        tacs_channel_ratio=cfg.tacs_channel_ratio,
        tacs_text_embedding_path=cfg.tacs_text_embedding_path,
        tacs_use_text=cfg.tacs_use_text,
        ct_backbone=cfg.ct_backbone,
        pet_backbone=cfg.pet_backbone,
        ct_pretrained_path=cfg.ct_pretrained_path if cfg.pretrained else None,
        pet_pretrained_path=cfg.pet_pretrained_path if cfg.pretrained else None,
        pretrained=bool(cfg.pretrained),
        decoder_channels=tuple(cfg.decoder_channels),
        use_deep_supervision=bool(cfg.use_deep_supervision or cfg.deep_supervision),
    )
    print(
        f"[tacs] scales=S1..S4 channels={model.fusion.selected_channels} "
        f"ratio={cfg.tacs_channel_ratio} grid={model.fusion.grid_sizes} "
        f"text={cfg.tacs_use_text} "
        f"fusion_params={sum(p.numel() for p in model.fusion.parameters())} "
        f"total_params={sum(p.numel() for p in model.parameters())}",
        flush=True,
    )
    return {"model": model}
