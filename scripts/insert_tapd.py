# helper: insert TAPD_Module into build_mdt_seg.py and update tasks/mdt_seg.py
import re

# ── 1. build_mdt_seg.py ──────────────────────────────────────────────────────
build_path = '/root/autodl-tmp/mkd-main/new-train/models/build_mdt_seg.py'
code = open(build_path).read()

tapd = '''
class TAPD_Module(nn.Module):
    """
    Topology-Aware Prototype Decoupling
    returns (z_out, loss_dict); loss_dict has loss_ortho/loss_uni/loss_align during training
    """
    def __init__(self, in_channels, proj_channels=256, t_uniformity=2.0):
        super().__init__()
        self.t = t_uniformity
        self.proj_c_ct  = nn.Conv2d(in_channels, proj_channels, 1)
        self.proj_s_ct  = nn.Conv2d(in_channels, proj_channels, 1)
        self.proj_c_pet = nn.Conv2d(in_channels, proj_channels, 1)
        self.proj_s_pet = nn.Conv2d(in_channels, proj_channels, 1)
        self.topology_attn = nn.Sequential(
            nn.Conv2d(proj_channels, proj_channels // 2, 3, padding=1),
            nn.BatchNorm2d(proj_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(proj_channels // 2, 1, 1),
            nn.Sigmoid(),
        )
        self.reconstruct = nn.Conv2d(proj_channels * 3, in_channels, 1)

    def forward(self, z_ct, z_pet):
        c_ct  = self.proj_c_ct(z_ct)
        s_ct  = self.proj_s_ct(z_ct)
        c_pet = self.proj_c_pet(z_pet)
        s_pet = self.proj_s_pet(z_pet)
        fused_c  = c_ct + c_pet
        loss_dict = {}
        if self.training:
            attn_weight = self.topology_attn(fused_c)
            weight_sum  = attn_weight.sum(dim=(2, 3), keepdim=True) + 1e-5
            proto_s_ct  = (s_ct  * attn_weight).sum(dim=(2, 3), keepdim=True) / weight_sum
            proto_s_pet = (s_pet * attn_weight).sum(dim=(2, 3), keepdim=True) / weight_sum
            proto_c_ct  = (c_ct  * attn_weight).sum(dim=(2, 3), keepdim=True) / weight_sum
            proto_c_pet = (c_pet * attn_weight).sum(dim=(2, 3), keepdim=True) / weight_sum
            v_s_ct  = F.normalize(proto_s_ct.view(proto_s_ct.size(0), -1),  dim=1)
            v_s_pet = F.normalize(proto_s_pet.view(proto_s_pet.size(0), -1), dim=1)
            v_c_ct  = proto_c_ct.view(proto_c_ct.size(0), -1)
            v_c_pet = proto_c_pet.view(proto_c_pet.size(0), -1)
            dist_sq_ct  = 2.0 - 2.0 * torch.mm(v_s_ct,  v_s_ct.t())
            dist_sq_pet = 2.0 - 2.0 * torch.mm(v_s_pet, v_s_pet.t())
            loss_dict[\'loss_ortho\'] = torch.mean((v_s_ct * v_s_pet).sum(dim=1) ** 2)
            loss_dict[\'loss_uni\']   = (torch.log(torch.mean(torch.exp(-self.t * dist_sq_ct))) +
                                       torch.log(torch.mean(torch.exp(-self.t * dist_sq_pet)))) / 2.0
            loss_dict[\'loss_align\'] = F.mse_loss(v_c_ct, v_c_pet)
        z_out = self.reconstruct(torch.cat([fused_c, s_ct, s_pet], dim=1))
        return z_out, loss_dict

'''

anchor = 'def build_mdt_seg_teacher(config):'
if 'TAPD_Module' not in code:
    code = code.replace(anchor, tapd + anchor)
    print('[1] TAPD_Module inserted')
else:
    print('[1] TAPD_Module already present')

# replace fuse_2 / fuse_3 lines (handle both comment variants)
code = re.sub(
    r"fuse_2=.*?DeepDecoupledFusion\(ch\[2\]\)[^,\n]*",
    "fuse_2=TAPD_Module(in_channels=ch[2], proj_channels=128),   # deep: TAPD",
    code
)
code = re.sub(
    r"fuse_3=.*?DeepDecoupledFusion\(ch\[3\]\)[^,\n]*",
    "fuse_3=TAPD_Module(in_channels=ch[3], proj_channels=256),   # deep: TAPD",
    code
)
# remove stray comment line if present
code = re.sub(r"        # fuse_2=BoundaryAwareGatedFusion.*?\n", "", code)
print('[2] fuse_2/fuse_3 replaced')

open(build_path, 'w').write(code)
print('[build_mdt_seg.py] done')

# ── 2. tasks/mdt_seg.py ──────────────────────────────────────────────────────
task_path = '/root/autodl-tmp/mkd-main/new-train/tasks/mdt_seg.py'
tcode = open(task_path).read()

# Replace _forward: fuse_{i} now returns (out, loss_dict) for i in [2,3]
old_forward = '''def _forward(nets, ct, pet, target_size):
    feats_ct  = nets["enc_ct" ](ct,  return_list=True)
    feats_pet = nets["enc_pet"](pet, return_list=True)

    # --- 新增正则化：对所有尺度的骨干特征进行通道丢弃 ---
    feats_ct  = [nets["feature_dropout"](f) for f in feats_ct]
    feats_pet = [nets["feature_dropout"](f) for f in feats_pet]

    fused = [nets[f"fuse_{i}"](feats_ct[i], feats_pet[i]) for i in range(4)]
    return nets["segmentor"](fused, target_size=target_size)'''

new_forward = '''def _forward(nets, ct, pet, target_size, training=False):
    feats_ct  = nets["enc_ct" ](ct,  return_list=True)
    feats_pet = nets["enc_pet"](pet, return_list=True)

    # Dropout2d regularization before fusion
    feats_ct  = [nets["feature_dropout"](f) for f in feats_ct]
    feats_pet = [nets["feature_dropout"](f) for f in feats_pet]

    # Layer 0/1: BoundaryAwareGatedFusion (single output)
    fused_0 = nets["fuse_0"](feats_ct[0], feats_pet[0])
    fused_1 = nets["fuse_1"](feats_ct[1], feats_pet[1])
    # Layer 2/3: TAPD_Module (returns (out, loss_dict))
    fused_2, tapd_ld2 = nets["fuse_2"](feats_ct[2], feats_pet[2])
    fused_3, tapd_ld3 = nets["fuse_3"](feats_ct[3], feats_pet[3])

    fused = [fused_0, fused_1, fused_2, fused_3]
    seg_out = nets["segmentor"](fused, target_size=target_size)

    # Aggregate TAPD losses (only non-empty during training)
    tapd_losses = {}
    if tapd_ld2 or tapd_ld3:
        dev = feats_ct[0].device
        tapd_losses["loss_ortho"] = tapd_ld2.get("loss_ortho", torch.zeros(1, device=dev)) + \
                                     tapd_ld3.get("loss_ortho", torch.zeros(1, device=dev))
        tapd_losses["loss_uni"]   = tapd_ld2.get("loss_uni",   torch.zeros(1, device=dev)) + \
                                     tapd_ld3.get("loss_uni",   torch.zeros(1, device=dev))
        tapd_losses["loss_align"] = tapd_ld2.get("loss_align", torch.zeros(1, device=dev)) + \
                                     tapd_ld3.get("loss_align", torch.zeros(1, device=dev))
    return seg_out, tapd_losses'''

if old_forward in tcode:
    tcode = tcode.replace(old_forward, new_forward)
    print('[3] _forward updated')
else:
    print('[3] _forward anchor not found - check manually')

# Update train_step to use new _forward signature and collect TAPD losses
old_step = '''        logit = _forward(self.networks, ct, pet, mask.shape[-2:])
        loss_seg = self.loss_seg(logit, mask)

        # 支持外部传入动态 alpha（用于衰减），默认使用初始化时的值
        cur_orth = alpha_orth if alpha_orth is not None else self.alpha_orth
        cur_adv  = alpha_adv  if alpha_adv  is not None else self.alpha_adv

        logit = _forward(self.networks, ct, pet, mask.shape[-2:])
        loss_seg = self.loss_seg(logit, mask)

        loss_orth, loss_adv = _decouple_losses(
            self.networks, ct, cur_orth, cur_adv
        )
        total_loss = loss_seg + loss_orth + loss_adv

        loss_dict = {
            "loss_seg":  loss_seg.detach(),
            "loss_orth": loss_orth.detach(),
            "loss_adv":  loss_adv.detach(),
        }'''

# Simpler: just find the train_step body and replace
import re as _re
pattern = r'(    def train_step\(self, batch.*?\n)(.*?)(        return total_loss, logit, mask, loss_dict)'
match = _re.search(pattern, tcode, _re.DOTALL)
if match:
    new_body = match.group(1) + '''        ct   = batch["ct"].float().to(self.device)
        pet  = batch["pet"].float().to(self.device)
        mask = batch["mask"].float().to(self.device)

        cur_orth = alpha_orth if alpha_orth is not None else self.alpha_orth
        cur_adv  = alpha_adv  if alpha_adv  is not None else self.alpha_adv

        logit, tapd_losses = _forward(self.networks, ct, pet, mask.shape[-2:], training=True)
        loss_seg = self.loss_seg(logit, mask)

        # TAPD prototype losses
        w_ortho = getattr(self.config, "alpha_tapd_ortho", 0.1)
        w_uni   = getattr(self.config, "alpha_tapd_uni",   0.05)
        w_align = getattr(self.config, "alpha_tapd_align", 0.1)
        loss_tapd = (
            w_ortho * tapd_losses.get("loss_ortho", torch.zeros(1, device=self.device)) +
            w_uni   * tapd_losses.get("loss_uni",   torch.zeros(1, device=self.device)) +
            w_align * tapd_losses.get("loss_align", torch.zeros(1, device=self.device))
        ).squeeze()
        total_loss = loss_seg + loss_tapd

        loss_dict = {
            "loss_seg":   loss_seg.detach(),
            "loss_tapd":  loss_tapd.detach(),
            "loss_ortho": tapd_losses.get("loss_ortho", torch.zeros(1)).detach(),
            "loss_uni":   tapd_losses.get("loss_uni",   torch.zeros(1)).detach(),
            "loss_align": tapd_losses.get("loss_align", torch.zeros(1)).detach(),
        }
''' + match.group(3)
    tcode = tcode[:match.start()] + new_body + tcode[match.end():]
    print('[4] train_step updated')
else:
    print('[4] train_step pattern not matched')

# Also fix evaluate / collect_logits to use new _forward signature
tcode = tcode.replace(
    'logit = _forward(self.networks, ct, pet, mask.shape[-2:])',
    'logit, _ = _forward(self.networks, ct, pet, mask.shape[-2:])'
)
tcode = tcode.replace(
    'logits.append(_forward(self.networks, ct, pet, mask.shape[-2:]).cpu())',
    'logit_t, _ = _forward(self.networks, ct, pet, mask.shape[-2:])\n            logits.append(logit_t.cpu())'
)
print('[5] evaluate/collect patched')

open(task_path, 'w').write(tcode)
print('[tasks/mdt_seg.py] done')
