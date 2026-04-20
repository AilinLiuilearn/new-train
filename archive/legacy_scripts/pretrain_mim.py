# -*- coding: utf-8 -*-
"""
Stage 1: PET/CT 跨模态掩码重建预训练（MIM）
- 非侵入式：复用 SegEncoderEfficientB2，不改现有分割代码
- 独立入口：pretrain_mim.py
- 输出：仅保存 encoder 权重，供 Stage 2 strict=False 加载
"""

import os
import json
import argparse
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from contextlib import nullcontext
from torch.utils.data import DataLoader, random_split

from configs.base import str2bool
from datasets.pclt20k_seg import PCLT20KSegDataset, _read_list, _collect_records
from models.seg_backbone import SegEncoderEfficientB2


class ComplementaryMasker(nn.Module):
    """互补掩码：CT 与 PET 在 patch 级别互补遮挡，A+B=全图。"""

    def __init__(self, patch_size=16, mask_ratio=0.5, fill_value=0.0):
        super().__init__()
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.fill_value = fill_value

    @torch.no_grad()
    def forward(self, x_ct, x_pet):
        # x_*: (B, C, 512, 512)
        b, _, h, w = x_ct.shape
        gh, gw = h // self.patch_size, w // self.patch_size  # 32 x 32

        # 1 表示“该模态被遮挡”
        m_ct_patch = (torch.rand(b, 1, gh, gw, device=x_ct.device) < self.mask_ratio).float()
        m_pet_patch = 1.0 - m_ct_patch  # 互补

        m_ct = F.interpolate(m_ct_patch, size=(h, w), mode='nearest')
        m_pet = F.interpolate(m_pet_patch, size=(h, w), mode='nearest')

        x_ct_masked = x_ct * (1.0 - m_ct) + self.fill_value * m_ct
        x_pet_masked = x_pet * (1.0 - m_pet) + self.fill_value * m_pet
        return x_ct_masked, x_pet_masked, m_ct, m_pet


class MIMReconstructor(nn.Module):
    """
    跨模态重建包装器
    - enc_ct / enc_pet: MiT 编码器
    - 输入使用 stage4 (16x16) 特征
    - Decoder: ConvTranspose2d strides = 4, 4, 2 -> 16 -> 64 -> 256 -> 512
    - 输出 2 通道: [rec_pet, rec_ct]，范围约 [-1.6, 1.6]
    """

    def __init__(self, backbone='mit_b2', pretrained_backbone=True, pretrained_path=None):
        super().__init__()
        self.enc_ct = SegEncoderEfficientB2(
            backbone_name=backbone,
            out_indices=(0, 1, 2, 3),
            pretrained=pretrained_backbone,
            pretrained_path=pretrained_path,
        )
        self.enc_pet = SegEncoderEfficientB2(
            backbone_name=backbone,
            out_indices=(0, 1, 2, 3),
            pretrained=pretrained_backbone,
            pretrained_path=pretrained_path,
        )

        c4 = self.enc_ct.out_channels  # mit_b1/b2: 512
        in_ch = c4 * 2
        self.up1 = self._make_up_block(in_ch, 512)   # 16 -> 32
        self.up2 = self._make_up_block(512, 256)     # 32 -> 64
        self.up3 = self._make_up_block(256, 128)     # 64 -> 128
        self.up4 = self._make_up_block(128, 64)      # 128 -> 256
        self.up5 = self._make_up_block(64, 32)       # 256 -> 512
        self.out_head = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, kernel_size=1),
            nn.Tanh(),
        )
        self._init_reconstruction_head()

    def _make_up_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.ReLU(inplace=True),
        )

    def _init_reconstruction_head(self):
        modules = [self.up1, self.up2, self.up3, self.up4, self.up5, self.out_head]
        for block in modules:
            for m in block.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.GroupNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

    def forward(self, x_ct_masked, x_pet_masked):
        f_ct4 = self.enc_ct(x_ct_masked, return_list=True)[-1]
        f_pet4 = self.enc_pet(x_pet_masked, return_list=True)[-1]
        z = torch.cat([f_ct4, f_pet4], dim=1)
        z = self.up1(z)
        z = self.up2(z)
        z = self.up3(z)
        z = self.up4(z)
        z = self.up5(z)
        out = self.out_head(z) * 1.6  # 匹配数据范围 [-1.6, 1.6]
        rec_pet = out[:, 0:1]
        rec_ct = out[:, 1:2]
        return rec_ct, rec_pet


def masked_l1(pred, target, mask):
    # 仅在被遮挡区域监督
    den = torch.clamp(mask.sum(), min=1.0)
    return torch.abs(pred - target).mul(mask).sum() / den


def masked_mse(pred, target, mask):
    den = torch.clamp(mask.sum(), min=1.0)
    return ((pred - target) ** 2).mul(mask).sum() / den


def compute_psnr(pred, target, data_range=3.2):
    mse = F.mse_loss(pred, target, reduction='mean')
    mse = torch.clamp(mse, min=1e-10)
    return 10.0 * torch.log10((data_range ** 2) / mse)


def masked_psnr(pred, target, mask, data_range=3.2):
    mse = ((pred - target) ** 2).mul(mask).sum() / torch.clamp(mask.sum(), min=1e-8)
    if mse < 1e-10:
        return torch.tensor(100.0, device=pred.device)
    return 10.0 * torch.log10((data_range ** 2) / mse)


def _to_u8(x):
    # x: (1,H,W), range [-1.6,1.6]
    x = (x + 1.6) / 3.2
    x = np.clip(x, 0.0, 1.0)
    x = (x * 255.0).astype(np.uint8)
    if x.ndim == 3 and x.shape[0] == 1:
        x = x[0]  # (H, W)
    return x


def save_reconstruction_results(save_dir, epoch, ct, pet, ct_masked, pet_masked, rec_ct, rec_pet):
    os.makedirs(save_dir, exist_ok=True)
    # 仅保存 batch 第一张
    ct0 = ct[0].detach().cpu().numpy()
    pet0 = pet[0].detach().cpu().numpy()
    ctm0 = ct_masked[0].detach().cpu().numpy()
    petm0 = pet_masked[0].detach().cpu().numpy()
    rct0 = rec_ct[0].detach().cpu().numpy()
    rpet0 = rec_pet[0].detach().cpu().numpy()

    row_ct = np.concatenate([_to_u8(ct0), _to_u8(ctm0), _to_u8(rct0)], axis=1)
    row_pet = np.concatenate([_to_u8(pet0), _to_u8(petm0), _to_u8(rpet0)], axis=1)
    canvas = np.concatenate([row_ct, row_pet], axis=0)
    out_path = os.path.join(save_dir, f'ep{epoch:03d}_recon.png')
    cv2.imwrite(out_path, canvas)


def build_stage2_train_records(root, random_state=2023):
    train_txt = os.path.join(root, 'train.txt')
    test_txt = os.path.join(root, 'test.txt')

    # 优先严格使用 Stage2 的 train.txt 母集
    if os.path.isfile(train_txt) and os.path.isfile(test_txt):
        train_ids = _read_list(train_txt)
        test_ids = _read_list(test_txt)
        train_records, _, _ = _collect_records(
            root,
            train_ids=train_ids,
            test_ids=test_ids,
            val_ids=None,
            val_ratio=0.0,
            random_state=random_state,
        )
    else:
        # 无官方划分文件时，退化为全量（并给出提示）
        print('[WARN] train.txt/test.txt 未找到，退化为全量样本做 Stage1 母集')
        train_records = []
        for name in sorted(os.listdir(root)):
            case_dir = os.path.join(root, name)
            if not os.path.isdir(case_dir):
                continue
            for fname in os.listdir(case_dir):
                if not fname.endswith('_CT.png'):
                    continue
                base = fname.replace('_CT.png', '')
                ct_path = os.path.join(case_dir, fname)
                pet_path = os.path.join(case_dir, f'{base}_PET.png')
                mask_path = os.path.join(case_dir, f'{base}_mask.png')
                if not os.path.isfile(mask_path) or not os.path.isfile(pet_path):
                    continue
                train_records.append({
                    'image_id': f"{name}_{base.split('_')[-1]}" if '_' in base else f"{name}_{base}",
                    'case_id': name,
                    'ct_path': ct_path,
                    'pet_path': pet_path,
                    'mask_path': mask_path,
                })

    # 仅保留完整 CT+PET 对
    train_records = [r for r in train_records if r.get('pet_path') is not None and os.path.isfile(r['pet_path'])]
    return train_records


def build_loaders(args):
    base_records = build_stage2_train_records(args.root, random_state=args.random_state)

    n_total = len(base_records)
    n_val = max(1, int(n_total * args.pretrain_val_ratio))
    n_train = n_total - n_val
    gen = torch.Generator().manual_seed(args.random_state)
    perm = torch.randperm(n_total, generator=gen).tolist()
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    train_records = [base_records[i] for i in train_idx]
    val_records = [base_records[i] for i in val_idx]

    train_ds = PCLT20KSegDataset(
        train_records,
        image_size=args.image_size_2d,
        train=True,
        pet_available_list=[True] * len(train_records),
        random_state=args.random_state,
        aug_strong=args.aug_strong,
    )
    val_ds = PCLT20KSegDataset(
        val_records,
        image_size=args.image_size_2d,
        train=False,
        pet_available_list=[True] * len(val_records),
        random_state=args.random_state,
        aug_strong=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )
    print(f'[MIM split from Stage2 train] total={n_total} train={n_train} val={n_val} (val_ratio={args.pretrain_val_ratio})')
    return train_loader, val_loader


def evaluate_recon(model, masker, loader, device, pet_loss_weight=1.5):
    model.eval()
    val_loss, val_mse, total_psnr, n = 0.0, 0.0, 0.0, 0
    with torch.no_grad():
        for batch in loader:
            ct = batch['ct'].float().to(device)
            pet = batch['pet'].float().to(device)
            ct_m, pet_m, m_ct, m_pet = masker(ct, pet)
            rec_ct, rec_pet = model(ct_m, pet_m)

            loss_ct = masked_l1(rec_ct, ct, m_ct)
            loss_pet = masked_l1(rec_pet, pet, m_pet)
            loss = loss_ct + pet_loss_weight * loss_pet

            mse_ct = masked_mse(rec_ct, ct, m_ct)
            mse_pet = masked_mse(rec_pet, pet, m_pet)
            mse = mse_ct + pet_loss_weight * mse_pet

            psnr_ct = masked_psnr(rec_ct, ct, m_ct)
            psnr_pet = masked_psnr(rec_pet, pet, m_pet)
            psnr = (psnr_ct + psnr_pet) / 2.0

            val_loss += float(loss.item()) * ct.size(0)
            val_mse += float(mse.item()) * ct.size(0)
            total_psnr += float(psnr.item()) * ct.size(0)
            n += ct.size(0)
    return val_loss / max(n, 1), val_mse / max(n, 1), total_psnr / max(n, 1)


def main():
    p = argparse.ArgumentParser('Stage1 MIM pretrain')
    p.add_argument('--root', type=str, default='../data/PCLT20K')
    p.add_argument('--backbone', type=str, default='mit_b2')
    p.add_argument('--pretrained_backbone', type=str2bool, default=True)
    p.add_argument('--pretrained_path', type=str, default=None)
    p.add_argument('--cipa_aligned', type=str2bool, default=True)
    p.add_argument('--use_case_split', type=str2bool, default=False)
    p.add_argument('--val_ratio', type=float, default=0.1)
    p.add_argument('--pretrain_val_ratio', type=float, default=0.1)
    p.add_argument('--image_size_2d', type=int, default=512)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--random_state', type=int, default=2023)
    p.add_argument('--aug_strong', type=str2bool, default=False)

    p.add_argument('--patch_size', type=int, default=16)
    p.add_argument('--mask_ratio', type=float, default=0.4)

    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--learning_rate', type=float, default=8e-5)
    p.add_argument('--warmup_lr', type=float, default=5e-6)
    p.add_argument('--warmup_epochs', type=int, default=5)
    p.add_argument('--weight_decay', type=float, default=0.05)
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--mixed_precision', type=str2bool, default=True)
    p.add_argument('--gradient_accumulation_steps', type=int, default=1)
    p.add_argument('--pet_loss_weight', type=float, default=2.0)

    p.add_argument('--checkpoint_root', type=str, default='./checkpoints_new/')
    p.add_argument('--save_every', type=int, default=5)
    p.add_argument('--early_stop_patience', type=int, default=10)
    args = p.parse_args()

    torch.manual_seed(args.random_state)
    torch.cuda.manual_seed_all(args.random_state)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    stamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    ckpt_dir = os.path.join(args.checkpoint_root, 'MIM', stamp)
    os.makedirs(ckpt_dir, exist_ok=True)

    with open(os.path.join(ckpt_dir, 'config_args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    train_loader, val_loader = build_loaders(args)

    model = MIMReconstructor(
        backbone=args.backbone,
        pretrained_backbone=args.pretrained_backbone,
        pretrained_path=args.pretrained_path,
    ).to(device)
    masker = ComplementaryMasker(patch_size=args.patch_size, mask_ratio=args.mask_ratio, fill_value=0.0).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler() if (args.mixed_precision and device.type == 'cuda') else None

    def set_lr(optim, lr):
        for pg in optim.param_groups:
            pg['lr'] = lr

    log_csv = os.path.join(ckpt_dir, 'pretrain_log.csv')
    with open(log_csv, 'w') as f:
        f.write('epoch,train_loss,val_loss,val_mse,train_psnr,val_psnr\n')

    best_val = 1e9
    best_epoch = 0
    best_val_psnr = 0.0
    no_improve = 0

    for ep in range(1, args.epochs + 1):
        cur_lr = args.warmup_lr if ep <= args.warmup_epochs else args.learning_rate
        set_lr(optimizer, cur_lr)
        model.train()
        s_loss, s_psnr, n = 0.0, 0.0, 0
        vis_cache = None
        acc_steps = max(1, int(args.gradient_accumulation_steps))
        optimizer.zero_grad(set_to_none=True)
        for i, batch in enumerate(train_loader):
            ct = batch['ct'].float().to(device)
            pet = batch['pet'].float().to(device)

            ct_m, pet_m, m_ct, m_pet = masker(ct, pet)

            if scaler is not None:
                amp_ctx = torch.cuda.amp.autocast()
            else:
                amp_ctx = nullcontext()
            if scaler is not None:
                with amp_ctx:
                    rec_ct, rec_pet = model(ct_m, pet_m)
                    loss_ct = masked_l1(rec_ct, ct, m_ct)
                    loss_pet = masked_l1(rec_pet, pet, m_pet)
                    loss_raw = loss_ct + args.pet_loss_weight * loss_pet
                    loss = loss_raw / acc_steps
                scaler.scale(loss).backward()
            else:
                with amp_ctx:
                    rec_ct, rec_pet = model(ct_m, pet_m)
                    loss_ct = masked_l1(rec_ct, ct, m_ct)
                    loss_pet = masked_l1(rec_pet, pet, m_pet)
                    loss_raw = loss_ct + args.pet_loss_weight * loss_pet
                    loss = loss_raw / acc_steps
                loss.backward()

            do_step = ((i + 1) % acc_steps == 0) or ((i + 1) == len(train_loader))
            if do_step:
                if scaler is not None:
                    if args.grad_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    if args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            psnr_ct = masked_psnr(rec_ct.detach(), ct, m_ct)
            psnr_pet = masked_psnr(rec_pet.detach(), pet, m_pet)
            psnr = (psnr_ct + psnr_pet) / 2.0

            s_loss += float(loss_raw.item())
            s_psnr += float(psnr.item())
            n += 1

            if vis_cache is None:
                vis_cache = (ct, pet, ct_m, pet_m, rec_ct, rec_pet)

            if (i + 1) % 50 == 0:
                print(f'Ep{ep}[{i+1}/{len(train_loader)}] loss={loss_raw.item():.4f} psnr={psnr.item():.2f}')

            if (i + 1) % 100 == 0:
                ct_mask_vals = ct_m[m_ct > 0.5]
                pet_mask_vals = pet_m[m_pet > 0.5]
                ct_mask_min = ct_mask_vals.min().item() if ct_mask_vals.numel() > 0 else float('nan')
                ct_mask_max = ct_mask_vals.max().item() if ct_mask_vals.numel() > 0 else float('nan')
                ct_mask_mean = ct_mask_vals.mean().item() if ct_mask_vals.numel() > 0 else float('nan')
                pet_mask_min = pet_mask_vals.min().item() if pet_mask_vals.numel() > 0 else float('nan')
                pet_mask_max = pet_mask_vals.max().item() if pet_mask_vals.numel() > 0 else float('nan')
                pet_mask_mean = pet_mask_vals.mean().item() if pet_mask_vals.numel() > 0 else float('nan')
                print(
                    f"[Debug Ep{ep} It{i+1}] "
                    f"CT(masked_region)=[{ct_mask_min:.3f},{ct_mask_max:.3f}] mean={ct_mask_mean:.3f} "
                    f"PET(masked_region)=[{pet_mask_min:.3f},{pet_mask_max:.3f}] mean={pet_mask_mean:.3f} "
                    f"PredCT=[{rec_ct.min().item():.3f},{rec_ct.max().item():.3f}] "
                    f"PredPET=[{rec_pet.min().item():.3f},{rec_pet.max().item():.3f}] "
                    f"TargetCT=[{ct.min().item():.3f},{ct.max().item():.3f}] "
                    f"TargetPET=[{pet.min().item():.3f},{pet.max().item():.3f}] "
                    f"mask_ct_mean={m_ct.mean().item():.3f}"
                )
                if (rec_ct.min().item() > 1.55 and rec_ct.max().item() > 1.55) or (rec_ct.min().item() < -1.55 and rec_ct.max().item() < -1.55):
                    print('[WARN] PredCT may be saturated near Tanh boundary')
                if (rec_pet.min().item() > 1.55 and rec_pet.max().item() > 1.55) or (rec_pet.min().item() < -1.55 and rec_pet.max().item() < -1.55):
                    print('[WARN] PredPET may be saturated near Tanh boundary')

        train_loss = s_loss / max(n, 1)
        train_psnr = s_psnr / max(n, 1)
        val_loss, val_mse, val_psnr = evaluate_recon(model, masker, val_loader, device, pet_loss_weight=args.pet_loss_weight)

        with open(log_csv, 'a') as f:
            f.write(f'{ep},{train_loss:.6f},{val_loss:.6f},{val_mse:.6f},{train_psnr:.4f},{val_psnr:.4f}\n')

        print(f'Epoch {ep}: lr={cur_lr:.2e} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_mse={val_mse:.4f} train_psnr={train_psnr:.2f} val_psnr={val_psnr:.2f}')

        if vis_cache is not None:
            save_reconstruction_results(
                save_dir=os.path.join(ckpt_dir, 'recon_vis'),
                epoch=ep,
                ct=vis_cache[0],
                pet=vis_cache[1],
                ct_masked=vis_cache[2],
                pet_masked=vis_cache[3],
                rec_ct=vis_cache[4],
                rec_pet=vis_cache[5],
            )

        enc_ckpt = {
            'epoch': ep,
            'enc_ct': model.enc_ct.state_dict(),
            'enc_pet': model.enc_pet.state_dict(),
            'backbone': args.backbone,
        }

        if ep % args.save_every == 0:
            torch.save(enc_ckpt, os.path.join(ckpt_dir, f'encoder_ep{ep:03d}.pth'))

        if val_mse < best_val:
            best_val = val_mse
            best_val_psnr = val_psnr
            best_epoch = ep
            no_improve = 0
            torch.save(enc_ckpt, os.path.join(ckpt_dir, 'encoder_best.pth'))
        else:
            no_improve += 1

        if no_improve >= args.early_stop_patience:
            print(f'Early stopping at epoch {ep}: val_mse did not improve for {args.early_stop_patience} epochs')
            break

    torch.save({'enc_ct': model.enc_ct.state_dict(), 'enc_pet': model.enc_pet.state_dict()},
               os.path.join(ckpt_dir, 'encoder_last.pth'))

    with open(os.path.join(ckpt_dir, 'summary.json'), 'w') as f:
        json.dump({'best_epoch': best_epoch, 'best_val_mse': best_val, 'best_val_psnr': best_val_psnr}, f, indent=2)

    print(f'Finished. Best epoch={best_epoch}, best val mse={best_val:.6f}, best val psnr={best_val_psnr:.2f}')
    print(f'Checkpoint dir: {ckpt_dir}')


if __name__ == '__main__':
    main()
