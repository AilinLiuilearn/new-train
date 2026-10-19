"""Official nnU-Net network implementation; custom 2D two-modality wrapper.

No copied/reimplemented U-Net. Uses MIC-DKFZ dynamic-network-architectures.
This is a fixed network configuration, NOT the full nnU-Net planning pipeline.
Inputs: normalized float B,1,H,W. Output: binary logits B,1,H,W.
"""
import torch
from torch import nn
from dynamic_network_architectures.architectures.unet import PlainConvUNet
from dynamic_network_architectures.initialization.weight_init import InitWeights_He

DEFAULT_FEATURES = (32, 64, 128, 256, 512, 512)

class Expert2D(nn.Module):
    def __init__(self, features=DEFAULT_FEATURES):
        super().__init__()
        self.features = tuple(features)
        self.divisor = 2 ** (len(features) - 1)
        self.network = PlainConvUNet(
            input_channels=1, n_stages=len(features), features_per_stage=features,
            conv_op=nn.Conv2d, kernel_sizes=3, strides=[1] + [2] * (len(features)-1),
            n_conv_per_stage=2, num_classes=1,
            n_conv_per_stage_decoder=[2] * (len(features)-1), conv_bias=True,
            norm_op=nn.InstanceNorm2d, norm_op_kwargs={'eps':1e-5, 'affine':True},
            nonlin=nn.LeakyReLU, nonlin_kwargs={'negative_slope':1e-2, 'inplace':True},
            deep_supervision=False)
        self.network.apply(InitWeights_He(1e-2))

    def forward(self, x, return_descriptor=False):
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError('expected B,1,H,W input')
        if any(s % self.divisor or s // self.divisor < 2 for s in x.shape[-2:]):
            raise ValueError(f'H,W must be multiples of {self.divisor}, bottleneck >= 2')
        skips = self.network.encoder(x)
        logits = self.network.decoder(skips)
        if return_descriptor:
            return logits, skips[-1].mean(dim=(-2,-1))
        return logits

class DualExpert2D(nn.Module):
    def __init__(self, features=DEFAULT_FEATURES, gate_hidden=64):
        super().__init__()
        self.ct = Expert2D(features)
        self.pet = Expert2D(features)
        self.gate = nn.Sequential(nn.Linear(2*features[-1], gate_hidden), nn.ReLU(),
                                  nn.Linear(gate_hidden, 2))
        # Initially exactly equal logits weighting; gradients learn routing later.
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def forward(self, ct, pet=None, mode='full'):
        if mode == 'missing':
            zc = self.ct(ct)
            weights = zc.new_tensor([1.,0.]).expand(ct.shape[0], 2)
            return {'logits':zc, 'ct_logits':zc, 'weights':weights}
        if mode != 'full':
            raise ValueError(f'unknown mode {mode}')
        if pet is None or pet.shape != ct.shape:
            raise ValueError('Full requires CT and PET with identical B,1,H,W shapes')
        zc, vc = self.ct(ct, True)
        zp, vp = self.pet(pet, True)
        # Float32 softmax and weighted logits also under AMP.
        weights = self.gate(torch.cat([vc,vp], dim=1)).float().softmax(dim=1)
        zf = weights[:,0,None,None,None]*zc.float() + weights[:,1,None,None,None]*zp.float()
        return {'logits':zf, 'ct_logits':zc, 'pet_logits':zp, 'weights':weights}
