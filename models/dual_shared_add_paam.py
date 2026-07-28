import torch

from models.baseline_blocks import _check_tensor, _check_tensor_list
from models.dual_shared_add_baseline import DualSharedAddPETCTBaseline
from models.paam_affine_action_memory import PETAffineActionMemory


class DualSharedAddPAAMPETCT(DualSharedAddPETCTBaseline):
    def __init__(self, *args, paam_k=8, **kwargs):
        super().__init__(*args, **kwargs)
        self.paam_k = int(paam_k)
        paam_channels = list(self.enc_pet.feature_info.channels())
        self.paam = PETAffineActionMemory(channels=paam_channels, K=self.paam_k)
        paam_params = sum(p.numel() for p in self.paam.parameters())
        print(f'[dual_shared_add_paam] fusion=PAAM paam_k={self.paam_k} paam_params={paam_params}')

    def _forward_paam(self, ct, pet, target_size, route, update_memory, capture_paam_visuals):
        ct_feats = self._encode_ct(ct)
        pet_feats = None
        if route == 'full':
            pet_feats = self._encode_pet(pet)
        elif self.training and pet is not None:
            with torch.no_grad():
                pet_feats = self._encode_pet(pet)
        fused_feats, paam_info = self.paam(
            ct_features=ct_feats,
            pet_features=pet_feats,
            route=route,
            update_memory=update_memory,
            capture_visuals=capture_paam_visuals,
        )
        out = self._decode(fused_feats, target_size)
        out['paam_info'] = paam_info
        return out

    def _forward_full(self, ct, pet, target_size, capture_paam_visuals=False):
        return self._forward_paam(ct, pet, target_size, route='full', update_memory=self.training, capture_paam_visuals=capture_paam_visuals)

    def _forward_missing(self, ct, pet, target_size, capture_paam_visuals=False):
        if self.training:
            return self._forward_paam(ct, pet, target_size, route='missing', update_memory=True, capture_paam_visuals=capture_paam_visuals)
        return self._forward_paam(ct, None, target_size, route='missing', update_memory=False, capture_paam_visuals=capture_paam_visuals)

    def _forward_auto(self, ct, pet, pet_available, target_size, capture_paam_visuals=False):
        if pet_available is None:
            pet_available = torch.ones(ct.shape[0], device=ct.device, dtype=torch.long)
        pet_available = pet_available.to(device=ct.device).long().view(-1)
        if pet_available.numel() != ct.shape[0]:
            raise ValueError('pet_available must contain one state per sample')
        if not torch.all((pet_available == 0) | (pet_available == 1)):
            raise ValueError('pet_available values must be 0 or 1')
        if torch.all(pet_available == 1):
            return self._forward_full(ct, pet, target_size, capture_paam_visuals=capture_paam_visuals)
        if torch.all(pet_available == 0):
            return self._forward_missing(ct, pet, target_size, capture_paam_visuals=capture_paam_visuals)
        raise NotImplementedError('PAAM currently supports only batch-level homogeneous modality state in auto mode')

    def begin_epoch(self, epoch):
        self.paam.begin_epoch(epoch)

    def finalize_epoch_memory(self):
        return self.paam.finalize_epoch_memory()

    def print_diagnostics(self):
        return self.paam.print_diagnostics()

    def export_diagnostics(self, *args, **kwargs):
        return self.paam.export_diagnostics(*args, **kwargs)

    def forward(self, ct, pet=None, pet_available=None, target_size=None, forward_mode='auto', capture_paam_visuals=False):
        if target_size is None:
            target_size = ct.shape[-2:]
        if forward_mode == 'full':
            return self._forward_full(ct, pet, target_size, capture_paam_visuals=capture_paam_visuals)
        if forward_mode == 'missing':
            return self._forward_missing(ct, pet, target_size, capture_paam_visuals=capture_paam_visuals)
        if forward_mode == 'auto':
            return self._forward_auto(ct, pet, pet_available, target_size, capture_paam_visuals=capture_paam_visuals)
        raise ValueError(f'Unsupported forward_mode={forward_mode!r}')
