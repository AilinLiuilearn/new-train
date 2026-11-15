"""CT-only pre-concat alignment; raw PET -> concat/compress -> unchanged GN decoder."""
from variants.tsbci.model import SHCIPETCT
from models.components.concat_fusion import ConcatFusion
from models.staged_petct import CHANNELS


class ConcatSingle(SHCIPETCT):
    def __init__(self,cfg,arm='full',load_generic=True,text_features=None):
        # Construct shared modules in exactly the old order, then remove the unused branches.
        # This preserves the seeded encoder/decoder tensors and the subsequent loader RNG stream.
        super().__init__(cfg,arm='m1',load_generic=load_generic)
        pet_channels=self.enc_pet.feature_info.channels()
        if tuple(pet_channels)!=tuple(CHANNELS):raise ValueError('This recipe requires native MiT-B1 channels')
        del self.pet_align,self.fusion
        # The existing concat fusion is a Conv1x1 compressor, initialized to [I,I].
        # It is not a separate modality aligner and keeps decoder inputs unchanged.
        self.fusion=ConcatFusion(CHANNELS)
        self.arm='full'

    def forward(self,ct,pet,capture=False):
        if ct.shape!=pet.shape:raise ValueError('Paired CT/PET inputs required')
        c=self.enc_ct(self._input(ct));p=self.enc_pet(self._input(pet))
        for x,y in zip(c,p):
            if x.shape[0]!=y.shape[0] or x.shape[-2:]!=y.shape[-2:]:
                raise ValueError('Encoder grids differ; no implicit spatial resampling in this baseline')
        ca=self.ct_align(c)
        # MiT LayerNorm may return FP32 while CT's convolution is autocast FP16.
        # Match arithmetic dtype only; PET has no learned projection or normalization here.
        fusion_pet=[y.to(dtype=x.dtype) for x,y in zip(ca,p)]
        f=self.fusion(ca,fusion_pet)
        o={'logits':self.decoder_full(f,ct.shape[-2:])['logits']}
        if capture:o['capture']={'ct_raw':c,'ct_aligned':ca,'pet_raw':p,'fused':f}
        return o
