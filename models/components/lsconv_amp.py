"""CUDA AMP decorator compatibility for the vendored official SKA operator."""
try:
    from torch.amp import custom_fwd, custom_bwd
except ImportError:
    # Older PyTorch exposes CUDA-only decorators without a device_type argument.
    from torch.cuda.amp import custom_fwd as _cuda_custom_fwd
    from torch.cuda.amp import custom_bwd as _cuda_custom_bwd

    def custom_fwd(fwd=None, *, device_type, cast_inputs=None):
        if device_type != 'cuda':
            raise ValueError('LSConv legacy AMP compatibility supports CUDA only')
        return _cuda_custom_fwd(fwd=fwd, cast_inputs=cast_inputs)

    def custom_bwd(bwd=None, *, device_type):
        if device_type != 'cuda':
            raise ValueError('LSConv legacy AMP compatibility supports CUDA only')
        if bwd is None:
            return _cuda_custom_bwd
        return _cuda_custom_bwd(bwd)
