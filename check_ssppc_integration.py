import torch

from models.ssppc_module import SpatialSemanticPairedPrototypeCompensation


def _random_feats(channels, sizes, batch_size=2, device='cpu'):
    return [torch.randn(batch_size, c, h, w, device=device, requires_grad=True) for c, (h, w) in zip(channels, sizes)]


def main():
    torch.manual_seed(2026)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    channels = [64, 128, 320, 512]
    sizes = [(128, 128), (64, 64), (32, 32), (16, 16)]
    model = SpatialSemanticPairedPrototypeCompensation(channels=channels, outlier_ratio=0.05, cache_on_cpu=False).to(device)

    ct = _random_feats(channels, sizes, device=device)
    pet = _random_feats(channels, sizes, device=device)
    mask = torch.zeros(2, 1, 512, 512, device=device)
    mask[:, :, 128:384, 128:384] = 1.0

    print('ready_before:', model.is_ready())
    for _ in range(3):
        model.collect(ct, pet, mask)
    report = model.finalize_epoch()
    diag = model.prototype_diagnostics()
    print('ready_after:', model.is_ready())
    print('report:', report)
    print('diag:', diag)

    comp_full = model(ct)
    comp_missing, debug = model(ct, return_debug=True)
    routed = model.route_pet_features(pet, comp_missing, pet_missing=torch.tensor([1, 0], device=device))
    print('comp_shapes:', [tuple(x.shape) for x in comp_missing])
    print('debug_tumor_shapes:', [tuple(d['tumor_probability'].shape) for d in debug])
    print('routed_shapes:', [tuple(x.shape) for x in routed])

    use_ssppc_false = SpatialSemanticPairedPrototypeCompensation(channels=channels, outlier_ratio=0.05, cache_on_cpu=False).to(device)
    zeros = use_ssppc_false(ct)
    print('zeros_before_ready:', [float(x.abs().sum().item()) for x in zeros])

    logits = sum(x.mean() for x in comp_missing)
    logits.backward()
    print('ct_grad_exists:', all(x.grad is not None for x in ct))
    print('pet_grad_exists:', all(x.grad is not None for x in pet))

    print('nan_inf_check:', all(torch.isfinite(x).all() for x in comp_missing))


if __name__ == '__main__':
    main()
