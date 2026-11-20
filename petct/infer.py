"""Export direct and case-refined masks for one directory of paired PNG slices."""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from .case_refinement import refine_case, valid_neighbors
from .model import load_model


IMAGE_SIZE = 512
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def read_png(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img


def model_input(image):
    if image.shape != (IMAGE_SIZE, IMAGE_SIZE):
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
    rgb = np.repeat((image.astype(np.float32) / 255.)[None], 3, axis=0)
    return torch.from_numpy((rgb - MEAN) / STD)


def paired_case(directory, require_consecutive=True):
    ct = {p.name[:-7]: p for p in directory.glob('*_CT.png')}
    pet = {p.name[:-8]: p for p in directory.glob('*_PET.png')}
    if not ct or ct.keys() != pet.keys():
        raise ValueError('Expected matching <case>_<slice>_CT.png and _PET.png files')
    keys = [name.rsplit('_', 1) for name in ct]
    if len({k[0] for k in keys}) != 1 or any(len(k) != 2 or not k[1].isdigit() for k in keys):
        raise ValueError('Expected numbered slices from a single case')
    ids = sorted(ct, key=lambda name: int(name.rsplit('_', 1)[1]))
    sequence = [int(name.rsplit('_', 1)[1]) for name in ids]
    if len(set(sequence)) != len(sequence):
        raise ValueError('Duplicate slice numbers')
    if require_consecutive and any(b - a != 1 for a, b in zip(sequence, sequence[1:])):
        raise ValueError('Case refinement requires consecutive slice numbers')
    ct_images = [read_png(ct[name]) for name in ids]
    pet_images = [read_png(pet[name]) for name in ids]
    if any(a.shape != b.shape for a, b in zip(ct_images, pet_images)):
        raise ValueError('Aligned CT/PET source image sizes must match')
    return ids, ct_images, pet_images


@torch.inference_mode()
def predict_slices(model, ct_images, pet_images, batch_size=8):
    if len(ct_images) != len(pet_images) or not ct_images or batch_size < 1:
        raise ValueError('Expected at least one CT/PET pair and a positive batch size')
    device = next(model.parameters()).device
    masks, logits = [], []
    for offset in range(0, len(ct_images), batch_size):
        portion = slice(offset, offset + batch_size)
        ct = torch.stack([model_input(image) for image in ct_images[portion]]).to(device)
        pet = torch.stack([model_input(image) for image in pet_images[portion]]).to(device)
        output = model(ct, pet)
        logits.append(output['native_logits'][:, 0].float().cpu().numpy())
        masks.append((output['logits'][:, 0].float().sigmoid() > .5).cpu().numpy())
    return np.concatenate(masks), np.concatenate(logits)


def save_masks(directory, ids, masks):
    directory.mkdir()
    for identity, mask in zip(ids, masks):
        path = directory / f'{identity}_mask.png'
        if not cv2.imwrite(str(path), mask.astype(np.uint8) * 255):
            raise OSError(f'Could not write {path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--case-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--direct-only', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output directory')
    cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ids, ct, pet = paired_case(args.case_dir, require_consecutive=not args.direct_only)
    model = load_model(args.checkpoint, args.device)
    masks, logits = predict_slices(model, ct, pet, args.batch_size)
    refined = None if args.direct_only else refine_case(masks, logits, ct, valid_neighbors(ct, pet))
    args.output.mkdir(parents=True)
    save_masks(args.output / 'direct', ids, masks)
    if refined is not None:
        save_masks(args.output / 'refined', ids, refined)


if __name__ == '__main__':
    main()
