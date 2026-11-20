"""Frozen inference refinement. Inputs contain no segmentation labels."""
import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy.special import expit


IN_PLANE = ndi.generate_binary_structure(2, 1)
IN_CASE = ndi.generate_binary_structure(3, 1)
CT_RADIUS = 3


def valid_neighbors(ct_images, pet_images):
    """PNG-space check used before tracking or bilateral completion."""
    if len(ct_images) != len(pet_images):
        raise ValueError('CT and PET must have the same slice count')
    if len(ct_images) < 5:
        return np.zeros(max(0, len(ct_images) - 1), dtype=bool)
    supported = []
    for images in (ct_images, pet_images):
        vectors = []
        for image in images:
            image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32)
            center = image[4:-4, 4:-4].ravel()
            center = center - center.mean()
            vectors.append(center / max(float(np.linalg.norm(center)), 1e-8))
        features = np.stack(vectors)
        similarity = features @ features.T
        supported.append(np.diag(similarity, 1) >= float(np.median(np.diag(similarity, 4))))
    return supported[0] & supported[1]


def track_ids(masks, valid):
    stops = [0, *(np.flatnonzero(~valid) + 1), len(masks)]
    labels = np.zeros(masks.shape, dtype=np.int32)
    offset = 0
    for start, stop in zip(stops, stops[1:]):
        local, count = ndi.label(masks[start:stop], structure=IN_CASE)
        labels[start:stop] = np.where(local > 0, local + offset, 0)
        offset += count
    return labels


def select_tracks(masks, labels, grid_logits):
    active = [np.unique(frame[frame > 0]) for frame in labels]
    candidates = {int(track) for present in active if len(present) > 1 for track in present}
    if not candidates:
        return masks
    bounds = ndi.find_objects(labels)
    probabilities = expit(grid_logits)
    scores = {}
    for track in candidates:
        slices = bounds[track - 1]
        region = labels[slices] == track
        prob = probabilities[slices]
        score = 0.
        for a, b, p, q in zip(region[:-1], region[1:], prob[:-1], prob[1:]):
            overlap = a & b
            score += np.minimum(p[overlap], q[overlap]).sum(dtype=np.float64) / np.count_nonzero(a | b)
        scores[track] = score
    result = masks.copy()
    for i, present in enumerate(active):
        if len(present) > 1:
            best = max(scores[k] for k in present)
            winners = [k for k in present if scores[k] == best]
            if best > 0 and len(winners) == 1:
                result[i] = labels[i] == winners[0]
    return result


def contour_and_fill(mask, grid_logit, ct_image):
    if not mask.any():
        return mask
    y, x = np.nonzero(mask)
    padding = 2 * CT_RADIUS + 3
    roi = np.s_[max(0, y.min() - padding):min(mask.shape[0], y.max() + padding + 1),
                max(0, x.min() - padding):min(mask.shape[1], x.max() + padding + 1)]
    local = mask[roi]
    foreground = ndi.binary_erosion(local, structure=IN_PLANE)
    background = ndi.binary_erosion(~local, structure=IN_PLANE)
    if not foreground.any() or not background.any():
        return ndi.binary_fill_holes(mask, structure=IN_PLANE)
    fg = ndi.distance_transform_edt(~foreground, return_distances=False, return_indices=True)
    bg = ndi.distance_transform_edt(~background, return_distances=False, return_indices=True)
    if ct_image.shape != mask.shape:
        ct_image = cv2.resize(ct_image, mask.shape[::-1], interpolation=cv2.INTER_LINEAR)
    gray = cv2.GaussianBlur(ct_image[roi].astype(np.float32), (3, 3), .65)
    f, b = gray[tuple(fg)], gray[tuple(bg)]
    evidence = (np.abs(gray - b) - np.abs(gray - f)) / (np.abs(f - b) + 1)
    margin = np.abs(np.tanh(grid_logit[roi] / 2))
    band = ndi.distance_transform_edt(local) + ndi.distance_transform_edt(~local) <= CT_RADIUS
    refined = mask.copy()
    refined[roi][band & (evidence > margin)] = True
    refined[roi][band & (evidence < -margin)] = False
    return ndi.binary_fill_holes(refined, structure=IN_PLANE)


def refine_case(masks, native_logits, ct_images, valid):
    """Apply the confirmed trajectory, CT contour, fill and bilateral operations."""
    if masks.shape != (len(ct_images), 512, 512) or native_logits.shape != (len(masks), 128, 128):
        raise ValueError('Expected 512-grid masks and native 128-grid logits')
    valid = np.asarray(valid, dtype=bool)
    if valid.shape != (len(masks) - 1,):
        raise ValueError('One validity flag is needed per adjacent slice pair')
    logits = ndi.zoom(native_logits, (1, 4, 4), order=1, mode='nearest', grid_mode=True)
    selected = select_tracks(masks, track_ids(masks, valid), logits)
    filled = np.stack([contour_and_fill(m, z, ct) for m, z, ct in zip(selected, logits, ct_images)])
    result = filled.copy()
    for i in range(1, len(filled) - 1):
        if valid[i - 1] and valid[i]:
            result[i] |= filled[i - 1] & filled[i + 1]
    return result
