"""V1 changes only crop location; baseline geometry/interpolation is reused."""
import cv2
import numpy as np
from utils.image_augmentation import randomShiftScaleRotate, randomHorizontalFlip

CONTEXT_RULE = 'max(4,round(0.1*max(target_width,target_height)));clip_to_image'
COUNTERS = ('samples', 'crop_triggered', 'guided_attempt', 'guided_success', 'random_crop',
            'selected_1', 'selected_2', 'selected_foreground', 'selected_hard',
            'both_available', 'selected_1_when_both', 'selected_2_when_both',
            'fallback_no_candidate', 'fallback_vanished', 'fallback_target_too_large', 'background_target',
            'window_iou_sum', 'pixel_change_sum', 'target_coverage_sum',
            'foreground_fraction_sum', 'target_area_sum', 'target_foreground_sum',
            'crop_size_sum', 'center_x_sum', 'center_y_sum', 'region_selected',
            'guided_window_iou_sum', 'guided_pixel_change_sum') + tuple(f'center_bin_{i}' for i in range(16))


def components(binary):
    return cv2.connectedComponentsWithStats(np.ascontiguousarray(binary, dtype=np.uint8), connectivity=8)


def choose_window(mask, difference, mode, side, region_sampling='uniform'):
    """Uniform over 8-connected regions (after class choice), then legal positions.

    All intersecting GT components join the selected region before adding context.
    The context is clipped at image boundaries, never by changing crop size.
    """
    if region_sampling not in ('uniform', 'sqrt_area'):
        raise ValueError('Invalid region sampling')
    h, w = mask.shape
    fg = mask >= 127.5
    info = {'both_available': 0, 'selected_1_when_both': 0, 'selected_2_when_both': 0,
            'background_target': 0, 'target_area_sum': 0., 'target_foreground_sum': 0.}
    if mode == 'dual_state':
        available = [i for i in (1, 2) if np.any(difference == i)]
        if not available:
            return None, None, dict(info, fallback='no_candidate')
        both = len(available) == 2
        category = available[np.random.randint(len(available))] if both else available[0]
        info['both_available'] = int(both)
        info[f'selected_{category}_when_both'] = int(both)
        selected = difference == category
        info['selection'] = str(category)
    else:
        selected = fg if mode == 'foreground' else difference != 0
        info['selection'] = mode
    n, labels, stats, _ = components(selected)
    if n <= 1:
        return None, None, dict(info, fallback='no_candidate')
    if region_sampling == 'uniform':
        label = np.random.randint(1, n)  # Keep v1 RNG exactly.
    else:
        weights = np.sqrt(stats[1:, cv2.CC_STAT_AREA].astype(np.float64))
        label = int(np.random.choice(np.arange(1, n), p=weights/weights.sum()))
    region = labels == label
    info['target_area_sum'] = int(region.sum())
    info['target_foreground_sum'] = int((region & fg).sum())
    _, foreground_labels, _, _ = components(fg)
    touching = np.unique(foreground_labels[region])
    touching = touching[touching != 0]
    target = region | np.isin(foreground_labels, touching)
    info['background_target'] = int(len(touching) == 0)
    yy, xx = np.where(target)
    x0, x1, y0, y1 = xx.min(), xx.max()+1, yy.min(), yy.max()+1
    context = max(4, round(.1*max(x1-x0, y1-y0)))
    bbox = tuple(int(v) for v in (max(0, x0-context), max(0, y0-context),
                                  min(w, x1+context), min(h, y1+context)))
    x0, y0, x1, y1 = bbox
    lo_x, hi_x = max(0, x1-side), min(x0, w-side)
    lo_y, hi_y = max(0, y1-side), min(y0, h-side)
    if lo_x > hi_x or lo_y > hi_y:
        return None, region, dict(info, fallback='target_too_large', bbox=bbox)
    x, y = np.random.randint(lo_x, hi_x+1), np.random.randint(lo_y, hi_y+1)
    return (int(x), int(y), int(side), int(side)), region, dict(info, bbox=bbox)


def crop_with_guidance(image, mask, difference, mode, probability, u=.6, region_sampling='uniform'):
    # Baseline draws the side BEFORE the trigger, including when no crop occurs.
    side = np.int32(image.shape[0]*np.random.uniform(.7, .9))
    info = {key: 0. for key in COUNTERS}
    info['samples'] = 1.
    trace = {'window': None, 'random_window': None, 'target_bbox': None}
    if np.random.random() >= u:
        return image, mask, difference, info, trace
    h, w = image.shape[:2]
    # Always draw the original random window first: same-sized counterfactual.
    y = np.random.randint(0, max(0, h-side+1))
    x = np.random.randint(0, max(0, w-side+1))
    window = (int(x), int(y), int(side), int(side))
    trace['random_window'] = window
    info['crop_triggered'] = 1.
    region = None
    if probability > 0 and np.random.random() < probability:
        info['guided_attempt'] = 1.
        chosen, region, detail = choose_window(mask, difference, mode, side, region_sampling=region_sampling)
        for key in ('both_available', 'selected_1_when_both', 'selected_2_when_both',
                    'background_target', 'target_area_sum', 'target_foreground_sum'):
            info[key] = detail[key]
        if 'selection' in detail:
            info['selected_'+detail['selection']] = 1.
        trace['target_bbox'] = detail.get('bbox')
        if chosen is None:
            info['fallback_'+detail['fallback']] = 1.
        else:
            window = chosen
            info['guided_success'] = 1.
    info['random_crop'] = 1.-info['guided_success']
    rx, ry, _, _ = trace['random_window']
    x, y, _, _ = window
    intersection = max(0, side-abs(x-rx))*max(0, side-abs(y-ry))
    info['window_iou_sum'] = float(intersection/(2*side*side-intersection))
    patch = image[y:y+side, x:x+side]
    random_patch = image[ry:ry+side, rx:rx+side]
    info['pixel_change_sum'] = float(np.abs(patch.astype(float)-random_patch.astype(float)).mean()/255.)
    if region is not None:
        info['region_selected'] = 1.
        info['target_coverage_sum'] = float(region[y:y+side, x:x+side].sum()/region.sum())
    info['foreground_fraction_sum'] = float((mask[y:y+side, x:x+side] >= 127.5).mean())
    info['crop_size_sum'] = float(side)
    info['center_x_sum'] = float((x+side/2)/w)
    info['center_y_sum'] = float((y+side/2)/h)
    info[f'center_bin_{min(3,int((y+side/2)/h*4))*4+min(3,int((x+side/2)/w*4))}'] = 1.
    if info['guided_success']:
        info['guided_window_iou_sum'] = info['window_iou_sum']
        info['guided_pixel_change_sum'] = info['pixel_change_sum']
    trace['window'] = window
    image = cv2.resize(patch, (w, h), interpolation=cv2.INTER_CUBIC)
    mask = cv2.resize(mask[y:y+side, x:x+side], (w, h), interpolation=cv2.INTER_CUBIC)
    difference = cv2.resize(difference[y:y+side, x:x+side], (w, h), interpolation=cv2.INTER_NEAREST)
    return image, mask, difference, info, trace


def augment(image, mask, difference, mode, probability, return_trace=False, region_sampling='uniform'):
    had_candidate = (np.any(mask >= 127.5) if mode == 'foreground' else
                     np.any(difference != 0) if mode == 'hard' else
                     np.any((difference == 1) | (difference == 2)))
    image, mask, difference = randomShiftScaleRotate(image, mask, auxiliary=difference)
    image, mask, difference = randomHorizontalFlip(image, mask, auxiliary=difference)
    # Audit uses the exact transformed inputs immediately before crop.
    before = (image.copy(), mask.copy(), difference.copy()) if return_trace else None
    image, mask, difference, info, trace = crop_with_guidance(
        image, mask, difference, mode, probability, region_sampling=region_sampling)
    if had_candidate and info['fallback_no_candidate']:
        info['fallback_no_candidate'] = 0.
        info['fallback_vanished'] = 1.
    if return_trace:
        return image, mask, difference, info, trace, before
    return image, mask, info


def summarize(counts):
    s = {k: float(v) for k, v in counts.items()}
    def rate(a, b):
        return s.get(a, 0.)/max(s.get(b, 0.), 1.)
    s.update(crop_rate=rate('crop_triggered', 'samples'),
             attempt_per_crop=rate('guided_attempt', 'crop_triggered'),
             success_per_attempt=rate('guided_success', 'guided_attempt'),
             success_per_sample=rate('guided_success', 'samples'),
             selected_1_given_both=rate('selected_1_when_both', 'both_available'),
             selected_2_given_both=rate('selected_2_when_both', 'both_available'),
             mean_window_iou=rate('window_iou_sum', 'crop_triggered'),
             mean_pixel_change=rate('pixel_change_sum', 'crop_triggered'),
             mean_guided_window_iou=rate('guided_window_iou_sum', 'guided_success'),
             mean_guided_pixel_change=rate('guided_pixel_change_sum', 'guided_success'),
             mean_selected_region_coverage=rate('target_coverage_sum', 'region_selected'),
             mean_foreground_fraction=rate('foreground_fraction_sum', 'crop_triggered'),
             mean_selected_region_area=rate('target_area_sum', 'guided_attempt'),
             selected_region_foreground_fraction=rate('target_foreground_sum', 'target_area_sum'))
    return s
