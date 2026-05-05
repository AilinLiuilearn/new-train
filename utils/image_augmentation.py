import cv2
import numpy as np
import math


def randomShiftScaleRotate(image, mask,
                           shift_limit=(-0.15, 0.15),
                           scale_limit=(-0.12, 0.12),
                           aspect_limit=(-0.1, 0.1),
                           rotate_limit=(-45, 45),
                           borderMode=cv2.BORDER_CONSTANT, u=0.7):
    if np.random.random() < u:
        height, width, channel = image.shape
        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)
        rad = angle / 180.0 * math.pi
        cc = math.cos(rad) * sx
        ss = math.sin(rad) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])
        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height]])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])
        mat = cv2.getPerspectiveTransform(box0.astype(np.float32), box1.astype(np.float32))
        image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR,
                                    borderMode=borderMode, borderValue=(0, 0, 0))
        mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_LINEAR,
                                   borderMode=borderMode, borderValue=(0, 0, 0))
    return image, mask


def randomHorizontalFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)
    return image, mask


def randomVerticalFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 0)
        mask = cv2.flip(mask, 0)
    return image, mask


def randomcrop(image, mask, u=0.6):
    crop_rate = np.random.uniform(0.7, 0.9)
    height = np.int32(image.shape[0] * crop_rate)
    width = height
    if np.random.random() < u:
        h, w, c = image.shape
        y = np.random.randint(0, max(0, h - height + 1))
        x = np.random.randint(0, max(0, w - width + 1))
        image = cv2.resize(image[y:y + height, x:x + width, :], (w, h), interpolation=cv2.INTER_CUBIC)
        mask_crop = mask[y:y + height, x:x + width]
        mask = cv2.resize(mask_crop, (w, h), interpolation=cv2.INTER_CUBIC)
    return image, mask


def randomcrop_lesion_center(image, mask, u=0.5, crop_range=(0.82, 0.94), jitter_ratio=0.08):
    if np.random.random() >= u:
        return image, mask

    h, w, c = image.shape
    pos = np.argwhere(mask > 127)
    if pos.size == 0:
        return randomcrop(image, mask, u=1.0)

    crop_rate = np.random.uniform(crop_range[0], crop_range[1])
    crop_h = max(8, int(h * crop_rate))
    crop_w = max(8, int(w * crop_rate))

    cy, cx = pos[np.random.randint(len(pos))]
    max_jy = max(1, int(crop_h * jitter_ratio))
    max_jx = max(1, int(crop_w * jitter_ratio))
    cy = int(np.clip(cy + np.random.randint(-max_jy, max_jy + 1), 0, h - 1))
    cx = int(np.clip(cx + np.random.randint(-max_jx, max_jx + 1), 0, w - 1))

    y1 = int(np.clip(cy - crop_h // 2, 0, max(0, h - crop_h)))
    x1 = int(np.clip(cx - crop_w // 2, 0, max(0, w - crop_w)))
    y2 = y1 + crop_h
    x2 = x1 + crop_w

    image = cv2.resize(image[y1:y2, x1:x2, :], (w, h), interpolation=cv2.INTER_CUBIC)
    mask = cv2.resize(mask[y1:y2, x1:x2], (w, h), interpolation=cv2.INTER_NEAREST)
    return image, mask


def elasticTransform(image, mask, alpha=80, sigma=8, u=0.5):
    """
    弹性形变（Elastic Transform）：对医学肿瘤边界非常重要。
    alpha: 形变强度；sigma: 高斯平滑核大小（控制形变平滑度）
    """
    if np.random.random() >= u:
        return image, mask
    h, w = image.shape[:2]
    dx = cv2.GaussianBlur(
        (np.random.rand(h, w) * 2 - 1).astype(np.float32), (0, 0), sigma
    ) * alpha
    dy = cv2.GaussianBlur(
        (np.random.rand(h, w) * 2 - 1).astype(np.float32), (0, 0), sigma
    ) * alpha
    gx, gy = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (gx + dx).astype(np.float32)
    map_y = (gy + dy).astype(np.float32)
    image_out = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT_101)
    mask_out = cv2.remap(mask, map_x, map_y, cv2.INTER_NEAREST,
                         borderMode=cv2.BORDER_REFLECT_101)
    return image_out, mask_out
