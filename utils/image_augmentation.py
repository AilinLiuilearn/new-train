import cv2
import numpy as np
import math


def randomShiftScaleRotate(image, mask,
                           shift_limit=(-0.15, 0.15),
                           scale_limit=(-0.12, 0.12),
                           aspect_limit=(-0.1, 0.1),
                           rotate_limit=(-45, 45),
                           borderMode=cv2.BORDER_CONSTANT, u=0.7, auxiliary=None):
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
        if auxiliary is not None:
            auxiliary = cv2.warpPerspective(auxiliary, mat, (width, height),
                                            flags=cv2.INTER_NEAREST,
                                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if auxiliary is not None:
        return image, mask, auxiliary
    return image, mask


def randomHorizontalFlip(image, mask, u=0.5, auxiliary=None):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)
        if auxiliary is not None:
            auxiliary = cv2.flip(auxiliary, 1)
    if auxiliary is not None:
        return image, mask, auxiliary
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
