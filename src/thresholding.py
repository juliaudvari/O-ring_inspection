from __future__ import annotations

import numpy as np


def compute_histogram(gray_image: np.ndarray) -> np.ndarray:
    if gray_image.ndim != 2:
        raise ValueError("Expected a 2D grayscale image.")
    if gray_image.dtype != np.uint8:
        gray_image = gray_image.astype(np.uint8, copy=False)
    # count each gray value
    return np.bincount(gray_image.ravel(), minlength=256).astype(np.int64)


def otsu_threshold(gray_image: np.ndarray) -> tuple[int, np.ndarray]:
    # get image hist
    hist = compute_histogram(gray_image).astype(np.float64)
    total = float(gray_image.size)
    bins = np.arange(256, dtype=np.float64)

    # split weights
    cumulative_weight_background = np.cumsum(hist)
    cumulative_weight_foreground = total - cumulative_weight_background

    # split sums
    cumulative_sum_background = np.cumsum(hist * bins)
    global_sum = cumulative_sum_background[-1]

    mean_background = np.divide(
        cumulative_sum_background,
        cumulative_weight_background,
        out=np.zeros_like(cumulative_sum_background),
        where=cumulative_weight_background > 0,
    )
    mean_foreground = np.divide(
        global_sum - cumulative_sum_background,
        cumulative_weight_foreground,
        out=np.zeros_like(cumulative_sum_background),
        where=cumulative_weight_foreground > 0,
    )

    between_class_variance = (
        cumulative_weight_background
        * cumulative_weight_foreground
        * (mean_background - mean_foreground) ** 2
    )
    # best cut point
    threshold = int(np.argmax(between_class_variance))
    return threshold, hist.astype(np.int64)


def apply_threshold(gray_image: np.ndarray, threshold: int) -> np.ndarray:
    # make binary mask
    mask = gray_image <= threshold
    # keep ring as fg
    if float(mask.mean()) > 0.5:
        mask = ~mask
    return mask


def otsu_threshold_mask(gray_image: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    threshold, hist = otsu_threshold(gray_image)
    mask = apply_threshold(gray_image, threshold)
    return threshold, mask, hist
