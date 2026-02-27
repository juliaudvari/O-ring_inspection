from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import cv2
import numpy as np

from ccl import largest_component
from classify import classify_ring
from morphology import binary_closing
from thresholding import otsu_threshold_mask


@dataclass
class ImageProcessingResult:
    filename: str
    threshold: int
    status: str
    time_ms: float
    continuity: float
    thickness_median: float
    thickness_mad: float
    reason: str
    output_path: Path


def _overlay_mask(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = image_bgr.copy()
    if not np.any(mask):
        return out
    alpha = 0.35
    green = np.array([0.0, 255.0, 0.0], dtype=np.float32)
    out[mask] = np.clip((1.0 - alpha) * out[mask].astype(np.float32) + alpha * green, 0, 255).astype(np.uint8)
    return out


def process_image(image_path: Path, output_dir: Path) -> ImageProcessingResult:
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    color = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if gray is None or color is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    start = time.perf_counter()
    threshold, threshold_mask, _ = otsu_threshold_mask(gray)
    closed_mask = binary_closing(threshold_mask, ksize=5, iterations=2)
    largest_mask, _, _, _ = largest_component(closed_mask, connectivity=8)
    result = classify_ring(largest_mask, n_angles=360)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    annotated = _overlay_mask(color, largest_mask)
    cv2.putText(
        annotated,
        f"{result.status} | t={threshold} | {elapsed_ms:.2f} ms",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0) if result.status == "PASS" else (0, 0, 255),
        1,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{image_path.stem}_result.png"
    cv2.imwrite(str(output_path), annotated)

    return ImageProcessingResult(
        filename=image_path.name,
        threshold=threshold,
        status=result.status,
        time_ms=elapsed_ms,
        continuity=result.continuity,
        thickness_median=result.thickness_median,
        thickness_mad=result.thickness_mad,
        reason=result.reason,
        output_path=output_path,
    )
