from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import cv2
import numpy as np

from ccl import colorize_labels, largest_component
from classify import ClassificationResult, classify_ring
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
    outlier_fraction: float
    max_angular_gap: float
    reason: str
    output_path: Path


def _overlay_mask(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    # draw green mask
    result = image_bgr.copy()
    if not np.any(mask):
        return result

    alpha = 0.35
    green = np.array([0.0, 255.0, 0.0], dtype=np.float32)
    blended = (1.0 - alpha) * result[mask].astype(np.float32) + alpha * green
    result[mask] = np.clip(blended, 0, 255).astype(np.uint8)
    return result


def _annotate_image(
    color_image: np.ndarray,
    largest_mask: np.ndarray,
    largest_stats: dict[str, int | tuple[int, int, int, int]] | None,
    classification: ClassificationResult,
    threshold: int,
    time_ms: float,
) -> np.ndarray:

    annotated = _overlay_mask(color_image, largest_mask)
    status_color = (0, 200, 0) if classification.status == "PASS" else (0, 0, 255)

    if largest_stats is not None:
        # draw box
        x_min, y_min, x_max, y_max = largest_stats["bbox"]
        cv2.rectangle(
            annotated,
            (int(x_min), int(y_min)),
            (int(x_max), int(y_max)),
            (0, 255, 255),
            1,
        )

    centroid_row, centroid_col = classification.centroid_rc
    center = (int(round(centroid_col)), int(round(centroid_row)))
    if 0 <= center[0] < annotated.shape[1] and 0 <= center[1] < annotated.shape[0]:
        # draw center
        cv2.circle(annotated, center, 3, (255, 0, 0), -1)

    line1 = f"{classification.status} | threshold={threshold} | {time_ms:.2f} ms"
    line2 = (
        f"continuity={classification.continuity:.3f} "
        f"gap={classification.max_angular_gap:.1f}deg "
        f"med={classification.thickness_median:.2f}"
    )
    line3 = classification.reason
    if len(line3) > 85:
        line3 = line3[:82] + "..."

    cv2.putText(annotated, line1, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
    cv2.putText(annotated, line2, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    cv2.putText(annotated, line3, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return annotated


def _save_debug_outputs(
    output_dir: Path,
    image_stem: str,
    threshold_mask: np.ndarray,
    closed_mask: np.ndarray,
    largest_mask: np.ndarray,
    labels: np.ndarray,
) -> None:
    # make debug folder
    debug_dir = output_dir / "debug" / image_stem
    debug_dir.mkdir(parents=True, exist_ok=True)

    # save debug masks
    cv2.imwrite(str(debug_dir / "threshold_mask.png"), threshold_mask.astype(np.uint8) * 255)
    cv2.imwrite(str(debug_dir / "closed_mask.png"), closed_mask.astype(np.uint8) * 255)
    cv2.imwrite(
        str(debug_dir / "largest_component_mask.png"),
        largest_mask.astype(np.uint8) * 255,
    )
    cv2.imwrite(str(debug_dir / "labeled_visualization.png"), colorize_labels(labels))


def process_image(
    image_path: Path,
    output_dir: Path,
    debug: bool = False,
    ksize: int = 3,
    morph_iterations: int = 1,
    n_angles: int = 360,
) -> ImageProcessingResult:
    # load input image
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    color = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if gray is None or color is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    # start process timer
    start_time = time.perf_counter()

    threshold, threshold_mask, _ = otsu_threshold_mask(gray)
    closed_mask = binary_closing(threshold_mask, ksize=ksize, iterations=morph_iterations)
    largest_mask, labels, largest_stats, _ = largest_component(
        closed_mask, connectivity=8
    )
    _, _, _, raw_stats = largest_component(threshold_mask, connectivity=8)

    # class + time
    classification = classify_ring(
        largest_mask,
        n_angles=n_angles,
        all_component_stats=raw_stats,
    )
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    # draw final image
    annotated = _annotate_image(
        color_image=color,
        largest_mask=largest_mask,
        largest_stats=largest_stats,
        classification=classification,
        threshold=threshold,
        time_ms=elapsed_ms,
    )

    # save result image
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{image_path.stem}_result.png"
    cv2.imwrite(str(output_path), annotated)

    if debug:
        # save debug views
        _save_debug_outputs(
            output_dir=output_dir,
            image_stem=image_path.stem,
            threshold_mask=threshold_mask,
            closed_mask=closed_mask,
            largest_mask=largest_mask,
            labels=labels,
        )

    return ImageProcessingResult(
        filename=image_path.name,
        threshold=threshold,
        status=classification.status,
        time_ms=elapsed_ms,
        continuity=classification.continuity,
        thickness_median=classification.thickness_median,
        thickness_mad=classification.thickness_mad,
        outlier_fraction=classification.outlier_fraction,
        max_angular_gap=classification.max_angular_gap,
        reason=classification.reason,
        output_path=output_path,
    )
