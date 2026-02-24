from __future__ import annotations

import numpy as np

NEIGHBORS_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
NEIGHBORS_8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]


def connected_components(mask: np.ndarray, connectivity: int = 8):
    # pick neighbor set
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8.")

    foreground = mask.astype(bool, copy=False)
    height, width = foreground.shape
    labels = np.zeros((height, width), dtype=np.int32)
    stats: list[dict[str, int | tuple[int, int, int, int]]] = []

    neighbors = NEIGHBORS_8 if connectivity == 8 else NEIGHBORS_4
    next_label = 0

    # scan all pixels
    for row in range(height):
        for col in range(width):
            if not foreground[row, col] or labels[row, col] != 0:
                continue

            # start new region
            next_label += 1
            labels[row, col] = next_label
            queue: list[tuple[int, int]] = [(row, col)]
            head = 0

            area = 0
            min_row = max_row = row
            min_col = max_col = col

            while head < len(queue):
                current_row, current_col = queue[head]
                head += 1

                # update region stats
                area += 1
                if current_row < min_row:
                    min_row = current_row
                if current_row > max_row:
                    max_row = current_row
                if current_col < min_col:
                    min_col = current_col
                if current_col > max_col:
                    max_col = current_col

                for d_row, d_col in neighbors:
                    n_row = current_row + d_row
                    n_col = current_col + d_col
                    if n_row < 0 or n_row >= height or n_col < 0 or n_col >= width:
                        continue
                    if not foreground[n_row, n_col]:
                        continue
                    if labels[n_row, n_col] != 0:
                        continue

                    labels[n_row, n_col] = next_label
                    queue.append((n_row, n_col))

            # save this region
            stats.append(
                {
                    "label": next_label,
                    "area": area,
                    "bbox": (min_col, min_row, max_col, max_row),
                }
            )

    return labels, stats


def largest_component(mask: np.ndarray, connectivity: int = 8):
    # keep biggest region
    labels, stats = connected_components(mask, connectivity=connectivity)
    if not stats:
        return np.zeros_like(mask, dtype=bool), labels, None, stats

    largest = max(stats, key=lambda item: int(item["area"]))
    largest_mask = labels == int(largest["label"])
    return largest_mask, labels, largest, stats
