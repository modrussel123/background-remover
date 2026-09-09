"""Image editing operations used by the desktop interface."""

import cv2
import numpy as np
from numpy.typing import NDArray


def get_drag_sample_points(
    rgb: NDArray[np.uint8],
    start: tuple[int, int],
    end: tuple[int, int],
    tolerance: int,
    spacing: int,
) -> list[tuple[int, int]]:
    """Sample a drag regularly and whenever its color changes."""
    height, width = rgb.shape[:2]
    x0, y0 = start
    x1, y1 = end
    distance = max(abs(x1 - x0), abs(y1 - y0))
    tolerance = max(0, min(255, int(tolerance)))
    spacing = max(1, int(spacing))

    points: list[tuple[int, int]] = []
    last_point: tuple[int, int] | None = None
    last_sample_color: NDArray[np.int16] | None = None
    last_sample_index = -spacing

    for index in range(distance + 1):
        ratio = index / distance if distance else 0
        point = (
            round(x0 + (x1 - x0) * ratio),
            round(y0 + (y1 - y0) * ratio),
        )
        if point == last_point:
            continue
        last_point = point

        px, py = point
        if px < 0 or px >= width or py < 0 or py >= height:
            continue

        color = rgb[py, px].astype(np.int16)
        color_changed = (
            last_sample_color is not None
            and int(np.max(np.abs(color - last_sample_color))) > tolerance
        )
        regularly_spaced = index - last_sample_index >= spacing
        if not points or color_changed or regularly_spaced or index == distance:
            points.append(point)
            last_sample_color = color
            last_sample_index = index

    return points


def erase_connected_color(
    alpha: NDArray[np.uint8],
    rgb: NDArray[np.uint8],
    center: tuple[int, int],
    radius: int,
    tolerance: int,
) -> int:
    """Erase the connected color region under a circular brush."""
    px, py = center
    height, width = alpha.shape
    if px < 0 or px >= width or py < 0 or py >= height:
        return 0

    radius = max(1, int(radius))
    tolerance = max(0, min(255, int(tolerance)))
    x0, y0 = max(0, px - radius), max(0, py - radius)
    x1, y1 = min(width, px + radius + 1), min(height, py + radius + 1)

    patch_rgb = rgb[y0:y1, x0:x1]
    patch_alpha = alpha[y0:y1, x0:x1]
    local_x, local_y = px - x0, py - y0
    patch_height, patch_width = patch_alpha.shape

    yy, xx = np.ogrid[:patch_height, :patch_width]
    circle = (xx - local_x) ** 2 + (yy - local_y) ** 2 <= radius ** 2

    flood_mask = np.ones((patch_height + 2, patch_width + 2), dtype=np.uint8)
    flood_mask[1:-1, 1:-1] = (~circle).astype(np.uint8)
    flood_mask[local_y + 1, local_x + 1] = 0

    cv2.floodFill(
        patch_rgb.copy(),
        flood_mask,
        (local_x, local_y),
        (0, 0, 0),
        loDiff=(tolerance, tolerance, tolerance),
        upDiff=(tolerance, tolerance, tolerance),
        flags=(
            8
            | cv2.FLOODFILL_FIXED_RANGE
            | cv2.FLOODFILL_MASK_ONLY
            | (255 << 8)
        ),
    )

    erase_mask = (flood_mask[1:-1, 1:-1] == 255) & (patch_alpha > 0)
    erased = int(np.count_nonzero(erase_mask))
    patch_alpha[erase_mask] = 0
    return erased
