"""Bomb sheep detector: bomb body plus countdown-digit classification."""
from __future__ import annotations

import cv2
import numpy as np
from ..masks import _cell_of, _exclude


def classify_bomb_digit(component_mask: np.ndarray):
    """Recognize the blue 1/2/3 glyph inside a bomb counter disc."""
    if not isinstance(component_mask, np.ndarray) or component_mask.size == 0:
        return None, 0.0, {}
    ys, xs = np.where(component_mask > 0)
    if len(xs) < 60:
        return None, 0.0, {}
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    glyph = (component_mask[y0:y1, x0:x1] > 0).astype(np.uint8)
    height, width = glyph.shape
    ratio = width / max(1.0, float(height))
    if not (8 <= width <= 25 and 18 <= height <= 34 and 0.32 <= ratio <= 0.92):
        return None, 0.0, {"width": width, "height": height, "ratio": round(ratio, 3)}
    normalized = cv2.resize(glyph, (24, 32), interpolation=cv2.INTER_NEAREST) > 0
    lower = normalized[16:24]
    middle_density = float(lower[:, 8:16].mean())
    right_density = float(lower[:, 16:24].mean())
    if ratio <= 0.56:
        digit = 1
        confidence = min(0.99, 0.72 + max(0.0, 0.56 - ratio) * 2.0)
    elif right_density > middle_density:
        digit = 3
        separation = (right_density - middle_density) / max(0.15, right_density + middle_density)
        confidence = min(0.99, 0.68 + 0.28 * separation)
    else:
        digit = 2
        separation = (middle_density - right_density) / max(0.15, right_density + middle_density)
        confidence = min(0.99, 0.68 + 0.28 * separation)
    return digit, round(float(confidence), 4), {
        "width": width, "height": height, "ratio": round(ratio, 3),
        "lower_middle": round(middle_density, 4),
        "lower_right": round(right_density, 4),
    }


def bomb_markers(rect: np.ndarray, rows: int, cols: int, exclusion_mask=None):
    """Find blue count discs mounted on red dynamite bundles."""
    hsv = cv2.cvtColor(rect, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    blue = ((hue >= 102) & (hue <= 135) & (sat >= 90) & (val >= 90)).astype(np.uint8) * 255
    red = (((hue <= 7) | (hue >= 172)) & (sat >= 90) & (val >= 60)).astype(np.uint8) * 255
    blue, red = _exclude(blue, exclusion_mask), _exclude(red, exclusion_mask)
    count, labels, stats, centers = cv2.connectedComponentsWithStats(blue, 8)
    components, marker_mask = [], np.zeros(blue.shape, np.uint8)
    for label_id in range(1, count):
        x, y, width, height, area = (int(v) for v in stats[label_id])
        # Active bombs add flame/smoke around the counter.  Perspective and
        # bloom can widen the blue disc beyond the old 45 px ceiling or split
        # it into a narrow component; red dynamite support remains the strong
        # discriminator against cyan sheep artwork.
        if not (35 <= area <= 1600 and 7 <= width <= 60 and 7 <= height <= 60):
            continue
        if not 0.40 <= width / float(height) <= 2.20:
            continue
        pad = 22
        y0, y1 = max(0, y - pad), min(red.shape[0], y + height + pad)
        x0, x1 = max(0, x - pad), min(red.shape[1], x + width + pad)
        red_support = int((red[y0:y1, x0:x1] > 0).sum())
        if red_support < 70:
            continue
        cx, cy = (float(v) for v in centers[label_id])
        cell = _cell_of(int(cx), int(cy), rows, cols)
        if cell is None:
            continue
        component_mask = (labels[y:y + height, x:x + width] == label_id).astype(np.uint8) * 255
        digit, digit_confidence, digit_features = classify_bomb_digit(component_mask)
        components.append({
            "cell": cell, "box": [x, y, width, height], "area": area,
            "red_support": red_support, "digit": digit,
            "digit_confidence": digit_confidence, "digit_features": digit_features,
        })
        marker_mask[labels == label_id] = 255
    grouped = {}
    for component in components:
        grouped.setdefault(tuple(component["cell"]), []).append(component)
    markers = []
    for cell, items in sorted(grouped.items()):
        base = max(items, key=lambda item: (item["red_support"], item["area"]))
        digit_items = [item for item in items
                       if item.get("digit") in {1, 2, 3}
                       and float(item.get("digit_confidence") or 0.0) >= 0.68]
        best_digit = max(digit_items, key=lambda item: item["digit_confidence"], default=None)
        # Unknown counters are deliberately treated as one remaining hit.  It
        # is safer to forbid a collision than to invent spare bomb capacity.
        hits_remaining = int(best_digit["digit"]) if best_digit else 1
        markers.append({
            "cell": cell,
            "hits_remaining": hits_remaining,
            "hit_limit": 3,
            "counter_confident": bool(best_digit),
            "counter_confidence": (float(best_digit["digit_confidence"])
                                   if best_digit else 0.0),
            "counter_unknown": best_digit is None,
            "counter_box": best_digit.get("box") if best_digit else None,
            "counter_features": best_digit.get("digit_features") if best_digit else None,
            "box": base["box"],
            "red_support": base["red_support"],
            "components": items,
        })
    return markers, marker_mask
