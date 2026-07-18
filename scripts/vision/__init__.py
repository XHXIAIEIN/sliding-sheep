"""Visual candidate detection for the rectified game board.

* ``masks``         rectification, grid constants, colour masks, gesture occlusion;
* ``detectors``     one module per piece type (arrow, pig, goat, rocket, bomb,
                    cattle, elephant, pink/black sheep, wolf, fence);
* ``segmentation``  watershed bodies and region scoring;
* ``conflicts``     cross-detector conflict resolution;
* ``pipeline``      full-frame analyze/detect and the CLI;
* ``render``        debug images;
* ``export``        solver board and layout JSON.
"""
from __future__ import annotations

from . import detectors
from .masks import (
    CELL,
    DIRS,
    gesture_occlusion,
    make_masks,
    arrow_mask,
    _grid_from_args,
)
from .detectors import (
    pink_sheep_candidates,
    pig_candidates,
    goat_candidates,
    rocket_masks,
    classify_bomb_digit,
    bomb_markers,
    cattle_masks,
    elephant_pieces,
    fence_edges,
    wolf_hazards,
    _arrow_candidates,
    _gesture_target_arrow_candidates,
    classify_black_sheep,
    recover_black_sheep_clusters,
)
from .segmentation import (
    watershed_regions,
)
from .conflicts import (
    resolve_candidates,
    suppress_special_hazard_overlaps,
    WOLF_FORWARD_MIN_CELLS,
    WOLF_DIAGONAL_MIN_CELLS,
    resolve_goat_wolf_conflicts,
    reject_hazard_piece_overlaps,
    reject_partial_exit_candidates,
    reject_departing_edge_pieces,
    apply_species_anchors,
    reject_internal_fence_overlaps,
)
from .pipeline import (
    analyze,
    detect,
    main,
)
from .render import (
    render,
    render_rect_debug,
    render_grid_labels,
    render_layout,
    render_segments,
    _remove_obsolete_images,
)
from .export import (
    to_board,
    to_layout,
)

__all__ = [
    "CELL",
    "DIRS",
    "gesture_occlusion",
    "make_masks",
    "arrow_mask",
    "_grid_from_args",
    "pink_sheep_candidates",
    "pig_candidates",
    "goat_candidates",
    "rocket_masks",
    "classify_bomb_digit",
    "bomb_markers",
    "cattle_masks",
    "elephant_pieces",
    "fence_edges",
    "wolf_hazards",
    "watershed_regions",
    "_arrow_candidates",
    "_gesture_target_arrow_candidates",
    "resolve_candidates",
    "suppress_special_hazard_overlaps",
    "WOLF_FORWARD_MIN_CELLS",
    "WOLF_DIAGONAL_MIN_CELLS",
    "resolve_goat_wolf_conflicts",
    "reject_hazard_piece_overlaps",
    "classify_black_sheep",
    "recover_black_sheep_clusters",
    "reject_partial_exit_candidates",
    "reject_departing_edge_pieces",
    "apply_species_anchors",
    "reject_internal_fence_overlaps",
    "analyze",
    "detect",
    "main",
    "render",
    "render_rect_debug",
    "render_grid_labels",
    "render_layout",
    "render_segments",
    "_remove_obsolete_images",
    "to_board",
    "to_layout",
]
