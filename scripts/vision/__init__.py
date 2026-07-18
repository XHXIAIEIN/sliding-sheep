"""Visual candidate detection for the rectified game board.

Split from the original detect_occupancy module:

* ``masks``           rectification, grid constants, colour masks, gesture occlusion;
* ``species_sheep``   pink sheep / pig / goat detectors;
* ``species_special`` rocket / bomb / cattle / elephant detectors;
* ``hazards``         fence edges and wolves;
* ``segmentation``    watershed bodies, arrows, region scoring;
* ``conflicts``       cross-detector conflict resolution;
* ``pipeline``        full-frame analyze/detect and the CLI;
* ``render``          debug images;
* ``export``          solver board and layout JSON.
"""
from __future__ import annotations

from .masks import (
    CELL,
    DIRS,
    gesture_occlusion,
    make_masks,
    arrow_mask,
    _grid_from_args,
)
from .species_sheep import (
    pink_sheep_candidates,
    pig_candidates,
    goat_candidates,
)
from .species_special import (
    rocket_masks,
    classify_bomb_digit,
    bomb_markers,
    cattle_masks,
    elephant_pieces,
)
from .hazards import (
    fence_edges,
    wolf_hazards,
)
from .segmentation import (
    watershed_regions,
    _arrow_candidates,
    _gesture_target_arrow_candidates,
)
from .conflicts import (
    resolve_candidates,
    suppress_special_hazard_overlaps,
    WOLF_FORWARD_MIN_CELLS,
    WOLF_DIAGONAL_MIN_CELLS,
    resolve_goat_wolf_conflicts,
    reject_hazard_piece_overlaps,
    classify_black_sheep,
    recover_black_sheep_clusters,
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
