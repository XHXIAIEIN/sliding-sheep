"""Per-species visual detectors, one module per piece type.

Tuning a single species touches exactly one file:

* ``arrow``        facing arrows on two-cell sheep;
* ``pink_sheep``   pink bow evidence;
* ``pig``          pig body and sleeping state;
* ``goat``         goat horns vs. wolf bodies;
* ``rocket``       rocket sheep;
* ``bomb``         bomb sheep and countdown digits;
* ``cattle``       cattle body/face/cell candidates;
* ``elephant``     2x3 elephants and trunk facing;
* ``black_sheep``  black sheep recovery out of wolf blobs;
* ``wolf``         wolf hazards;
* ``fence``        fence segments.
"""
from . import (arrow, black_sheep, bomb, cattle, elephant, fence, goat, pig,
               pink_sheep, rocket, wolf)
from .arrow import _arrow_candidates, _gesture_target_arrow_candidates
from .black_sheep import classify_black_sheep, recover_black_sheep_clusters
from .bomb import bomb_markers, classify_bomb_digit
from .cattle import _cattle_candidates, cattle_masks
from .elephant import elephant_pieces
from .fence import fence_edges
from .goat import goat_candidates
from .pig import pig_candidates
from .pink_sheep import pink_sheep_candidates
from .rocket import _rocket_candidates, rocket_masks
from .wolf import wolf_hazards

__all__ = [
    "arrow", "black_sheep", "bomb", "cattle", "elephant", "fence", "goat",
    "pig", "pink_sheep", "rocket", "wolf",
    "_arrow_candidates", "_gesture_target_arrow_candidates",
    "classify_black_sheep", "recover_black_sheep_clusters",
    "bomb_markers", "classify_bomb_digit",
    "_cattle_candidates", "cattle_masks",
    "elephant_pieces", "fence_edges", "goat_candidates", "pig_candidates",
    "pink_sheep_candidates", "_rocket_candidates", "rocket_masks",
    "wolf_hazards",
]
