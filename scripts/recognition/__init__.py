"""Detector fusion, global cell assignment, and learning-backed board evidence.

The visual detectors deliberately remain in ``detect_occupancy``.  This
package owns the model-level decisions so they can be tested with synthetic
candidates instead of screenshots:

* ``features``           position-independent cell/pair visual features;
* ``fusion``             calibrate and fuse detector candidates, global assignment;
* ``manual_learning``    manual sample persistence and label application;
* ``direction_learning`` facing correction learning;
* ``temporal``           stabilize species/facing confidence over recent frames.
"""
from __future__ import annotations

from .features import (
    CELL_SIZE,
    PAIR_FEATURE_NAMES,
    PAIR_FEATURE_SCHEMA,
    cell_key,
    pair_visual_feature,
)
from .manual_learning import (
    MANUAL_LEARNING_SCHEMA,
    MANUAL_PRESENCE_SPECIES,
    MANUAL_LABEL_SPECIES,
    board_corrections,
    load_manual_learning,
    record_manual_learning,
    manual_candidate_proposals,
    manual_candidate_rejections,
    apply_manual_label_learning,
)
from .direction_learning import (
    direction_feature,
    record_direction_correction,
    load_direction_corrections,
    manual_direction_corrections,
    apply_direction_learning,
)
from .fusion import (
    DETECTOR_RELIABILITY,
    fuse_candidates,
    global_assignment,
)
from .temporal import (
    observation_record,
    apply_temporal,
)

from . import direction_learning, features, fusion, manual_learning, temporal

# 学习目录路径是可变的(测试会重定向到临时目录),读取必须动态代理到
# 持有模块,否则包级绑定会停留在导入时的旧值。
_DYNAMIC_PATH_ATTRS = {
    "MANUAL_LEARNING_DIR": manual_learning,
    "MANUAL_LEARNING_INDEX": manual_learning,
    "DIRECTION_LEARNING_DIR": direction_learning,
    "DIRECTION_LEARNING_INDEX": direction_learning,
}


def __getattr__(name):
    module = _DYNAMIC_PATH_ATTRS.get(name)
    if module is not None:
        return getattr(module, name)
    raise AttributeError(f"module 'recognition' has no attribute {name!r}")


__all__ = [
    "MANUAL_LEARNING_DIR",
    "MANUAL_LEARNING_INDEX",
    "DIRECTION_LEARNING_DIR",
    "DIRECTION_LEARNING_INDEX",
    "CELL_SIZE",
    "PAIR_FEATURE_NAMES",
    "PAIR_FEATURE_SCHEMA",
    "cell_key",
    "pair_visual_feature",
    "MANUAL_LEARNING_SCHEMA",
    "MANUAL_PRESENCE_SPECIES",
    "MANUAL_LABEL_SPECIES",
    "board_corrections",
    "load_manual_learning",
    "record_manual_learning",
    "manual_candidate_proposals",
    "manual_candidate_rejections",
    "apply_manual_label_learning",
    "direction_feature",
    "record_direction_correction",
    "load_direction_corrections",
    "manual_direction_corrections",
    "apply_direction_learning",
    "DETECTOR_RELIABILITY",
    "fuse_candidates",
    "global_assignment",
    "observation_record",
    "apply_temporal",
]
