"""GUI API for the pywebview desktop app, split by responsibility.

``app.Api`` is assembled from the mixins below; every mixin operates on the
shared ``Api`` instance state created in ``Api.__init__``:

* ``common``      shared constants, runtime settings, error plumbing;
* ``geometry``    grid-to-pixel helpers;
* ``window``      window targeting, mode switching, global hotkeys;
* ``settings``    runtime settings persistence, hard refresh;
* ``analysis``    capture and frame analysis;
* ``editor``      manual board review and learning samples;
* ``board_state`` board serialization and detection artifacts;
* ``solving``     budgeted solve and solution payloads;
* ``workflow``    single-intent background jobs;
* ``execution``   verified clicks, retries, refresh cycles;
* ``wolf``        wolf observation and risk-aware scheduling;
* ``calibration`` grid calibration and seeding.
"""
from . import common
from .common import ExecutionReviewRequired, _safe_error, _wrap
from .geometry import GridGeometryOps
from .window import WindowOps
from .settings import SettingsOps
from .analysis import AnalysisOps
from .editor import EditorOps
from .board_state import BoardStateOps
from .solving import SolveOps
from .workflow import WorkflowOps
from .execution import ExecutionOps
from .wolf import WolfOps
from .calibration import CalibrationOps

__all__ = [
    "ExecutionReviewRequired",
    "common",
    "_safe_error",
    "_wrap",
    "GridGeometryOps",
    "WindowOps",
    "SettingsOps",
    "AnalysisOps",
    "EditorOps",
    "BoardStateOps",
    "SolveOps",
    "WorkflowOps",
    "ExecutionOps",
    "WolfOps",
    "CalibrationOps",
]
