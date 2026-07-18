"""Core runtime domain shared by GUI and CLI.

Submodules are imported on demand (``from core import safety``); ``capture``
in particular sets process DPI awareness at import time, so nothing here is
imported eagerly.

* ``runtime``   OperationCoordinator: single background job, cancel token;
* ``safety``    scene classification and execution blockers;
* ``analysis``  one recognition pass converged into an AnalysisBundle;
* ``capture``   Win32 window capture.
"""
