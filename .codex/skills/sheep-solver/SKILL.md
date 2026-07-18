---
name: sheep-solver
description: Run, debug, and improve the local "套住那只羊" sheep puzzle solver. Use when working in or referring to the sheep-solver repo, capturing a game screenshot, calibrating grid_params.json, detecting sheep occupancy or facing direction, inspecting visual masks/overlays, generating board.json, solving the board, or fixing recognition mistakes such as wrong sheep direction, bad perspective grid, duplicate candidates, or cell conflicts.
---

# Sheep Solver

## Core Workflow

Work in the solver repo, usually `D:\Agent\tmp\auto-clicker\sheep-solver`.

1. Check current files:
   ```powershell
   Get-ChildItem -Force
   ```

2. Capture or reuse a screenshot:
   ```powershell
   python scripts/run.py --capture
   ```
   Use `python scripts/detect_occupancy.py` directly when `images/_game.png` already exists.

3. Detect the board:
   ```powershell
   python scripts/detect_occupancy.py
   ```
   Expected outputs:
   `board_grid.json`, `board.json`, `sheep_candidates.json`, `images/_occ_axis_rect.png`, `images/_grid_labels.png`.

4. Inspect the overlay before trusting the solver:
   - `images/_occ_axis_rect.png`: rectified board overlay, best for checking cell occupancy and facing.
   - `images/_grid_labels.png`: rectified board with A/B/C column labels, 1/2/3 row labels, and sheep ids.
   - `sheep_candidates.json`: kept/dropped candidates and per-end scores.

5. Solve:
   ```powershell
   python scripts/solve_board.py board.json
   ```
   `scripts/solve_board.py` draws the click order on `images/_occ_axis_rect.png` and writes `images/_solution.png`.
   For larger boards it uses weighted A* first, then beam search, then greedy fallback.

## Recognition Rules

- Treat `grid_params.json` as the source of truth for perspective calibration. If the grid is offset, fix calibration before tuning masks.
- The detector uses a rectified `rows x cols x 64px` board via `scripts/board_grid.py`; do not duplicate perspective math in ad hoc scripts.
- Sheep are two-cell pieces. `cells[0]` is rump, `cells[1]` is head, and `facing` is `rump -> head`.
- Head/tail is primarily a shape decision: the head is the pointier end with less white body mask and lower distance-transform radius. Warm face/ear pixels are only a weak tie-breaker because horns, ears, and feet can appear near the rump.
- Prefer fixing `scripts/detect_occupancy.py` scoring or masks over hand-editing `board.json`; generated JSON should be reproducible.

## Debugging

When direction is wrong:

1. Open `images/_occ_axis_rect.png` and identify the sheep id.
2. Open `sheep_candidates.json`; match `kept[id].source_id` to the raw candidate.
3. Compare the two endpoint metrics in `raw[].metrics`: `white`, `face`, `dt_mean`, and `hist`.
4. Adjust scoring in `scripts/detect_occupancy.py` so the rule generalizes, then rerun:
   ```powershell
   python scripts/detect_occupancy.py
   python scripts/solve_board.py board.json
   ```

When occupancy conflicts appear:

- Check `候选/保留/丢弃` and `冲突` counts from `python scripts/detect_occupancy.py`.
- Inspect `sheep_candidates.json` and `images/_occ_axis_rect.png` for over-splitting, merged sheep, or wrong cell pairs.
- Keep the non-overlap resolver conservative: a no-conflict board is more useful than an over-eager board with duplicate cells.

For more detail, read `references/workflow.md`.

## Validation

Run after detector or solver edits:

```powershell
python -m compileall -q scripts
python -m pytest -q tests/test_solver.py
python scripts/detect_occupancy.py
python scripts/solve_board.py board.json
```

If `images/_game.png` is absent, run `python scripts/run.py --capture` first or ask for/provide a screenshot.
