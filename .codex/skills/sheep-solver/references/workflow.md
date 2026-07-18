# Sheep Solver Workflow Reference

## Files To Keep In The Repo Root

Keep source directories and hand-maintained configuration in the repo:

- `app/index.html`, `app/grid_tuner.html`
- `scripts/app.py`, `scripts/board_grid.py`, `scripts/detect_occupancy.py`, `scripts/board_io.py`
- `scripts/solver.py`, `scripts/solver_search.py`, `scripts/solve_board.py`, `scripts/run.py`, `scripts/capture_window.py`
- `tests/test_solver.py`, `README.md`
- `grid_params.json`
- `reference/`

Generated files may be moved to `.trash` when cleaning:

- `images/_game.png`
- `board.json`
- `board_grid.json`
- `sheep_candidates.json`
- `images/_occ_axis_rect.png`
- `images/_grid_labels.png`
- `images/_solution.png`
- temporary crop/contact/annotation images

## Direct Command Recipes

Use current screenshot:

```powershell
python scripts/detect_occupancy.py
python scripts/solve_board.py board.json
```

Capture then solve:

```powershell
python scripts/run.py --capture
```

Compile and smoke test:

```powershell
python -m compileall -q scripts
python -m pytest -q tests/test_solver.py
```

Stop the GUI if direct scripts are preferred:

```powershell
Get-CimInstance Win32_Process |
  Where-Object { $_.CommandLine -match 'app\.py' } |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
```

## Detector Structure

`scripts/board_grid.py` owns perspective mapping:

- Load `grid_params.json`.
- Scale corners when screenshot size differs from calibration size.
- Warp the source image into rectified board pixels.
- Export `board_grid.json` for cell center and polygon audit data.

`scripts/detect_occupancy.py` owns recognition:

- Create body mask from low saturation and high value.
- Create weak face mask from warm and dark pixels.
- Use distance-transform peaks as watershed seeds.
- Score each segmented region against adjacent two-cell candidates.
- Resolve duplicates and overlaps.
- Emit a solver-compatible `board.json`.

`scripts/solver_search.py` owns large-board search:

- Try weighted A* first.
- Try beam search if weighted A* does not solve within budget.
- Compare against greedy and keep the best result.
- Rank states by remaining sheep, exit blockers, stuck pieces, direct exits, and deadlock pairs.

`scripts/solve_board.py` owns solving:

- Load `board.json`.
- Use A* for small boards and `solver_search.search_solve()` for larger boards.
- Draw click order on `images/_occ_axis_rect.png` and emit `images/_solution.png`.

## Direction Debugging Checklist

1. Inspect `images/_occ_axis_rect.png`; it has rectified cells and sheep ids.
2. Look up `kept[id]` in `sheep_candidates.json`.
3. Use `source_id` to find the raw candidate with metrics.
4. For each endpoint, compare:
   - `white`: body-mask pixels in that cell.
   - `dt_mean`: how round/fat the endpoint is.
   - `face`: weak color evidence.
   - `hist`: region contribution to that cell.
5. Favor the endpoint with less `white` and lower `dt_mean` as the head. Use `face` only as a tie-breaker.
6. Rerun detection and inspect direction counts plus overlay.

## Calibration Debugging Checklist

If many sheep occupy wrong cells or the grid visibly drifts:

1. Open `app/grid_tuner.html` or use the GUI calibration flow.
2. Adjust the four corners, not per-sheep offsets.
3. Save `grid_params.json`.
4. Rerun `python scripts/detect_occupancy.py`.
5. Verify `images/_occ_axis_rect.png` before solving.

## Cleanup Policy

Move old scripts and generated artifacts to a dated folder under `.trash`, for example:

```powershell
$dest = Join-Path (Resolve-Path .) '.trash\YYYY-MM-DD-cleanup'
New-Item -ItemType Directory -Force -Path $dest | Out-Null
Move-Item -LiteralPath '.\old_file.py' -Destination $dest
```

Before moving many files, list exact names. Avoid broad destructive globs.
