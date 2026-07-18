
import time
from copy import deepcopy
import numpy as np
import vision as D
from solver import DIRS, Move


class WolfOps:
    """Mixin: wolf observation, patrol zones, and risk-aware click scheduling."""

    @staticmethod
    def _wolf_motion_summary(observations, rows, cols):
        """Infer wolf patrol lanes from explicit tracks and consecutive frames."""
        recent = list(observations or [])[-8:]
        if not recent:
            return None
        track_cells = set()
        current_cells = {
            tuple(cell) for cell in (recent[-1].get("hazards") or [])
        }
        tracks = []
        center_frames = []
        for observation in recent:
            centers = []
            for component in ((observation.get("wolf") or {}).get("components") or []):
                explicit = [tuple(cell) for cell in component.get("track") or []]
                if explicit:
                    track_cells.update(explicit)
                    tracks.append({
                        "kind": component.get("kind", "runner"),
                        "axis": component.get("axis"),
                        "direction": None,
                        "cells": [list(cell) for cell in explicit],
                        "observed": True,
                    })
                center = component.get("center_rect")
                if isinstance(center, (list, tuple)) and len(center) == 2:
                    centers.append({
                        "kind": str(component.get("kind") or "wolf"),
                        "x": float(center[0]), "y": float(center[1]),
                    })
            center_frames.append(centers)

        # Associate detections frame-to-frame, not merely first-to-last.  With
        # two identical dark wolves, matching by component kind alone can swap
        # identities and manufacture a lane between different animals.
        paths = []
        active = []
        max_link = D.CELL * 2.25
        for frame_index, centers in enumerate(center_frames):
            candidates = []
            for path_index in active:
                prior = paths[path_index]["points"][-1]
                for center_index, center in enumerate(centers):
                    if paths[path_index]["kind"] != center["kind"]:
                        continue
                    distance = float(np.hypot(center["x"] - prior["x"],
                                              center["y"] - prior["y"]))
                    if distance <= max_link:
                        candidates.append((distance, path_index, center_index))
            used_paths, used_centers, next_active = set(), set(), []
            for _distance, path_index, center_index in sorted(candidates):
                if path_index in used_paths or center_index in used_centers:
                    continue
                paths[path_index]["points"].append({
                    **centers[center_index], "frame": frame_index,
                })
                used_paths.add(path_index)
                used_centers.add(center_index)
                next_active.append(path_index)
            for center_index, center in enumerate(centers):
                if center_index in used_centers:
                    continue
                paths.append({
                    "kind": center["kind"],
                    "points": [{**center, "frame": frame_index}],
                })
                next_active.append(len(paths) - 1)
            active = next_active

        min_motion = max(14.0, D.CELL * 0.35)
        max_lane_drift = D.CELL * 0.55
        for path in paths:
            points = path["points"]
            if len(points) < 3:
                continue
            xs = [item["x"] for item in points]
            ys = [item["y"] for item in points]
            x_span, y_span = max(xs) - min(xs), max(ys) - min(ys)
            dominant, cross = (x_span, y_span) if x_span >= y_span else (y_span, x_span)
            if (dominant < min_motion or cross > max_lane_drift
                    or dominant < max(1.8 * cross, min_motion)):
                continue
            axis = "H" if x_span >= y_span else "V"
            if axis == "H":
                lane = max(0, min(rows - 1, int(round(
                    float(np.median(ys)) / D.CELL - 0.5))))
                cells = [(lane, col) for col in range(cols)]
                delta = points[-1]["x"] - points[-2]["x"]
                direction = "R" if delta >= 0 else "L"
            else:
                lane = max(0, min(cols - 1, int(round(
                    float(np.median(xs)) / D.CELL - 0.5))))
                cells = [(row, lane) for row in range(rows)]
                delta = points[-1]["y"] - points[-2]["y"]
                direction = "D" if delta >= 0 else "U"
            track_cells.update(cells)
            tracks.append({
                "kind": path["kind"], "axis": axis, "direction": direction,
                "cells": [list(cell) for cell in cells],
                "center_rect": [round(float(points[-1]["x"]), 2),
                                round(float(points[-1]["y"]), 2)],
                "distance_px": round(float(dominant), 2),
                "samples": len(points), "observed": True,
            })
        present = any((item.get("wolf") or {}).get("components") for item in recent)
        return {
            "present": bool(present),
            "observed": bool(tracks),
            "sample_count": len(recent),
            "tracks": tracks,
            "track_cells": [list(cell) for cell in sorted(track_cells)],
            "current_cells": [list(cell) for cell in sorted(current_cells)],
        }

    @staticmethod
    def _wolf_patrol_zone(pieces, motion, rows, cols):
        """Flood each confirmed wolf footprint along its observed movement axis.

        Sheep cells are walls.  Flooding footprint placements instead of bare
        cells prevents a two-cell-wide wolf from leaking through one-cell gaps,
        while unioning every reachable footprint fills the empty middle of the
        patrol corridor.
        """
        if not motion:
            return set()
        current = {tuple(cell) for cell in motion.get("current_cells") or []}
        occupied = {
            tuple(cell) for piece in (pieces or []) for cell in piece.get("cells") or []
        }
        pending = set(current)
        groups = []
        while pending:
            seed = pending.pop()
            group, queue = {seed}, [seed]
            while queue:
                r, c = queue.pop()
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        neighbour = (r + dr, c + dc)
                        if neighbour in pending:
                            pending.remove(neighbour)
                            group.add(neighbour)
                            queue.append(neighbour)
            groups.append(group)

        danger = set(current)
        unused = set(range(len(groups)))
        for track in motion.get("tracks") or []:
            if not track.get("observed"):
                continue
            center = track.get("center_rect") or []
            if len(center) != 2:
                danger.update(tuple(cell) for cell in track.get("cells") or [])
                continue
            if not unused:
                # No current body component is available to size and anchor
                # the corridor. Keep the previously confirmed zone unchanged;
                # never fall back to the motion summary's full row/column.
                continue
            target = (float(center[1]) / D.CELL - 0.5,
                      float(center[0]) / D.CELL - 0.5)
            group_index = min(
                unused,
                key=lambda index: min(
                    (r - target[0]) ** 2 + (c - target[1]) ** 2
                    for r, c in groups[index]),
            )
            unused.remove(group_index)
            group = groups[group_index]
            min_r, max_r = min(r for r, _c in group), max(r for r, _c in group)
            min_c, max_c = min(c for _r, c in group), max(c for _r, c in group)
            height, width = max_r - min_r + 1, max_c - min_c + 1
            rectangle_offsets = {
                (dr, dc) for dr in range(height) for dc in range(width)
            }
            exact_offsets = {(r - min_r, c - min_c) for r, c in group}

            def footprint(anchor, offsets):
                ar, ac = anchor
                return {(ar + dr, ac + dc) for dr, dc in offsets}

            def fits(anchor, offsets):
                cells = footprint(anchor, offsets)
                return (all(0 <= r < rows and 0 <= c < cols for r, c in cells)
                        and not cells & occupied)

            offsets = rectangle_offsets
            start = (min_r, min_c)
            if not fits(start, offsets):
                offsets = exact_offsets
            axis = str(track.get("axis") or "")
            deltas = ((0, -1), (0, 1)) if axis == "H" else ((-1, 0), (1, 0))
            anchors, queue = set(), [start]
            while queue:
                anchor = queue.pop()
                if anchor in anchors or not fits(anchor, offsets):
                    continue
                anchors.add(anchor)
                queue.extend((anchor[0] + dr, anchor[1] + dc) for dr, dc in deltas)
            zone = set().union(*(footprint(anchor, offsets) for anchor in anchors)) if anchors else set(group)
            danger.update(zone)
            if axis == "H" and zone:
                lane = max(0, min(rows - 1, int(round(target[0]))))
                path_cells = [(lane, c) for c in range(min(c for _r, c in zone),
                                                       max(c for _r, c in zone) + 1)]
            elif zone:
                lane = max(0, min(cols - 1, int(round(target[1]))))
                path_cells = [(r, lane) for r in range(min(r for r, _c in zone),
                                                       max(r for r, _c in zone) + 1)]
            else:
                path_cells = []
            track["cells"] = [list(cell) for cell in path_cells]
            track["zone_cells"] = [list(cell) for cell in sorted(zone)]
        return danger

    def _remember_wolf_observation(self, debug=None):
        debug = debug or self.debug or {}
        hazards = []
        for item in debug.get("hazards") or []:
            if isinstance(item, dict):
                hazards.append([int(item["row"]), int(item["col"])])
            else:
                hazards.append([int(item[0]), int(item[1])])
        # Only surviving wolf cells count as a live observation.  Species and
        # environment rules may discard a raw dark component as goat artwork,
        # bomb smoke, or another special piece.
        wolf = deepcopy(debug.get("wolf_meta")) if hazards else None
        if wolf or self._wolf_observations:
            self._wolf_observations.append({
                "at": time.monotonic(), "wolf": wolf, "hazards": hazards,
            })
        self._wolf_motion = self._wolf_motion_summary(
            self._wolf_observations, self.rows, self.cols)
        fresh_zone = self._wolf_patrol_zone(
            self.sheep, self._wolf_motion, self.rows, self.cols)
        confirmed_now = set()
        for track in (self._wolf_motion or {}).get("tracks") or []:
            if not track.get("observed"):
                continue
            confirmed_now.update(
                tuple(cell) for cell in (track.get("zone_cells") or track.get("cells") or []))
        self._wolf_confirmed_cells.update(confirmed_now)
        current_cells = {
            tuple(cell) for cell in ((self._wolf_motion or {}).get("current_cells") or [])
        }
        # Current detections are replace-on-refresh.  Only a 3-frame-confirmed
        # patrol zone may survive a later frame; this prevents one bad wolf
        # component from becoming a permanent no-stop residue.
        self._wolf_danger_cells = set(self._wolf_confirmed_cells) | current_cells
        if self._wolf_motion is not None:
            self._wolf_motion["danger_cells"] = [
                list(cell) for cell in sorted(self._wolf_danger_cells)
            ]
            self._wolf_motion["track_cells"] = [
                list(cell) for cell in sorted(self._wolf_danger_cells)
            ]
        return self._wolf_motion

    def _wolf_track_cells(self):
        return set(self._wolf_danger_cells)

    def _wolf_guard_cells(self):
        """Current wolf footprint plus its short-horizon next positions."""
        motion = self._wolf_motion or {}
        guarded = {tuple(cell) for cell in motion.get("current_cells") or []}
        for track in motion.get("tracks") or []:
            direction = track.get("direction")
            if direction not in DIRS:
                continue
            zone = {tuple(cell) for cell in track.get("zone_cells") or []}
            active = guarded & zone
            dr, dc = DIRS[direction]
            for distance in (1, 2):
                guarded.update({
                    (r + dr * distance, c + dc * distance)
                    for r, c in active
                    if (r + dr * distance, c + dc * distance) in zone
                })
        return guarded

    @staticmethod
    def _move_wolf_risk(board, move, track_cells):
        track = {tuple(cell) for cell in (track_cells or [])}
        if not track or move.result != "EXIT":
            return {"risky": False, "overlap": [], "track_cells": len(track)}
        motion = WolfOps._move_motion(board, move)
        overlap = sorted(motion["trail"] & track)
        return {
            "risky": bool(overlap),
            "overlap": [list(cell) for cell in overlap],
            "track_cells": len(track),
        }

    @staticmethod
    def _move_motion(board, move):
        """Model the in-board cells swept by one click animation."""
        cells = set(board.pieces[str(move.piece_id)]["cells"])
        dr, dc = DIRS[move.direction]
        frontier = set(cells)
        trail = set(cells)
        max_steps = max(board.rows, board.cols) + max(2, len(cells))
        steps = move.distance if move.result != "EXIT" else max_steps
        landing = set(cells)
        travel_steps = 0
        for _ in range(max(0, int(steps))):
            frontier = {(r + dr, c + dc) for r, c in frontier}
            travel_steps += 1
            inside = {cell for cell in frontier if board.in_board(*cell)}
            trail.update(inside)
            landing = inside
            if move.result == "EXIT" and not inside:
                landing = set()
                break
        return {
            "piece": str(move.piece_id),
            "direction": move.direction,
            "result": move.result,
            "travel_steps": travel_steps,
            "start": cells,
            "trail": trail,
            "landing": landing,
        }

    @staticmethod
    def _burst_gap_schedule(motions, base_interval_ms):
        """Scale dependent gaps by the earlier sheep's travel distance."""
        base = max(20, int(round(float(base_interval_ms))))
        click_cost_ms = 50
        gaps, reasons = [], []
        for index in range(1, len(motions)):
            current = motions[index]
            gap = base
            transition_reasons = []
            for lookback in range(1, min(3, index) + 1):
                previous = motions[index - lookback]
                overlap = previous["trail"] & current["trail"]
                previous_blocks_path = previous["landing"] & current["trail"]
                if overlap or previous_blocks_path:
                    same_direction = previous.get("direction") == current.get("direction")
                    horizontal = previous.get("direction") in {"L", "R"}
                    same_corridor = bool(overlap and same_direction and (
                        ({r for r, _c in previous["start"]}
                         == {r for r, _c in current["start"]}) if horizontal else
                        ({c for _r, c in previous["start"]}
                         == {c for _r, c in current["start"]})))
                    travel_steps = max(1, int(previous.get("travel_steps") or 1))
                    if previous_blocks_path:
                        required_age = round(max(720, 240 + travel_steps * 90) * 0.40)
                        kind = "landing_on_path"
                    elif same_corridor:
                        required_age = round(max(650, 220 + travel_steps * 80) * 0.40)
                        kind = "same_corridor"
                    else:
                        required_age = round(max(420, 180 + travel_steps * 60) * 0.40)
                        kind = "path_overlap"
                    previous_index = index - lookback
                    elapsed = (sum(gaps[previous_index:index - 1])
                               + click_cost_ms * max(0, lookback - 1))
                    needed_gap = max(base, required_age - elapsed)
                    gap = max(gap, needed_gap)
                    transition_reasons.append({
                        "lookback": lookback,
                        "kind": kind,
                        "travel_steps": travel_steps,
                        "required_age_ms": required_age,
                        "elapsed_before_gap_ms": elapsed,
                        "needed_gap_ms": needed_gap,
                        "cells": [list(cell) for cell in sorted(overlap or previous_blocks_path)[:8]],
                    })
                elif lookback == 1:
                    adjacent = any(
                        abs(ar - br) + abs(ac - bc) == 1
                        for ar, ac in previous["trail"] for br, bc in current["start"])
                    if adjacent:
                        gap = max(gap, 72)
                        transition_reasons.append({
                            "lookback": 1, "kind": "adjacent_path",
                            "required_age_ms": 72, "cells": []})
            gaps.append(gap)
            reasons.append(transition_reasons)
        return gaps, reasons

    @staticmethod
    def _schedule_wait_avoiding_exits(board, planned_moves, base_interval_ms,
                                      *, previous_motion=None, limit=16,
                                      wolf_track=None, review_ids=None):
        """Move an independent direct exit ahead of a corridor-delayed exit.

        Only ordinary sheep EXIT moves are commuted.  Removing such a piece
        cannot change another piece's location; we still prove that the
        deferred move remains legal after the candidate exits.
        """
        base = max(20, int(round(float(base_interval_ms))))
        cursor = board
        remaining = list(planned_moves or [])
        ordered, reorders = [], []
        prior_motion = previous_motion
        review_ids = {str(piece_id) for piece_id in (review_ids or [])}
        while remaining and len(ordered) < max(1, int(limit)):
            legal = cursor.legal_moves()
            first = next((item for item in legal if item == remaining[0]), None)
            if first is None:
                break
            chosen_index = 0
            if str(first.piece_id) in review_ids:
                for index in range(1, min(len(remaining), 9)):
                    candidate = next((item for item in legal if item == remaining[index]), None)
                    if (candidate is None or candidate.result != "EXIT"
                            or str(candidate.piece_id) in review_ids):
                        continue
                    piece = cursor.pieces[str(candidate.piece_id)]
                    if piece.get("species", "sheep") != "sheep":
                        continue
                    after_candidate = cursor.apply(candidate)
                    if not any(item == first for item in after_candidate.legal_moves()):
                        continue
                    chosen_index = index
                    reorders.append({
                        "deferred_piece": str(first.piece_id),
                        "preferred_piece": str(candidate.piece_id),
                        "reason": "low_confidence_avoidance",
                        "avoided_wait_ms": 0,
                    })
                    break
            for index, planned in enumerate(remaining):
                if chosen_index:
                    break
                candidate = next((item for item in legal if item == planned), None)
                if (candidate is not None
                        and not (str(candidate.piece_id) in review_ids
                                 and str(first.piece_id) not in review_ids)
                        and WolfOps._move_wolf_risk(cursor, candidate, wolf_track)["risky"]):
                    chosen_index = index
                    if index:
                        reorders.append({
                            "deferred_piece": str(first.piece_id),
                            "preferred_piece": str(candidate.piece_id),
                            "reason": "wolf_track_priority",
                            "avoided_wait_ms": 0,
                        })
                    break
            first_motion = WolfOps._move_motion(cursor, first)
            first_gap = base
            if prior_motion is not None:
                schedule, _ = WolfOps._burst_gap_schedule([prior_motion, first_motion], base)
                first_gap = schedule[0] if schedule else base
            if chosen_index == 0 and prior_motion is not None and first_gap > base:
                for index in range(1, min(len(remaining), 9)):
                    candidate = next((item for item in legal if item == remaining[index]), None)
                    if (candidate is None or candidate.result != "EXIT"
                            or (str(candidate.piece_id) in review_ids
                                and str(first.piece_id) not in review_ids)):
                        continue
                    piece = cursor.pieces[str(candidate.piece_id)]
                    if piece.get("species", "sheep") != "sheep":
                        continue
                    candidate_motion = WolfOps._move_motion(cursor, candidate)
                    schedule, _ = WolfOps._burst_gap_schedule(
                        [prior_motion, candidate_motion], base)
                    if schedule and schedule[0] > base:
                        continue
                    after_candidate = cursor.apply(candidate)
                    if not any(item == first for item in after_candidate.legal_moves()):
                        continue
                    chosen_index = index
                    reorders.append({
                        "deferred_piece": str(first.piece_id),
                        "preferred_piece": str(candidate.piece_id),
                        "avoided_wait_ms": int(first_gap - base),
                    })
                    break
            chosen = remaining.pop(chosen_index)
            chosen = next(item for item in legal if item == chosen)
            prior_motion = WolfOps._move_motion(cursor, chosen)
            ordered.append(chosen)
            cursor = cursor.apply(chosen)
        return ordered + remaining, reorders

    def _guard_wolf_risk_move(self, board, move, *, max_wait_ms=2600,
                              retry_interval_ms=120):
        """Confirm a track-crossing EXIT against the wolf's latest position."""
        initial = self._move_wolf_risk(board, move, self._wolf_track_cells())
        if not initial["risky"]:
            return board, move, None
        deadline = time.monotonic() + max(0.3, float(max_wait_ms) / 1000.0)
        attempts = 0
        last_report = None
        while time.monotonic() < deadline:
            attempts += 1
            self._wait_or_cancel(max(0.04, float(retry_interval_ms) / 1000.0))
            _rectinfo, mode = self._capture_live(require_same_window=True)
            last_report = self._analyze_frame(source="app-wolf-trajectory-preflight")
            if (not last_report.get("executable")
                    and not self._batch_soft_report(last_report)):
                continue
            if self.board is None:
                continue
            fresh = self._match_planned_move(self.board, move)
            if fresh is None:
                # The wolf currently occupies the departure ray, or the live
                # board changed. Keep observing instead of clicking blindly.
                continue
            risk = self._move_wolf_risk(
                self.board, fresh, self._wolf_track_cells())
            current_risk = self._move_wolf_risk(
                self.board, fresh,
                self._wolf_guard_cells())
            if current_risk["risky"]:
                # A dynamic wolf is not a permanent Board obstacle, so test
                # its latest body cells explicitly before allowing the click.
                continue
            return self._clone_board(self.board), fresh, {
                "required": True,
                "observed": bool((self._wolf_motion or {}).get("observed")),
                "attempts": attempts,
                "capture_mode": mode,
                "risk": risk,
                "current_risk": current_risk,
                "motion": deepcopy(self._wolf_motion),
            }
        message = "狼仍在该羊的离场轨迹上，已观察动线但未出现安全点击窗口"
        blockers = list((last_report or {}).get("execution_blockers") or [])
        if blockers:
            message += "；" + "；".join(item.get("message", "") for item in blockers)
        raise RuntimeError(message)
