import re
from html.parser import HTMLParser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HTML = (ROOT / "app" / "index.html").read_text(encoding="utf-8")
SCRIPT = (ROOT / "app" / "app.js").read_text(encoding="utf-8")
CSS = (ROOT / "app" / "app.css").read_text(encoding="utf-8")
APP_PY = "\n".join(
    path.read_text(encoding="utf-8")
    for path in [ROOT / "scripts" / "app.py",
                 *sorted((ROOT / "scripts" / "gui").glob("*.py"))])


class _IdCollector(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = []

    def handle_starttag(self, _tag, attrs):
        attributes = dict(attrs)
        if attributes.get("id"):
            self.ids.append(attributes["id"])


def _html_ids():
    parser = _IdCollector()
    parser.feed(HTML)
    return parser.ids


def test_frontend_id_contract_has_no_missing_or_duplicate_elements():
    ids = _html_ids()
    assert len(ids) == len(set(ids))
    referenced = set(re.findall(r'\$\("#([A-Za-z][\w-]*)"\)', SCRIPT))
    assert not referenced.difference(ids)


def test_primary_shell_uses_compact_information_regions():
    assert 'class="app-identity"' in HTML
    assert 'class="workflow-summary"' in HTML
    assert 'class="stage-heading"' in HTML
    assert 'class="stage-meta"' in HTML
    assert 'class="safety-summary"' in HTML
    assert HTML.index('id="sandboxControls"') < HTML.index('id="viewport"')
    assert HTML.index('id="typeLegend"') > HTML.index('id="viewport"')
    assert "brand-mark" not in HTML
    assert "step-number" not in HTML


def test_operator_and_mobile_reference_modes_are_separate_products():
    ids = set(_html_ids())
    assert {
        "screenshotInput", "mobileUploadHero", "mobileUploadQuick",
        "mobileSolveButton", "mobileSheet", "mobileAnalysisPanel",
        "mobileSolutionPanel", "mobileMoveList",
    }.issubset(ids)
    assert 'data-app-mode="operator"' in HTML
    assert 'data-app-mode="reference"' in HTML
    assert "function setAppMode(mode" in SCRIPT
    assert 'window.matchMedia("(max-width: 720px)")' in SCRIPT
    assert 'body[data-app-mode="reference"] .workflow' in CSS
    assert 'body[data-app-mode="reference"] .inspector' in CSS
    assert 'async function syncHostWindowMode(mode)' in SCRIPT
    assert 'call("set_window_mode", mode)' in SCRIPT
    assert 'MIN_WINDOW_SIZE = (390, 640)' in APP_PY
    assert 'def set_window_mode(self, mode):' in APP_PY


def test_mobile_reference_uses_album_upload_without_execution_controls():
    assert "function prepareScreenshot(file)" in SCRIPT
    assert 'start("upload"' in SCRIPT
    assert "image_data: prepared.encoded" in SCRIPT
    assert 'capture_if_missing: false' in SCRIPT
    assert '移动参考模式只提供相册分析与解法参考' in SCRIPT
    assert 'ui.mobileSolve.onclick = () => start("solve"' in SCRIPT
    assert 'ui.mobileReview.onclick = () => openMobileSheet("analysis")' in SCRIPT
    assert '.mobile-action-card' in CSS
    assert '.mobile-insight-card' in CSS


def test_mobile_reference_keeps_apple_hig_layout_and_interaction_contracts():
    assert 'role="dialog" aria-modal="true"' in HTML
    assert 'role="tablist"' in HTML and HTML.count('role="tab"') == 2
    assert 'aria-labelledby="mobileSheetTitle"' in HTML
    assert "font-family: -apple-system" in CSS
    assert "env(safe-area-inset-top)" in CSS
    assert "env(safe-area-inset-bottom)" in CSS
    assert 'body[data-app-mode="reference"] .mobile-sheet-header button' in CSS
    assert 'body[data-app-mode="reference"] .mobile-layer-options label' in CSS
    assert 'min-height: 44px' in CSS
    assert "@media (prefers-contrast: more)" in CSS
    assert "@media (prefers-color-scheme: dark)" in CSS
    assert "let mobileSheetReturnFocus = null" in SCRIPT
    assert "function beginMobileSheetDrag(event)" in SCRIPT
    assert "function trapMobileSheetFocus(event)" in SCRIPT
    assert 'button.setAttribute("aria-selected", String(selected))' in SCRIPT
    assert 'mobile-has-board' in SCRIPT and 'mobile-has-insight' in SCRIPT


def test_manual_review_exposes_compact_visual_workflow():
    ids = set(_html_ids())
    assert {
        "reviewDirtyBadge", "previousReview", "reviewQueueStatus", "nextReview",
        "emptyAddPiece", "selectedSpeciesLabel", "draftSteps", "editorAutoSave",
        "confirmBoard", "saveSample", "reviewToolName", "inspectorTitle",
        "duplicatePiece", "continuousAddControl", "continueAdding",
        "cancelEditorDraft", "selectPieceMode", "addPieceMode", "wolfMode",
        "fenceMode", "clearCellMode", "toolboxHint",
    }.issubset(ids)
    assert HTML.count("完成复核") >= 2
    assert "修改即时保存" in HTML
    assert "editor-toolbox" in HTML
    assert "review-tool-dock" not in HTML
    assert "review-commandbar" in HTML


def test_manual_review_draft_changes_are_visual_and_persisted():
    assert "function reviewPreviewArrow" in SCRIPT
    assert "function rotatedCellsForFacing" in SCRIPT
    assert "function reviewPieceWithDraft" in SCRIPT
    assert "function drawReviewPieceBadge" in SCRIPT
    assert "applyPieceEdit({ silent: true })" in SCRIPT
    assert "cells: S.editorDraft.cells || piece.cells" in SCRIPT
    assert 'S.reviewMode === "select" && S.editorPieceId' in SCRIPT
    assert "function beginPieceDrag" in SCRIPT
    assert "function updatePieceDrag" in SCRIPT
    assert "function finishPieceDrag" in SCRIPT
    assert 'addEventListener("pointermove", updatePieceDrag)' in SCRIPT
    assert "applyPieceEdit({ moved: true })" in SCRIPT


def test_manual_review_supports_fast_repetitive_editing():
    assert "function editorTargetPiece()" in SCRIPT
    assert "function duplicateSelectedPiece()" in SCRIPT
    assert "function cancelEditorAction()" in SCRIPT
    assert "function rotateEditorFacing()" in SCRIPT
    assert 'const editorSpeciesOrder = ["sheep", "rocket", "bomb"' in SCRIPT
    assert 'const continueAdding = $("#continueAdding").checked' in SCRIPT
    assert 'event.key === "["' in SCRIPT and 'event.key === "]"' in SCRIPT
    assert 'event.ctrlKey && event.key === "Enter"' in SCRIPT
    assert ".editor-tool-grid" in CSS
    assert ".review-commandbar" in CSS


def test_manual_review_obstacles_are_add_only_and_clear_is_explicit():
    assert 'action: "add_hazard"' in SCRIPT
    assert 'action: "add_fence"' in SCRIPT
    assert 'action: "clear_cell"' in SCRIPT
    assert 'enterObstacleMode("clear")' in SCRIPT
    assert 'key === "e"' in SCRIPT
    assert "function drawReviewToolPreview" in SCRIPT
    assert "此工具只负责添加" in HTML
    assert "一键清除误判" in SCRIPT


def test_all_board_review_entry_points_share_one_completion_flow():
    assert "async function completeBoardReview" in SCRIPT
    assert 'completeBoardReview({ closeEditor: false })' in SCRIPT
    assert 'completeBoardReview({ closeEditor: true })' in SCRIPT
    assert "confirm_manual_board" in SCRIPT
    assert "确认并使用棋盘" not in HTML


def test_manual_review_uses_atomic_frame_and_scroll_space_geometry():
    assert "applyPayload(grid)" in SCRIPT
    assert "parent.scrollLeft" in SCRIPT
    assert "parent.scrollTop" in SCRIPT
    assert "function scheduleReviewGeometrySync" in SCRIPT
    assert "new ResizeObserver(scheduleReviewGeometrySync)" in SCRIPT


def test_bomb_plan_and_calibration_review_are_visible_in_ui():
    assert "bomb_changes" in SCRIPT
    assert "逐步核验" in SCRIPT
    assert "在主棋盘复核" in SCRIPT
    assert "reviewAfterCalibration" in SCRIPT


def test_calibration_supports_visual_zoom_and_keyboard_nudging():
    ids = set(_html_ids())
    assert {
        "calibrationPreview", "calibrationStage", "calibrationZoom",
        "calibrationZoomValue", "calibrationZoomOut", "calibrationZoomIn",
        "calibrationZoomFit", "calibrationSelection",
    }.issubset(ids)
    assert "Shift + 方向键移动 10 px" in HTML
    assert 'tabindex="0"' in HTML
    assert "function setCalibrationZoom(percent)" in SCRIPT
    assert "function nudgeCalibrationCorner(event)" in SCRIPT
    assert "event.shiftKey ? 10 : 1" in SCRIPT
    assert 'S.calibration.selectedCorner = nearest' in SCRIPT
    assert 'addEventListener("wheel"' in SCRIPT
    assert "focusSource[0] / image.naturalWidth" in SCRIPT


def test_default_post_click_wait_is_sixty_milliseconds():
    assert 'id="settleMs" type="number" min="20" max="3000" value="60"' in HTML
    assert "settle_ms: settings.settle_ms" in SCRIPT


def test_elephant_direction_arrow_is_compact_in_both_board_views():
    assert "compact = false" in SCRIPT
    assert 'piece.species === "elephant"' in SCRIPT
    assert 'species === "elephant"' in SCRIPT


def test_manual_review_elephant_arrow_uses_edge_centers_not_diagonal_corners():
    assert "function reviewArrowForCells" in SCRIPT
    assert "edgeCenter(minProjection)" in SCRIPT
    assert "edgeCenter(maxProjection)" in SCRIPT
    assert "reviewArrowForCells(piece?.cells, facing, piece?.arrow)" in SCRIPT
    assert "reviewArrowForCells(selectedCells" in SCRIPT


def test_tutorial_gesture_is_advisory_without_execution_action():
    assert "执行红框目标" not in SCRIPT
    assert "不阻止执行" in SCRIPT
    assert 'item.code !== "gesture_occlusion"' in SCRIPT


def test_low_confidence_notice_identifies_and_selects_the_exact_piece():
    assert "function reviewNoticePieces(notice)" in SCRIPT
    assert "function focusReviewPiece(info, message = null)" in SCRIPT
    assert "function handleReviewGuidance(payload)" in SCRIPT
    assert "选中并复核 #${piece.id}" in SCRIPT
    assert 'action.addEventListener("click", () => focusReviewPiece(piece))' in SCRIPT
    assert "S.selectedId = String(id)" in SCRIPT
    assert "请检查类型、方向和占格" in SCRIPT
    assert "不会把它当作整盘门禁" in SCRIPT


def test_execute_step_button_is_replaced_by_three_layer_quick_solution():
    assert 'data-action="quick"' in HTML
    assert "快速解法" in HTML
    assert "F7 快速" in HTML
    assert 'exec: () => start("quick")' in SCRIPT
    assert 'data-action="step"' not in HTML
    assert 'if (["analyze", "solve"].includes(action))' in SCRIPT
    assert 'if (["analyze", "solve", "quick"].includes(action))' not in SCRIPT
    assert "执行红框目标" not in SCRIPT


def test_solution_steps_open_reversible_sandbox_rehearsal():
    ids = set(_html_ids())
    assert {
        "sandboxControls", "sandboxPosition", "sandboxPrevious", "sandboxNext",
        "sandboxBackground", "sandboxExit",
    }.issubset(ids)
    assert "function enterSimulation(index)" in SCRIPT
    assert "S.state = data.states[nextIndex]" in SCRIPT
    assert 'item.addEventListener("click", () => enterSimulation(moveIndex))' in SCRIPT
    assert "function stepSimulation(delta)" in SCRIPT
    assert "function exitSimulation(render = true)" in SCRIPT
    assert 'event.key === "ArrowLeft"' in SCRIPT
    assert 'event.key === "ArrowRight"' in SCRIPT
    assert "选择右侧步骤开始" in SCRIPT


def test_exit_steps_are_grouped_collapsible_and_visually_deemphasized():
    assert "function groupPlanMoves(moves)" in SCRIPT
    assert 'move.result === "EXIT"' in SCRIPT
    assert 'document.createElement("details")' in SCRIPT
    assert 'wrapper.className = "move-group exit-group"' in SCRIPT
    assert ".move-list li.exit-step" in CSS
    assert ".move-list > li.move-group" in CSS


def test_solution_outcome_is_persistent_and_blocks_unsafe_execution():
    ids = set(_html_ids())
    assert {"planPanel", "planFeedback"}.issubset(ids)
    assert "function solutionFailure(solution)" in SCRIPT
    assert 'solution.result_type === "structural_conflict"' in SCRIPT
    assert "搜索超时，未找到完整解" in SCRIPT
    assert 'button.classList.toggle("plan-blocked", planBlocked)' in SCRIPT
    assert "当前没有可安全执行的完整解法" in SCRIPT


def test_solver_process_and_elastic_budget_are_visible_and_bounded():
    ids = set(_html_ids())
    assert {
        "searchProcess", "processSummary", "processTimeline",
        "elasticTimeout", "timeoutExtension", "timeoutMax",
    }.issubset(ids)
    assert 'id="timeoutMax" type="number" min="1" max="300" value="30"' in HTML
    assert "function renderSearchProcess()" in SCRIPT
    assert "timeout_extension_ms: settings.timeout_extension_s * 1000" in SCRIPT
    assert "elastic_timeout: settings.elastic_timeout" in SCRIPT
    assert 'event.event === "extension"' in SCRIPT
    assert ".process-timeline" in CSS and ".process-event.extension" in CSS


def test_runtime_settings_restore_immediately_and_keep_a_backend_copy():
    assert 'runtimeSettingsStorageKey = "sheep-solver:runtime-settings:v1"' in SCRIPT
    assert "function normalizeRuntimeSettings(data = {})" in SCRIPT
    assert "function readRuntimeSettingsCache()" in SCRIPT
    assert "function writeRuntimeSettingsCache(settings)" in SCRIPT
    assert 'await call("load_runtime_settings")' in SCRIPT
    assert 'await call("save_runtime_settings", settings)' in SCRIPT
    assert "function bindRuntimeSettings()" in SCRIPT
    assert 'addEventListener("input", () => { void persistRuntimeSettings(); })' in SCRIPT
    assert "await restoreRuntimeSettings();" in SCRIPT
    assert SCRIPT.index("await restoreRuntimeSettings();") < SCRIPT.index("bind(); setBusy(false)")


def test_new_solution_selects_the_first_plan_item_by_default():
    assert "function selectFirstSolutionStep()" in SCRIPT
    assert "S.simulationIndex = 0" in SCRIPT
    assert "S.state = data.states[0]" in SCRIPT
    assert "第 1 步 · #" in SCRIPT


def test_plan_items_use_compact_structured_fields():
    css = (ROOT / "app" / "app.css").read_text(encoding="utf-8")
    assert 'copy.className = "move-copy"' in SCRIPT
    assert 'title.className = "move-title"' in SCRIPT
    assert "phaseLabels" in SCRIPT and "resultLabels" in SCRIPT
    assert ".move-title" in css and ".move-direction" in css


def test_solution_view_fades_or_hides_the_live_screenshot():
    css = (ROOT / "app" / "app.css").read_text(encoding="utf-8")
    assert "background-dim .board-surface img" in css
    assert "background-hidden .board-surface img" in css
    assert "cycleSimulationBackground" in SCRIPT
    assert 'backgroundMode: "dim"' in SCRIPT


def test_active_sandbox_piece_uses_high_contrast_shared_color():
    css = (ROOT / "app" / "app.css").read_text(encoding="utf-8")
    assert 'simulationSelectionColor = "#e11d48"' in SCRIPT
    assert "selected && simulation ? simulationSelectionColor" in SCRIPT
    assert 'ctx.arc(cx * sx, cy * sy, 15' in SCRIPT
    assert ".move-list li.selected { background: #fff1f4; box-shadow: inset 2px 0 #e11d48" in css


def test_quick_review_is_available_in_the_main_board_view():
    ids = set(_html_ids())
    assert {
        "quickReviewBar", "quickReviewStatus", "quickSpecies", "quickDirections",
        "quickAddPiece", "quickDeletePiece", "quickUndo", "quickConfirmBoard",
        "quickAdvancedReview", "quickPieceControls", "quickPieceLabel",
        "quickActionGroup",
    }.issubset(ids)
    assert "主棋盘快速复核" in HTML
    assert "点击棋子，可直接修改类型和方向" in HTML
    assert "修改当前棋子类型" in HTML
    assert '<button id="reviewButton">快速复核</button>' in HTML
    assert "function quickUpdateSelected(changes)" in SCRIPT
    assert 'action: "update_piece"' in SCRIPT
    assert "function toggleQuickAdd()" in SCRIPT
    assert "function quickCreatePiece()" in SCRIPT
    assert 'action: "add_piece"' in SCRIPT
    assert "function quickConfirmBoard()" in SCRIPT
    assert 'ui.overlay.addEventListener("pointerdown", handleMainBoardPointer)' in SCRIPT


def test_main_board_quick_actions_only_show_when_contextually_useful():
    assert "function quickReviewNeedsCompletion()" in SCRIPT
    assert '$("#quickPieceControls").hidden = !editingPiece' in SCRIPT
    assert '$("#quickDeletePiece").hidden = S.quickAdding || !selected' in SCRIPT
    assert '$("#quickUndo").hidden = !S.analysis?.can_undo' in SCRIPT
    assert '$("#quickConfirmBoard").hidden = !needsCompletion && !S.boardConfirming' in SCRIPT
    assert "completionBlockers" in SCRIPT
    assert ".quick-piece-controls[hidden]" in CSS
