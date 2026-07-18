// 人工复核编辑器与棋盘确认
"use strict";

function renderPieces() {
  const selected = editorTargetPiece();
  const adding = S.reviewMode === "add";
  const obstacle = ["wolf", "fence", "clear"].includes(S.reviewMode);
  if (!adding && selected && !S.editorDraft) {
    S.editorDraft = { species: selected.species || "sheep", facing: selected.facing || "R", cells: (selected.cells || []).map(cell => [...cell]), awake: !!selected.awake, hitLimit: selected.hit_limit || 3, hitsRemaining: selected.hits_remaining || selected.hit_limit || 3 };
  }
  const draft = S.editorDraft || { species: "sheep", facing: "R", awake: false, hitLimit: 3, hitsRemaining: 3 };
  const modeButtons = {
    select: $("#selectPieceMode"), add: $("#addPieceMode"),
    wolf: $("#wolfMode"), fence: $("#fenceMode"), clear: $("#clearCellMode"),
  };
  for (const [mode, button] of Object.entries(modeButtons)) {
    const active = S.reviewMode === mode;
    button.classList.toggle("selected", active);
    button.setAttribute("aria-pressed", String(active));
    button.disabled = S.editorSaving || S.boardConfirming;
  }
  $(".review-preview").dataset.mode = S.reviewMode;
  const modeCopy = editorModeCopy[S.reviewMode] || editorModeCopy.select;
  $(".canvas-hint").textContent = modeCopy.canvas;
  $("#reviewToolName").textContent = modeCopy.name;
  $("#reviewToolHint").textContent = modeCopy.hint;
  $("#inspectorTitle").textContent = selected ? `棋子 #${selected.id}` : modeCopy.inspector;
  $("#inspectorModePill").textContent = modeCopy.pill;
  $("#toolboxHint").textContent = modeCopy.hint;
  $("#cancelEditorDraft").hidden = S.reviewMode === "select";
  $("#cancelEditorDraft").textContent = adding ? "取消补棋子" : "返回选择";
  $("#obstacleForm").hidden = !obstacle; $("#fenceDirections").hidden = S.reviewMode !== "fence";
  $("#obstacleForm").dataset.mode = obstacle ? S.reviewMode : "";
  if (obstacle) {
    const obstacleCopy = {
      wolf: { title: "添加狼危险格", icon: "◆", help: "点击狼所在的空格即可添加标记。再次点击不会删除，避免修正时误操作。", action: "此工具只负责添加；误标请切换“清除”。" },
      fence: { title: "选择栅栏位置", icon: "╫", help: "格内横/竖栏可放在棋盘内部；出口边只可放在对应的棋盘边缘。", action: "选好位置后点击棋盘格添加；误标请切换“清除”。" },
      clear: { title: "一键清除误判", icon: "⌫", help: "点击一个格子，会一起清除占用它的棋子、狼格和以它为锚点的栅栏。", action: "清除后可按 Ctrl Z 撤销，不需要先选中对象。" },
    }[S.reviewMode];
    $("#obstacleTitle").textContent = obstacleCopy.title;
    $("#obstacleIcon").textContent = obstacleCopy.icon;
    $("#obstacleHelp").textContent = obstacleCopy.help;
    $("#obstacleActionHint").textContent = obstacleCopy.action;
  }
  $("#editorEmpty").hidden = adding || obstacle || !!selected; $("#editorForm").hidden = obstacle || (!adding && !selected);
  $("#selectedCells").textContent = "—";
  const expected = draft.species === "elephant" ? 6 : 2;
  if (adding) {
    $("#selectedPieceId").textContent = "新增棋子";
    $("#selectedPieceState").textContent = "位置草稿";
    $("#selectedPieceState").className = "";
    $("#selectedSpeciesLabel").textContent = speciesNames[draft.species] || draft.species;
    $("#selectedCells").textContent = S.draftCells.length ? S.draftCells.map(cellName).join("、") : `请选择 ${expected} 格`;
    $("#applyPieceEdit").textContent = "添加棋子";
    $("#applyPieceEdit").disabled = S.draftCells.length !== expected || S.editorSaving || S.boardConfirming;
    $("#applyPieceEdit").hidden = false;
    $("#editorAutoSave").hidden = true;
    $("#deletePiece").hidden = true;
    $("#duplicatePiece").hidden = true;
    $("#continuousAddControl").hidden = false;
    $("#draftSteps").hidden = false;
    const steps = $$("#draftSteps span");
    steps.forEach(step => step.className = "");
    steps[0].className = "done";
    steps[1].className = S.draftCells.length === expected ? "done" : "active";
    steps[2].className = S.draftCells.length === expected ? "active" : "";
  } else if (selected) {
    $("#selectedPieceId").textContent = `棋子 #${selected.id}`;
    $("#selectedCells").textContent = (draft.cells || selected.cells).map(cellName).join("–");
    $("#selectedSpeciesLabel").textContent = `${speciesNames[draft.species] || draft.species} ${facingNames[draft.facing] || ""}`;
    $("#selectedPieceState").textContent = S.editorSaving ? "保存中" : selected.review ? "建议复核" : "已识别";
    $("#selectedPieceState").className = S.editorSaving ? "saving" : "";
    $("#applyPieceEdit").hidden = true;
    $("#editorAutoSave").hidden = false;
    $("#deletePiece").hidden = false;
    $("#duplicatePiece").hidden = false;
    $("#continuousAddControl").hidden = true;
    $("#duplicatePiece").disabled = S.editorSaving || S.boardConfirming;
    $("#deletePiece").disabled = S.editorSaving || S.boardConfirming;
    $("#draftSteps").hidden = true;
  }
  $("#editorSpecies").value = draft.species; $("#editorFacing").value = draft.facing;
  $("#editorAwake").checked = !!draft.awake; $("#editorHitLimit").value = draft.hitLimit || 3; $("#editorHitsRemaining").value = draft.hitsRemaining || draft.hitLimit || 3;
  $$(".species-palette button").forEach(button => {
    const style = speciesStyles[button.dataset.species] || speciesStyles.sheep;
    button.style.setProperty("--species-color", style.color);
    button.classList.toggle("selected", button.dataset.species === draft.species);
    button.disabled = S.editorSaving || S.boardConfirming;
  });
  $$(".direction-pad button").forEach(button => {
    button.classList.toggle("selected", button.dataset.facing === draft.facing);
    button.disabled = S.editorSaving || S.boardConfirming;
  });
  $("#pigFields").hidden = draft.species !== "pig"; $("#bombFields").hidden = draft.species !== "bomb";
  $$('[data-fence-direction]').forEach(button => button.disabled = S.editorSaving || S.boardConfirming);
  $("#editorUndo").disabled = S.editorSaving || S.boardConfirming || !S.analysis?.can_undo; $("#editorRedo").disabled = S.editorSaving || S.boardConfirming || !S.analysis?.can_redo;
  $("#editorReset").disabled = S.editorSaving || S.boardConfirming;
  $("#confirmBoard span").textContent = S.boardConfirming ? "正在完成复核…" : "完成复核";
  $("#confirmBoard").disabled = S.editorSaving || S.boardConfirming || (adding && S.draftCells.length > 0);
  $("#cancelEditorDraft").disabled = S.editorSaving || S.boardConfirming;
  const queue = reviewQueuePieces();
  const queueIndex = queue.findIndex(piece => String(piece.id) === String(S.selectedId));
  const hasPending = (S.state?.pieces || []).some(piece => piece.review);
  $("#reviewQueueStatus").textContent = queue.length ? `${queueIndex < 0 ? "—" : queueIndex + 1} / ${queue.length}${hasPending ? " 待核" : ""}` : "0 / 0";
  $("#previousReview").disabled = S.editorSaving || S.boardConfirming || !queue.length;
  $("#nextReview").disabled = S.editorSaving || S.boardConfirming || !queue.length;
  const dirty = !!S.analysis?.manual_pending || (adding && S.draftCells.length > 0);
  $("#reviewDirtyBadge").textContent = dirty ? (S.analysis?.manual_pending ? "待完成复核" : "新增草稿") : "可直接完成复核";
  $("#reviewDirtyBadge").classList.toggle("modified", dirty);
  renderReviewMeta();
}

function renderReviewMeta() {
  const pieces = S.state?.pieces || [];
  const wolfKeys = new Set([...(S.state?.hazards || []), ...(S.state?.dynamic_hazards || [])].map(cell => cell.join(",")));
  $("#reviewBoardMeta").textContent = `${S.state?.rows || "?"}×${S.state?.cols || "?"} · ${pieces.length} 棋子 · ${wolfKeys.size} 狼格 · ${S.state?.fences?.length || 0} 栅栏${S.analysis?.manual_pending ? " · 修正已保存，待完成复核" : ""}`;
  const legend = $("#reviewLegend"); legend.replaceChildren();
  const present = new Set(pieces.map(piece => piece.species || "sheep"));
  for (const species of Object.keys(speciesStyles)) {
    if (!present.has(species)) continue;
    const item = document.createElement("span"); item.style.setProperty("--legend-color", speciesStyles[species].color); item.textContent = speciesNames[species]; legend.append(item);
  }
  if (wolfKeys.size) { const item = document.createElement("span"); item.style.setProperty("--legend-color", "#b42318"); item.textContent = "狼"; legend.append(item); }
  if (S.state?.fences?.length) { const item = document.createElement("span"); item.style.setProperty("--legend-color", "#8a5a2b"); item.textContent = "栅栏"; legend.append(item); }
}

function cellName([row, col]) { return `${col < 26 ? String.fromCharCode(65 + Number(col)) : Number(col) + 1}${Number(row) + 1}`; }

async function applyPieceEdit(options = {}) {
  const piece = editorTargetPiece();
  if (S.editorSaving || S.boardConfirming || (S.reviewMode !== "add" && !piece)) return;
  const silent = options?.silent === true;
  const targetPieceId = piece ? String(piece.id) : null;
  S.editorSaving = true;
  renderPieces();
  try {
    const adding = S.reviewMode === "add";
    const previousIds = new Set((S.state?.pieces || []).map(item => String(item.id)));
    const result = await call("edit_board", adding ? {
      action: "add_piece", cells: S.draftCells,
      species: S.editorDraft.species, facing: S.editorDraft.facing,
      awake: $("#editorAwake").checked,
      hit_limit: +$("#editorHitLimit").value || 3,
      hits_remaining: +$("#editorHitsRemaining").value || 3,
    } : {
      action: "update_piece", piece_id: String(piece.id), cells: S.editorDraft.cells || piece.cells,
      species: S.editorDraft.species, facing: S.editorDraft.facing,
      awake: $("#editorAwake").checked,
      hit_limit: +$("#editorHitLimit").value || 3,
      hits_remaining: +$("#editorHitsRemaining").value || 3,
    });
    if (!result.ok) throw new Error(result.error);
    applyPayload(result);
    if (adding) {
      const created = (result.state?.pieces || []).find(item => !previousIds.has(String(item.id)));
      const continueAdding = $("#continueAdding").checked;
      S.draftCells = [];
      if (continueAdding) {
        S.reviewMode = "add"; S.selectedId = null; S.editorPieceId = null;
        $("#reviewMessage").textContent = `已新增${speciesNames[S.editorDraft.species] || "棋子"}；继续点击下一只的占用格。`;
      } else {
        S.reviewMode = "select";
        if (created) selectPiece(created.id);
        $("#reviewMessage").textContent = "棋子已添加并保存；可继续修正，完成后点“完成复核”。";
      }
    } else {
      const updated = (result.state?.pieces || []).find(item => String(item.id) === targetPieceId);
      S.editorDraft = null; selectPiece(targetPieceId);
      $("#reviewMessage").textContent = options?.moved
        ? `棋子 #${targetPieceId} 已移动到 ${(updated?.cells || []).map(cellName).join("–")}`
        : silent
          ? `棋子 #${targetPieceId} 已保存：${speciesNames[updated?.species] || "已更新"} ${facingNames[updated?.facing] || ""}`
          : "已原位替换所选棋子；完成整盘检查后点“完成复核”。";
    }
  } catch (error) {
    if (targetPieceId) selectPiece(targetPieceId);
    $("#reviewMessage").textContent = errorText(error);
    toast(errorText(error), true);
  } finally {
    S.editorSaving = false;
    renderPieces();
    drawReview();
  }
}

function commitSelectedDraft() {
  renderPieces();
  drawReview();
  if (S.reviewMode === "select" && S.editorPieceId) applyPieceEdit({ silent: true });
}

function enterAddMode(seed = null) {
  if (S.editorSaving || S.boardConfirming) return;
  const source = seed && seed.id !== undefined ? seed : null;
  S.reviewMode = "add"; S.selectedId = null; S.editorPieceId = null; S.draftCells = []; S.reviewHoverCell = null;
  S.editorDraft = {
    species: source?.species || "sheep", facing: source?.facing || "R",
    awake: !!source?.awake, hitLimit: source?.hit_limit || 3,
    hitsRemaining: source?.hits_remaining || source?.hit_limit || 3,
  };
  $("#reviewMessage").textContent = source
    ? `已复制棋子 #${source.id} 的属性；点击新棋子的占用格。`
    : "补棋子：先选类型和方向，再点它占用的格子。";
  renderPieces(); renderPlan(); draw(); drawReview();
}

function enterSelectMode() {
  if (S.editorSaving || S.boardConfirming) return;
  S.reviewMode = "select"; S.draftCells = [];
  S.editorDraft = null;
  if (!S.selectedId && reviewQueuePieces().length) selectPiece(reviewQueuePieces()[0].id);
  else if (S.selectedId) S.editorPieceId = String(S.selectedId);
  S.reviewHoverCell = null;
  $("#reviewMessage").textContent = "直接点棋子；类型、方向和拖动修改都会即时保存。";
  renderPieces(); drawReview();
}

function enterObstacleMode(mode) {
  if (S.editorSaving || S.boardConfirming) return;
  S.reviewMode = mode; S.selectedId = null; S.editorPieceId = null; S.draftCells = []; S.editorDraft = null; S.reviewHoverCell = null;
  $("#reviewMessage").textContent = mode === "wolf"
    ? "添加狼格：点击空格；需要删除时切换“清除”。"
    : mode === "fence"
      ? "添加栅栏：先选位置，再点击棋盘格；需要删除时切换“清除”。"
      : "清除误判：直接点击含有错误标注的格子。";
  renderPieces(); renderPlan(); draw(); drawReview();
}

function duplicateSelectedPiece() {
  const piece = editorTargetPiece();
  if (piece) enterAddMode(piece);
}

function cancelEditorAction() {
  if (S.editorSaving || S.reviewMode === "select") return;
  const discarded = S.reviewMode === "add" && S.draftCells.length;
  enterSelectMode();
  $("#reviewMessage").textContent = discarded ? "已丢弃未完成的新增草稿。" : "已返回选择工具。";
}

function chooseEditorSpecies(species) {
  if (S.editorSaving || S.boardConfirming || !S.editorDraft || !editorSpeciesOrder.includes(species)) return;
  S.editorDraft.species = species;
  const expected = species === "elephant" ? 6 : 2;
  if (S.reviewMode === "add") S.draftCells = S.draftCells.slice(0, expected);
  commitSelectedDraft();
}

function chooseEditorFacing(facing) {
  if (S.editorSaving || S.boardConfirming || !S.editorDraft || !facingVectors[facing]) return;
  const piece = editorTargetPiece();
  if (piece) S.editorDraft.cells = rotatedCellsForFacing(piece, facing);
  S.editorDraft.facing = facing;
  commitSelectedDraft();
}

function rotateEditorFacing() {
  if (!S.editorDraft) return;
  const clockwise = { U: "R", R: "D", D: "L", L: "U" };
  chooseEditorFacing(clockwise[S.editorDraft.facing] || "R");
}

async function applyBoardToolAtCell(event) {
  if (S.editorSaving || S.boardConfirming) return;
  const point = reviewPoint(event);
  const cell = (S.reviewGrid?.cells || []).find(item => pointInPolygon(point, item.poly));
  if (!cell) return;
  if (S.reviewMode === "fence") {
    const rows = Number(S.state?.rows || S.reviewGrid?.rows || 0);
    const cols = Number(S.state?.cols || S.reviewGrid?.cols || 0);
    const onRequiredEdge = {
      U: cell.row === 0, D: cell.row === rows - 1,
      L: cell.col === 0, R: cell.col === cols - 1,
      H: true, V: true,
    }[S.fenceDirection];
    if (!onRequiredEdge) {
      const edgeName = { U: "最上行", D: "最下行", L: "最左列", R: "最右列" }[S.fenceDirection];
      $("#reviewMessage").textContent = `${facingNames[S.fenceDirection]} 边栏只能添加在棋盘${edgeName}；请点击对应边缘格。`;
      return;
    }
  }
  S.editorSaving = true;
  renderPieces();
  try {
    const target = [cell.row, cell.col];
    const command = S.reviewMode === "wolf"
      ? { action: "add_hazard", cell: target }
      : S.reviewMode === "fence"
        ? { action: "add_fence", cell: target, direction: S.fenceDirection }
        : { action: "clear_cell", cell: target };
    const result = await call("edit_board", command);
    if (!result.ok) throw new Error(result.error);
    applyPayload(result); drawReview();
    if (S.reviewMode === "clear") {
      const detail = result.edit_detail || {};
      const removed = [];
      if (detail.removed_piece_ids?.length) removed.push(`棋子 #${detail.removed_piece_ids.join("、#")}`);
      if (detail.removed_hazard) removed.push("狼格");
      if (detail.removed_fence_directions?.length) removed.push(`${detail.removed_fence_directions.length} 个栅栏`);
      $("#reviewMessage").textContent = removed.length
        ? `${cellName(target)} 已清除：${removed.join("、")}；可按 Ctrl Z 撤销。`
        : `${cellName(target)} 没有可清除的标注。`;
    } else if (!result.changed) {
      $("#reviewMessage").textContent = S.reviewMode === "wolf"
        ? `${cellName(target)} 已经是狼格；删除请用“清除”。`
        : `${cellName(target)} 已有同方向栅栏；删除请用“清除”。`;
    } else {
      $("#reviewMessage").textContent = S.reviewMode === "wolf"
        ? `${cellName(target)} 已添加狼格。`
        : `${cellName(target)} 已添加栅栏。`;
    }
  } catch (error) {
    $("#reviewMessage").textContent = errorText(error);
  } finally {
    S.editorSaving = false;
    renderPieces();
    drawReview();
  }
}

function reviewCellAt(point) {
  return (S.reviewGrid?.cells || []).find(item => pointInPolygon(point, item.poly));
}

function updateReviewHover(event) {
  if (S.reviewDrag) return;
  const cell = reviewCellAt(reviewPoint(event));
  const next = cell ? [cell.row, cell.col] : null;
  const current = S.reviewHoverCell;
  if ((current === null && next === null)
      || (current && next && current[0] === next[0] && current[1] === next[1])) return;
  S.reviewHoverCell = next;
  drawReview();
}

function clearReviewHover() {
  if (!S.reviewHoverCell) return;
  S.reviewHoverCell = null;
  drawReview();
}

function reviewPieceAt(point) {
  return [...(S.state?.pieces || [])].reverse().find(piece =>
    (piece.polys || []).some(polygon => pointInPolygon(point, polygon)));
}

function beginPieceDrag(event) {
  const point = reviewPoint(event);
  const piece = reviewPieceAt(point);
  const cell = reviewCellAt(point);
  if (!piece || !cell) return;
  if (String(piece.id) !== String(S.editorPieceId)) selectPiece(piece.id);
  S.reviewDrag = {
    pointerId: event.pointerId,
    pieceId: String(piece.id),
    startCell: [cell.row, cell.col],
    originalCells: (piece.cells || []).map(value => [...value]),
    moved: false,
    valid: true,
  };
  ui.reviewCanvas.setPointerCapture?.(event.pointerId);
  ui.reviewCanvas.classList.add("dragging");
  $("#reviewMessage").textContent = `拖动棋子 #${piece.id}，松手后保存到新格子。`;
  event.preventDefault();
}

function updatePieceDrag(event) {
  const drag = S.reviewDrag;
  if (!drag || drag.pointerId !== event.pointerId || !S.editorDraft) return;
  const cell = reviewCellAt(reviewPoint(event));
  if (!cell) {
    drag.valid = false;
    $("#reviewMessage").textContent = "目标位置超出棋盘；松手会恢复原位置。";
    return;
  }
  const rowDelta = cell.row - drag.startCell[0], colDelta = cell.col - drag.startCell[1];
  const translated = drag.originalCells.map(([row, col]) => [row + rowDelta, col + colDelta]);
  if (!reviewCellsAvailable(drag.pieceId, translated)) {
    drag.valid = false;
    $("#reviewMessage").textContent = "目标格被占用或不可落点；松手会恢复原位置。";
    return;
  }
  drag.valid = true;
  drag.moved = translated.some((value, index) => value[0] !== drag.originalCells[index][0] || value[1] !== drag.originalCells[index][1]);
  S.editorDraft.cells = translated;
  $("#reviewMessage").textContent = drag.moved
    ? `将棋子 #${drag.pieceId} 移到 ${translated.map(cellName).join("–")}`
    : `棋子 #${drag.pieceId} 保持原位置`;
  renderPieces(); drawReview();
  event.preventDefault();
}

function finishPieceDrag(event, cancelled = false) {
  const drag = S.reviewDrag;
  if (!drag || drag.pointerId !== event.pointerId) return;
  try { ui.reviewCanvas.releasePointerCapture?.(event.pointerId); } catch (_error) { /* already released */ }
  ui.reviewCanvas.classList.remove("dragging");
  S.reviewDrag = null;
  if (cancelled || !drag.valid) {
    selectPiece(drag.pieceId);
    $("#reviewMessage").textContent = cancelled ? "已取消拖动。" : "该位置不可用，棋子已恢复原位。";
    return;
  }
  if (drag.moved) applyPieceEdit({ moved: true });
}

function handleReviewPointer(event) {
  if (S.editorSaving || S.boardConfirming) return;
  if (S.reviewMode === "add") selectDraftCell(event);
  else if (["wolf", "fence", "clear"].includes(S.reviewMode)) applyBoardToolAtCell(event);
  else beginPieceDrag(event);
}

async function deleteSelectedPiece() {
  const piece = editorTargetPiece();
  if (!piece || S.editorSaving || S.boardConfirming) return;
  try {
    const result = await call("edit_board", { action: "delete_piece", piece_id: String(piece.id) });
    if (!result.ok) throw new Error(result.error);
    S.selectedId = null; S.editorPieceId = null; S.editorDraft = null; applyPayload(result); drawReview();
    $("#reviewMessage").textContent = `已删除棋子 #${piece.id}；可使用撤销恢复。`;
  } catch (error) { $("#reviewMessage").textContent = errorText(error); }
}

async function editHistory(action) {
  if (S.editorSaving || S.boardConfirming) return;
  try {
    const result = await call("edit_board", { action });
    if (!result.ok) throw new Error(result.error);
    S.editorDraft = null;
    if (!(result.state?.pieces || []).some(piece => String(piece.id) === String(S.selectedId))) {
      S.selectedId = null; S.editorPieceId = null;
    } else if (S.selectedId) S.editorPieceId = String(S.selectedId);
    applyPayload(result); drawReview();
    $("#reviewMessage").textContent = action === "undo" ? "已撤销上一步修改。" : action === "redo" ? "已重做修改。" : "已恢复本次识别结果。";
  } catch (error) { $("#reviewMessage").textContent = errorText(error); }
}

async function completeBoardReview({ closeEditor = false } = {}) {
  if (S.editorSaving || S.quickSaving || S.boardConfirming) return false;
  if (!ui.review.open && S.reviewMode === "add") {
    S.reviewMode = "select"; S.draftCells = []; S.editorDraft = null;
  }
  if ((S.reviewMode === "add" && S.draftCells.length) || (S.quickAdding && S.quickDraftCells.length)) {
    const message = "还有未完成的补棋子草稿；请先添加完成或取消草稿。";
    S.quickMessage = message;
    $("#reviewMessage").textContent = message;
    renderAll();
    return false;
  }
  S.boardConfirming = true;
  $("#reviewMessage").textContent = "正在保存整盘复核结果…";
  renderAll();
  try {
    const result = await call("confirm_manual_board");
    if (!result.ok) throw new Error(result.error);
    applyPayload(result);
    S.quickAdding = false; S.quickDraftCells = [];
    const message = result.executable
      ? "复核已完成，可以开始连续执行"
      : "复核已完成；仍有安全项需要处理";
    S.quickMessage = message;
    $("#reviewMessage").textContent = message;
    if (closeEditor && ui.review.open) ui.review.close();
    toast(message);
    return true;
  } catch (error) {
    const message = errorText(error);
    S.quickMessage = message;
    $("#reviewMessage").textContent = message;
    toast(message, true);
    return false;
  } finally {
    S.boardConfirming = false;
    renderAll();
  }
}

async function confirmBoard() {
  await completeBoardReview({ closeEditor: true });
}

async function saveSample() {
  try { const result = await call("save_manual_sample", $("#sampleNote").value); if (!result.ok) throw new Error(result.error); $("#reviewMessage").textContent = `样本已保存：${result.path}`; }
  catch (error) { $("#reviewMessage").textContent = errorText(error); }
}
