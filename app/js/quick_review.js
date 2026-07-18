// 快速复核条与画布点选
"use strict";

function reviewQueuePieces() {
  const pieces = S.state?.pieces || [];
  const pending = pieces.filter(piece => piece.review);
  return pending.length ? pending : pieces;
}

const facingVectors = { U: [-1, 0], D: [1, 0], L: [0, -1], R: [0, 1] };

function reviewCellsAvailable(pieceId, cells) {
  const rows = Number(S.state?.rows || 0), cols = Number(S.state?.cols || 0);
  const blocked = new Set((S.state?.pieces || [])
    .filter(item => String(item.id) !== String(pieceId))
    .flatMap(item => item.cells || []).map(cell => `${cell[0]},${cell[1]}`));
  for (const cell of S.state?.hazards || []) blocked.add(`${cell[0]},${cell[1]}`);
  for (const fence of S.state?.fences || []) {
    if (fence.direction === "H" || fence.direction === "V") blocked.add(`${fence.cell[0]},${fence.cell[1]}`);
  }
  return cells.length > 0 && new Set(cells.map(cell => `${cell[0]},${cell[1]}`)).size === cells.length
    && cells.every(([row, col]) => row >= 0 && row < rows && col >= 0 && col < cols && !blocked.has(`${row},${col}`));
}

function rotatedCellsForFacing(piece, facing) {
  const cells = (piece?.cells || []).map(cell => [Number(cell[0]), Number(cell[1])]);
  if (cells.length !== 2 || !facingVectors[facing]) return cells;
  const targetAxis = facing === "U" || facing === "D" ? "V" : "H";
  const currentAxis = cells[0][1] === cells[1][1] ? "V" : cells[0][0] === cells[1][0] ? "H" : null;
  if (currentAxis === targetAxis) return cells;
  const oldVector = facingVectors[piece.facing] || (currentAxis === "V" ? facingVectors.D : facingVectors.R);
  const nextVector = facingVectors[facing];
  const ordered = [...cells].sort((a, b) => (a[0] * oldVector[0] + a[1] * oldVector[1]) - (b[0] * oldVector[0] + b[1] * oldVector[1]));
  const rump = ordered[0], head = ordered.at(-1);
  const add = (cell, vector) => [cell[0] + vector[0], cell[1] + vector[1]];
  const subtract = (cell, vector) => [cell[0] - vector[0], cell[1] - vector[1]];
  const candidates = [
    [rump, add(rump, nextVector)],
    [subtract(head, nextVector), head],
    [head, add(head, nextVector)],
    [subtract(rump, nextVector), rump],
  ];
  const valid = candidates.find(candidate => reviewCellsAvailable(piece.id, candidate));
  return (valid || candidates[0]).map(cell => [...cell]);
}

function reviewPieceWithDraft(piece, draft) {
  const cells = (draft?.cells || piece.cells || []).map(cell => [Number(cell[0]), Number(cell[1])]);
  const gridCells = cells.map(([row, col]) => (S.reviewGrid?.cells || []).find(item => item.row === row && item.col === col)).filter(Boolean);
  if (gridCells.length !== cells.length) return { ...piece, ...draft, cells };
  return {
    ...piece,
    ...draft,
    cells,
    polys: gridCells.map(cell => cell.poly),
    center: [
      gridCells.reduce((sum, cell) => sum + cell.center[0], 0) / gridCells.length,
      gridCells.reduce((sum, cell) => sum + cell.center[1], 0) / gridCells.length,
    ],
  };
}

function editorTargetPiece() {
  if (S.reviewMode !== "select" || S.editorPieceId === null) return null;
  return (S.state?.pieces || []).find(piece => String(piece.id) === String(S.editorPieceId)) || null;
}

function selectRelativeReview(offset) {
  if (S.editorSaving || S.boardConfirming) return;
  const queue = reviewQueuePieces();
  if (!queue.length) return;
  const current = queue.findIndex(piece => String(piece.id) === String(S.selectedId));
  const index = current < 0 ? (offset > 0 ? 0 : queue.length - 1) : (current + offset + queue.length) % queue.length;
  selectPiece(queue[index].id);
  $("#reviewMessage").textContent = `正在校验 ${index + 1} / ${queue.length}：棋子 #${queue[index].id}`;
}

function selectPiece(id, focus = false) {
  if (String(id) !== String(S.selectedId)) S.quickMessage = null;
  S.reviewMode = "select"; S.draftCells = [];
  S.selectedId = String(id); S.editorPieceId = String(id);
  const piece = (S.state?.pieces || []).find(item => String(item.id) === S.selectedId);
  S.editorDraft = piece ? {
    species: piece.species || "sheep", facing: piece.facing || "R",
    cells: (piece.cells || []).map(cell => [...cell]),
    awake: !!piece.awake, hitLimit: piece.hit_limit || 3,
    hitsRemaining: piece.hits_remaining || piece.hit_limit || 3,
  } : null;
  $("#selectionInfo").textContent = piece ? `已选 #${piece.id} · ${speciesNames[piece.species] || piece.species} ${facingNames[piece.facing] || ""}` : "未选中";
  renderPieces(); renderPlan(); draw(); drawReview(); renderQuickReview();
  if (focus && piece) focusPiece(piece);
}

function quickSelectedPiece() {
  return (S.state?.pieces || []).find(item => String(item.id) === String(S.selectedId));
}

function quickReviewNeedsCompletion() {
  const completionBlockers = new Set([
    "manual_review_required", "manual_learning_confirmation_required", "manual_board_unconfirmed",
  ]);
  return !!S.analysis?.manual_pending
    || (S.state?.pieces || []).some(piece => piece.review)
    || (S.analysis?.execution_blockers || []).some(item => completionBlockers.has(item.code));
}

function renderQuickReview() {
  const available = !!S.liveState && S.simulationIndex === null;
  ui.quickBar.hidden = !available;
  if (!available) return;
  const selected = quickSelectedPiece();
  if (selected && !S.quickAdding && !S.quickSaving) {
    S.quickSpecies = selected.species || "sheep";
    S.quickFacing = selected.facing || "R";
  }
  ui.quickBar.classList.toggle("adding", S.quickAdding);
  const locked = S.busy || S.quickSaving || S.boardConfirming;
  const editingPiece = !!selected || S.quickAdding;
  const needsCompletion = quickReviewNeedsCompletion();
  ui.quickStatus.className = (S.quickSaving || S.boardConfirming) ? "saving" : S.analysis?.manual_pending ? "modified" : "";
  if (S.boardConfirming) ui.quickStatus.textContent = "正在完成复核…";
  else if (S.quickSaving) ui.quickStatus.textContent = "正在保存修改…";
  else if (S.quickMessage) ui.quickStatus.textContent = S.quickMessage;
  else if (S.quickAdding) {
    const expected = S.quickSpecies === "elephant" ? 6 : 2;
    ui.quickStatus.textContent = `补${speciesNames[S.quickSpecies] || "棋子"}：点占用格 ${S.quickDraftCells.length} / ${expected}`;
  } else if (selected) {
    ui.quickStatus.textContent = `已选 #${selected.id} · ${speciesNames[selected.species] || selected.species} ${facingNames[selected.facing] || ""} · 点击类型或方向立即保存`;
  } else if (S.analysis?.manual_pending) ui.quickStatus.textContent = "修改已保存；继续修正，或完成整盘复核";
  else ui.quickStatus.textContent = "点击主棋盘中的棋子，可直接修改类型和方向";
  $("#quickPieceControls").hidden = !editingPiece;
  $("#quickPieceLabel").textContent = S.quickAdding
    ? "新增棋子"
    : selected ? `棋子 #${selected.id}` : "当前棋子";
  ui.quickSpecies.value = S.quickSpecies;
  ui.quickSpecies.disabled = locked || (!selected && !S.quickAdding);
  [...ui.quickSpecies.options].forEach(option => {
    const expected = option.value === "elephant" ? 6 : 2;
    option.disabled = !!selected && (selected.cells || []).length !== expected;
  });
  $("#quickDirections").hidden = !!selected && !S.quickAdding && (selected.cells || []).length !== 2;
  $$('[data-quick-facing]').forEach(button => {
    button.classList.toggle("selected", button.dataset.quickFacing === S.quickFacing);
    button.disabled = locked || (!S.quickAdding && (!selected || (selected.cells || []).length !== 2));
  });
  $("#quickAddPiece").textContent = S.quickAdding ? "取消补棋子" : "＋ 补棋子";
  $("#quickAddPiece").disabled = locked;
  $("#quickDeletePiece").hidden = S.quickAdding || !selected;
  $("#quickDeletePiece").disabled = locked || S.quickAdding || !selected;
  $("#quickUndo").hidden = !S.analysis?.can_undo;
  $("#quickUndo").disabled = locked || !S.analysis?.can_undo;
  $("#quickConfirmBoard").textContent = S.boardConfirming ? "正在完成…" : "完成复核";
  $("#quickConfirmBoard").hidden = !needsCompletion && !S.boardConfirming;
  $("#quickConfirmBoard").disabled = locked || (S.quickAdding && S.quickDraftCells.length > 0);
  $("#quickAdvancedReview").disabled = locked;
  ui.overlay.classList.toggle("quick-adding", S.quickAdding);
}

function focusQuickReview() {
  if (!S.liveState) { toast("请先采集并分析棋盘", true); return; }
  if (S.simulationIndex !== null) exitSimulation(false);
  renderAll();
  ui.quickBar.scrollIntoView({ block: "nearest", behavior: "smooth" });
  if (!quickSelectedPiece()) toast("快速复核已就绪：直接点击主棋盘中的棋子");
}

async function quickUpdateSelected(changes) {
  const piece = quickSelectedPiece();
  if (!piece || S.quickSaving || S.boardConfirming || S.quickAdding) return;
  const species = changes.species || piece.species || "sheep";
  const facing = changes.facing || piece.facing || "R";
  const cells = changes.facing ? rotatedCellsForFacing(piece, facing) : (piece.cells || []).map(cell => [...cell]);
  const expected = species === "elephant" ? 6 : 2;
  if (cells.length !== expected) {
    S.quickMessage = "大象与普通棋子换形请使用“高级编辑”";
    renderQuickReview();
    return;
  }
  S.quickSaving = true; S.quickMessage = null; renderQuickReview();
  try {
    const result = await call("edit_board", {
      action: "update_piece", piece_id: String(piece.id), cells,
      species, facing, awake: !!piece.awake,
      hit_limit: piece.hit_limit || 3,
      hits_remaining: piece.hits_remaining || piece.hit_limit || 3,
    });
    if (!result.ok) throw new Error(result.error);
    S.solution = null; S.simulationIndex = null;
    applyPayload(result);
    S.selectedId = String(piece.id); S.editorDraft = null;
    S.quickMessage = `已保存 #${piece.id}：${speciesNames[species] || species} ${facingNames[facing] || ""}`;
  } catch (error) {
    S.quickMessage = errorText(error); toast(S.quickMessage, true);
  } finally {
    S.quickSaving = false; renderAll();
  }
}

async function ensureQuickGrid() {
  if (S.reviewGrid?.cells?.length) return true;
  const result = await call("editor_grid");
  if (!result.ok) throw new Error(result.error);
  S.reviewGrid = result;
  applyPayload(result);
  return true;
}

async function toggleQuickAdd() {
  if (S.quickSaving || S.boardConfirming) return;
  if (S.quickAdding) {
    S.quickAdding = false; S.quickDraftCells = []; S.quickMessage = "已取消补棋子";
    renderQuickReview(); draw(); return;
  }
  try {
    S.quickSaving = true; S.quickMessage = null; renderQuickReview();
    await ensureQuickGrid();
    S.quickAdding = true; S.quickDraftCells = []; S.selectedId = null; S.editorDraft = null;
    S.quickMessage = null;
  } catch (error) {
    S.quickMessage = errorText(error); toast(S.quickMessage, true);
  } finally {
    S.quickSaving = false; renderAll();
  }
}

function mainBoardPoint(event) {
  const rect = ui.overlay.getBoundingClientRect();
  return [
    (event.clientX - rect.left) * ui.image.naturalWidth / rect.width,
    (event.clientY - rect.top) * ui.image.naturalHeight / rect.height,
  ];
}

function quickDraftCell(event) {
  const point = mainBoardPoint(event);
  return (S.reviewGrid?.cells || []).find(item => pointInPolygon(point, item.poly));
}

async function quickCreatePiece() {
  if (S.quickSaving || S.boardConfirming) return;
  const previousIds = new Set((S.state?.pieces || []).map(item => String(item.id)));
  S.quickSaving = true; renderQuickReview();
  try {
    const result = await call("edit_board", {
      action: "add_piece", cells: S.quickDraftCells,
      species: S.quickSpecies, facing: S.quickFacing,
      awake: false, hit_limit: 3, hits_remaining: 3,
    });
    if (!result.ok) throw new Error(result.error);
    S.solution = null; applyPayload(result);
    const created = (result.state?.pieces || []).find(item => !previousIds.has(String(item.id)));
    S.quickAdding = false; S.quickDraftCells = [];
    if (created) S.selectedId = String(created.id);
    S.editorDraft = null;
      S.quickMessage = created ? `已补棋子 #${created.id}；修改已保存，可继续修正或完成复核` : "棋子已添加";
  } catch (error) {
    S.quickMessage = errorText(error); toast(S.quickMessage, true);
  } finally {
    S.quickSaving = false; renderAll();
  }
}

function quickSelectDraftCell(event) {
  const cell = quickDraftCell(event);
  if (!cell || S.quickSaving || S.boardConfirming) return;
  const target = [cell.row, cell.col];
  const occupied = (S.state?.pieces || []).some(piece => (piece.cells || []).some(value => value[0] === target[0] && value[1] === target[1]));
  if (occupied) { S.quickMessage = `${cellName(target)} 已有棋子`; renderQuickReview(); return; }
  const existing = S.quickDraftCells.findIndex(value => value[0] === target[0] && value[1] === target[1]);
  if (existing >= 0) S.quickDraftCells.splice(existing, 1);
  else {
    const expected = S.quickSpecies === "elephant" ? 6 : 2;
    if (S.quickDraftCells.length >= expected) S.quickDraftCells = [];
    if (expected === 2 && S.quickDraftCells.length === 1) {
      const first = S.quickDraftCells[0];
      if (Math.abs(first[0] - target[0]) + Math.abs(first[1] - target[1]) !== 1) S.quickDraftCells = [];
    }
    S.quickDraftCells.push(target);
  }
  S.quickMessage = null; renderQuickReview(); draw();
  const expected = S.quickSpecies === "elephant" ? 6 : 2;
  if (S.quickDraftCells.length === expected) quickCreatePiece();
}

async function quickDeleteSelected() {
  const piece = quickSelectedPiece();
  if (!piece || S.quickSaving || S.boardConfirming) return;
  S.quickSaving = true; renderQuickReview();
  try {
    const result = await call("edit_board", { action: "delete_piece", piece_id: String(piece.id) });
    if (!result.ok) throw new Error(result.error);
    S.solution = null; S.selectedId = null; S.editorDraft = null; applyPayload(result);
    S.quickMessage = `已删除 #${piece.id}；可点“撤销”恢复`;
  } catch (error) { S.quickMessage = errorText(error); toast(S.quickMessage, true); }
  finally { S.quickSaving = false; renderAll(); }
}

async function quickUndo() {
  if (S.quickSaving || S.boardConfirming || !S.analysis?.can_undo) return;
  S.quickSaving = true; renderQuickReview();
  try {
    const result = await call("edit_board", { action: "undo" });
    if (!result.ok) throw new Error(result.error);
    S.solution = null; S.editorDraft = null;
    if (!(result.state?.pieces || []).some(piece => String(piece.id) === String(S.selectedId))) {
      S.selectedId = null; S.editorPieceId = null;
    }
    applyPayload(result); S.quickMessage = "已撤销上一步修改";
  } catch (error) { S.quickMessage = errorText(error); toast(S.quickMessage, true); }
  finally { S.quickSaving = false; renderAll(); }
}

async function quickConfirmBoard() {
  await completeBoardReview({ closeEditor: false });
}

function handleMainBoardPointer(event) {
  if (S.quickAdding) quickSelectDraftCell(event);
  else selectNearest(event);
}
