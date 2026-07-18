// 棋盘画布、沙盘模拟与棋子选择
"use strict";

function simulationStates() {
  const states = S.solution?.states;
  const moves = S.solution?.moves;
  return Array.isArray(states) && Array.isArray(moves) && states.length === moves.length + 1
    ? { states, moves } : null;
}

function selectFirstSolutionStep() {
  const data = simulationStates();
  if (!data?.moves?.length) {
    S.simulationIndex = null;
    S.selectedId = null;
    S.state = S.liveState || data?.states?.[0] || null;
    return;
  }
  S.simulationIndex = 0;
  S.state = data.states[0];
  S.selectedId = String(data.moves[0].piece);
  S.editorDraft = null;
  $("#selectionInfo").textContent = `第 1 步 · #${data.moves[0].piece} · ${data.moves[0].desc || "准备执行"}`;
}

function enterSimulation(index) {
  if (S.busy) return;
  const data = simulationStates();
  if (!data) return;
  const nextIndex = Math.max(0, Math.min(data.moves.length, Number(index) || 0));
  S.simulationIndex = nextIndex;
  S.state = data.states[nextIndex];
  const move = data.moves[nextIndex];
  S.selectedId = move ? String(move.piece) : null;
  S.editorDraft = null;
  renderAll();
  if (move) {
    $("#selectionInfo").textContent = `第 ${nextIndex + 1} 步 · #${move.piece} · ${move.desc || "推演"}`;
  } else {
    $("#selectionInfo").textContent = `沙盘完成 · ${data.moves.length} 步`;
  }
}

function stepSimulation(delta) {
  if (S.simulationIndex === null) return;
  enterSimulation(S.simulationIndex + delta);
}

function exitSimulation(render = true) {
  if (S.simulationIndex === null) return;
  S.simulationIndex = null;
  S.state = S.liveState || simulationStates()?.states?.[0] || null;
  S.selectedId = null;
  S.editorDraft = null;
  $("#selectionInfo").textContent = "实时识别棋盘";
  if (render) renderAll();
}

function cycleSimulationBackground() {
  if (!simulationStates()) return;
  S.backgroundMode = ({ dim: "hidden", hidden: "full", full: "dim" })[S.backgroundMode] || "dim";
  renderSimulation();
  draw();
}

function renderSimulation() {
  const data = simulationStates();
  const active = !!data && S.simulationIndex !== null;
  ui.sandbox.hidden = !data;
  ui.viewport.classList.toggle("solution-ready", !!data);
  ui.viewport.classList.toggle("sandbox-mode", active);
  for (const mode of ["dim", "hidden", "full"]) {
    ui.viewport.classList.toggle(`background-${mode}`, !!data && S.backgroundMode === mode);
  }
  if (!data) return;
  const final = active && S.simulationIndex === data.moves.length;
  ui.sandboxPosition.textContent = !active
    ? "选择右侧步骤开始"
    : final ? `${data.moves.length} / ${data.moves.length} 步`
      : `${S.simulationIndex + 1} / ${data.moves.length} 步`;
  ui.sandboxPrevious.disabled = !active || S.simulationIndex <= 0;
  ui.sandboxNext.disabled = !active || final;
  ui.sandboxNext.textContent = final ? "已完成" : "下一步";
  ui.sandboxExit.disabled = !active;
  ui.sandboxBackground.textContent = `底图${{ dim: "淡化", hidden: "隐藏", full: "显示" }[S.backgroundMode]}`;
  if (active) $("#boardTitle").textContent = final ? "沙盘完成" : `沙盘 ${S.simulationIndex + 1} / ${data.moves.length}`;
}

function fitOverlay() {
  if (!ui.image.complete || !ui.image.naturalWidth) return;
  const rect = ui.image.getBoundingClientRect();
  const host = ui.surface.getBoundingClientRect();
  ui.overlay.style.left = `${rect.left - host.left}px`;
  ui.overlay.style.top = `${rect.top - host.top}px`;
  ui.overlay.style.width = `${rect.width}px`;
  ui.overlay.style.height = `${rect.height}px`;
  const ratio = window.devicePixelRatio || 1;
  ui.overlay.width = Math.round(rect.width * ratio);
  ui.overlay.height = Math.round(rect.height * ratio);
}

function path(ctx, points, sx, sy) {
  if (!points?.length) return;
  ctx.beginPath(); ctx.moveTo(points[0][0] * sx, points[0][1] * sy);
  for (const point of points.slice(1)) ctx.lineTo(point[0] * sx, point[1] * sy);
  ctx.closePath();
}

function drawDirectionArrow(ctx, arrow, sx, sy, selected, compact = false) {
  if (!arrow?.[0] || !arrow?.[1]) return;
  let x1 = arrow[0][0] * sx, y1 = arrow[0][1] * sy;
  let x2 = arrow[1][0] * sx, y2 = arrow[1][1] * sy;
  if (compact) {
    const startX = x1, startY = y1, dx = x2 - x1, dy = y2 - y1;
    x1 = startX + dx * 0.46; y1 = startY + dy * 0.46;
    x2 = startX + dx * 0.88; y2 = startY + dy * 0.88;
  }
  const angle = Math.atan2(y2 - y1, x2 - x1);
  const head = selected ? 10 : 8;
  const stroke = selected && S.simulationIndex !== null ? simulationSelectionColor : selected ? "#1d4ed8" : "#202522";
  const trace = () => {
    ctx.beginPath();
    ctx.moveTo(x1, y1); ctx.lineTo(x2, y2);
    ctx.moveTo(x2, y2); ctx.lineTo(x2 - head * Math.cos(angle - Math.PI / 6), y2 - head * Math.sin(angle - Math.PI / 6));
    ctx.moveTo(x2, y2); ctx.lineTo(x2 - head * Math.cos(angle + Math.PI / 6), y2 - head * Math.sin(angle + Math.PI / 6));
    ctx.stroke();
  };
  ctx.lineCap = "round"; ctx.lineJoin = "round";
  ctx.strokeStyle = "rgba(255,255,255,.92)"; ctx.lineWidth = selected ? 6 : 5; trace();
  ctx.strokeStyle = stroke; ctx.lineWidth = selected ? 3 : 2.25; trace();
}

function drawBoardFeatures(ctx, state, sx, sy) {
  const hazards = new Map();
  for (const item of state?.hazard_polys || []) hazards.set(item.cell.join(","), { ...item, dynamic: false });
  for (const item of state?.dynamic_hazard_polys || []) hazards.set(item.cell.join(","), { ...item, dynamic: true });
  for (const item of hazards.values()) {
    path(ctx, item.poly, sx, sy);
    ctx.fillStyle = item.dynamic ? "rgba(180,35,24,.24)" : "rgba(180,35,24,.16)";
    ctx.fill(); ctx.strokeStyle = "#b42318"; ctx.lineWidth = 2; ctx.stroke();
    const center = item.poly.reduce((sum, point) => [sum[0] + point[0] / item.poly.length, sum[1] + point[1] / item.poly.length], [0, 0]);
    ctx.fillStyle = "#b42318"; ctx.font = "800 11px Segoe UI"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillText("狼", center[0] * sx, center[1] * sy);
  }
  for (const fence of state?.fences || []) {
    const [a, b] = fence.segment || [];
    if (!a || !b) continue;
    ctx.beginPath(); ctx.moveTo(a[0] * sx, a[1] * sy); ctx.lineTo(b[0] * sx, b[1] * sy);
    ctx.strokeStyle = "rgba(255,255,255,.9)"; ctx.lineWidth = 7; ctx.lineCap = "round"; ctx.stroke();
    ctx.strokeStyle = "#8a5a2b"; ctx.lineWidth = 4; ctx.stroke();
  }
}

function draw() {
  if (!ui.overlay.width || !ui.image.naturalWidth) return;
  const ctx = ui.overlay.getContext("2d");
  const ratio = window.devicePixelRatio || 1;
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  const width = ui.overlay.width / ratio, height = ui.overlay.height / ratio;
  ctx.clearRect(0, 0, width, height);
  const simulation = S.simulationIndex !== null;
  if (simulation && S.backgroundMode !== "full") {
    ctx.fillStyle = S.backgroundMode === "hidden" ? "rgb(248,250,248)" : "rgba(248,250,248,.28)";
    ctx.fillRect(0, 0, width, height);
  }
  const sx = width / ui.image.naturalWidth, sy = height / ui.image.naturalHeight;
  const data = S.analysis || {};
  if ($("#showGrid").checked) {
    ctx.strokeStyle = simulation ? "rgba(32,37,34,.30)" : "rgba(32,37,34,.16)"; ctx.lineWidth = 1;
    for (const line of data.grid || []) {
      ctx.beginPath(); ctx.moveTo(line[0][0] * sx, line[0][1] * sy); ctx.lineTo(line[1][0] * sx, line[1][1] * sy); ctx.stroke();
    }
  }
  const showPieces = $("#showPieces").checked;
  const showDirections = $("#showDirections").checked;
  if (!showPieces && !showDirections) return;
  if (showPieces) drawBoardFeatures(ctx, S.state, sx, sy);
  if (S.quickAdding && S.reviewGrid?.cells?.length) {
    for (const cellKey of S.quickDraftCells) {
      const cell = S.reviewGrid.cells.find(item => item.row === cellKey[0] && item.col === cellKey[1]);
      if (!cell) continue;
      path(ctx, cell.poly, sx, sy);
      ctx.fillStyle = "rgba(217,119,6,.28)"; ctx.fill();
      ctx.strokeStyle = "#a16207"; ctx.lineWidth = 3; ctx.stroke();
    }
  }
  for (const piece of S.state?.pieces || []) {
    const selected = String(piece.id) === String(S.selectedId);
    const review = piece.review;
    const style = speciesStyles[piece.species] || speciesStyles.sheep;
    if (showPieces) {
      for (const polygon of piece.polys || []) {
        path(ctx, polygon, sx, sy);
        ctx.fillStyle = selected && simulation ? "rgba(225,29,72,.30)" : selected ? "rgba(37,99,235,.26)" : review ? "rgba(183,121,31,.16)" : simulation ? `${style.color}2f` : style.fill;
        ctx.fill(); ctx.strokeStyle = selected && simulation ? simulationSelectionColor : selected ? "#2563eb" : review ? "#b7791f" : style.color;
        ctx.lineWidth = selected && simulation ? 4 : selected ? 3 : simulation ? 2 : 1.5; ctx.stroke();
      }
    }
    if (showDirections) drawDirectionArrow(ctx, piece.arrow, sx, sy, selected, piece.species === "elephant");
    if (showPieces) {
      const [cx, cy] = piece.center || [0, 0];
      if (selected && simulation) {
        ctx.fillStyle = "rgba(255,255,255,.96)"; ctx.beginPath(); ctx.arc(cx * sx, cy * sy, 15, 0, Math.PI * 2); ctx.fill();
      }
      ctx.fillStyle = selected && simulation ? simulationSelectionColor : selected ? "#1d4ed8" : style.color;
      ctx.beginPath(); ctx.arc(cx * sx, cy * sy, selected && simulation ? 12 : 11, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = "#fff"; ctx.font = "700 9px Segoe UI"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
      ctx.fillText(piece.species && piece.species !== "sheep" ? style.mark : String(piece.id), cx * sx, cy * sy);
    }
  }
}

function selectNearest(event) {
  selectNearestOn(event, ui.overlay, ui.image);
}

function selectNearestOn(event, canvas, image) {
  if (!S.state?.pieces?.length || !image.naturalWidth) return;
  const rect = canvas.getBoundingClientRect();
  const x = (event.clientX - rect.left) * image.naturalWidth / rect.width;
  const y = (event.clientY - rect.top) * image.naturalHeight / rect.height;
  let best = null, distance = Infinity;
  for (const piece of S.state.pieces) {
    const d = Math.hypot(piece.center[0] - x, piece.center[1] - y);
    if (d < distance) { best = piece; distance = d; }
  }
  if (best && distance < 90) selectPiece(best.id);
}

function focusPiece(piece) {
  if (!piece?.center || !ui.image.naturalWidth) return;
  const imageRect = ui.image.getBoundingClientRect();
  const surfaceRect = ui.surface.getBoundingClientRect();
  const x = imageRect.left - surfaceRect.left + piece.center[0] * imageRect.width / ui.image.naturalWidth;
  const y = imageRect.top - surfaceRect.top + piece.center[1] * imageRect.height / ui.image.naturalHeight;
  ui.viewport.scrollTo({
    left: Math.max(0, x - ui.viewport.clientWidth / 2),
    top: Math.max(0, y - ui.viewport.clientHeight / 2),
    behavior: "smooth",
  });
}
