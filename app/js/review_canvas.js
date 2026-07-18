// 复核画布几何与绘制
"use strict";

let reviewGeometryFrame = 0;

function fitReview() {
  const image = ui.reviewImage, canvas = ui.reviewCanvas;
  if (!image.naturalWidth) return;
  const parent = image.parentElement;
  const rect = image.getBoundingClientRect(), host = parent.getBoundingClientRect();
  // The canvas is positioned in the scroll container's content space. DOM
  // rectangles are viewport-relative, so restore the consumed scroll offset.
  canvas.style.left = `${rect.left - host.left + parent.scrollLeft}px`;
  canvas.style.top = `${rect.top - host.top + parent.scrollTop}px`;
  canvas.style.width = `${rect.width}px`; canvas.style.height = `${rect.height}px`;
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(rect.width * ratio); canvas.height = Math.round(rect.height * ratio);
}

function scheduleReviewGeometrySync() {
  cancelAnimationFrame(reviewGeometryFrame);
  reviewGeometryFrame = requestAnimationFrame(() => {
    fitReview(); drawReview();
    // pywebview can finish dialog/image layout one paint later at >100% DPI.
    reviewGeometryFrame = requestAnimationFrame(() => { fitReview(); drawReview(); });
  });
}

function reviewPoint(event) {
  const rect = ui.reviewCanvas.getBoundingClientRect();
  return [
    (event.clientX - rect.left) * ui.reviewImage.naturalWidth / rect.width,
    (event.clientY - rect.top) * ui.reviewImage.naturalHeight / rect.height,
  ];
}

function selectDraftCell(event) {
  const point = reviewPoint(event);
  const cell = (S.reviewGrid?.cells || []).find(item => pointInPolygon(point, item.poly));
  if (!cell) return;
  const target = [cell.row, cell.col];
  const occupied = (S.state?.pieces || []).some(piece => (piece.cells || []).some(value => value[0] === target[0] && value[1] === target[1]));
  if (occupied) { $("#reviewMessage").textContent = `${cellName(target)} 已有棋子，请选择漏识别羊所在的空格。`; return; }
  const existing = S.draftCells.findIndex(value => value[0] === target[0] && value[1] === target[1]);
  if (existing >= 0) S.draftCells.splice(existing, 1);
  else {
    const expected = $("#editorSpecies").value === "elephant" ? 6 : 2;
    if (S.draftCells.length >= expected) S.draftCells = [];
    if (S.draftCells.length === 1 && expected === 2) {
      const first = S.draftCells[0];
      if (Math.abs(first[0] - target[0]) + Math.abs(first[1] - target[1]) !== 1) {
        $("#reviewMessage").textContent = "普通棋子必须选择两个相邻格；请重新选择第二格。";
        drawReview(); return;
      }
    }
    S.draftCells.push(target);
  }
  $("#reviewMessage").textContent = `已选择 ${S.draftCells.length} 格：${S.draftCells.map(cellName).join("、")}`;
  renderPieces(); drawReview();
}

function reviewArrowForCells(cells, facing, fallback = null) {
  if (!facing || !cells?.length) return fallback;
  const vector = facingVectors[facing];
  if (!vector) return fallback;
  const centers = cells.map(value => {
    const row = Array.isArray(value) ? Number(value[0]) : Number(value?.row);
    const col = Array.isArray(value) ? Number(value[1]) : Number(value?.col);
    const cell = Array.isArray(value?.center)
      ? value
      : (S.reviewGrid?.cells || []).find(item => item.row === row && item.col === col);
    return cell ? { row, col, center: cell.center, projection: row * vector[0] + col * vector[1] } : null;
  }).filter(Boolean);
  if (centers.length < 2) return fallback;
  const minProjection = Math.min(...centers.map(item => item.projection));
  const maxProjection = Math.max(...centers.map(item => item.projection));
  if (minProjection === maxProjection) return fallback;
  const edgeCenter = projection => {
    const edge = centers.filter(item => item.projection === projection);
    return [
      edge.reduce((sum, item) => sum + item.center[0], 0) / edge.length,
      edge.reduce((sum, item) => sum + item.center[1], 0) / edge.length,
    ];
  };
  return [edgeCenter(minProjection), edgeCenter(maxProjection)];
}

function reviewPreviewArrow(piece, facing) {
  return reviewArrowForCells(piece?.cells, facing, piece?.arrow);
}

function drawReviewPieceBadge(ctx, piece, species, facing, selected, sx, sy, width) {
  const [cx, cy] = piece.center || [0, 0];
  const label = selected
    ? `#${piece.id}  ${speciesNames[species] || species} ${facingNames[facing] || ""}`
    : `#${piece.id}`;
  ctx.save();
  ctx.font = selected ? "700 11px Segoe UI" : "700 9px Segoe UI";
  const padding = selected ? 7 : 5;
  const badgeWidth = Math.ceil(ctx.measureText(label).width + padding * 2);
  const badgeHeight = selected ? 23 : 18;
  const x = Math.max(3, Math.min(width - badgeWidth - 3, cx * sx - badgeWidth / 2));
  const y = Math.max(3, cy * sy - badgeHeight / 2);
  ctx.fillStyle = selected ? "#1d4ed8" : (speciesStyles[species] || speciesStyles.sheep).color;
  ctx.fillRect(x, y, badgeWidth, badgeHeight);
  ctx.strokeStyle = "rgba(255,255,255,.96)"; ctx.lineWidth = 2; ctx.strokeRect(x, y, badgeWidth, badgeHeight);
  ctx.fillStyle = "#fff"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
  ctx.fillText(label, x + badgeWidth / 2, y + badgeHeight / 2 + .5);
  ctx.restore();
}

function reviewFencePreviewSegment(cell, direction) {
  const [tl, tr, br, bl] = cell.poly || [];
  if (!tl || !tr || !br || !bl) return null;
  const midpoint = (a, b) => [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
  return {
    U: [tl, tr], D: [bl, br], L: [tl, bl], R: [tr, br],
    H: [midpoint(tl, bl), midpoint(tr, br)],
    V: [midpoint(tl, tr), midpoint(bl, br)],
  }[direction] || null;
}

function drawReviewToolPreview(ctx, sx, sy) {
  if (!S.reviewHoverCell || !["wolf", "fence", "clear"].includes(S.reviewMode)) return;
  const cell = (S.reviewGrid?.cells || []).find(item =>
    item.row === S.reviewHoverCell[0] && item.col === S.reviewHoverCell[1]);
  if (!cell) return;
  path(ctx, cell.poly, sx, sy);
  if (S.reviewMode === "wolf") {
    ctx.fillStyle = "rgba(180,35,24,.16)"; ctx.fill();
    ctx.strokeStyle = "#b42318"; ctx.lineWidth = 3; ctx.stroke();
    return;
  }
  if (S.reviewMode === "clear") {
    ctx.fillStyle = "rgba(180,35,24,.22)"; ctx.fill();
    ctx.strokeStyle = "#b42318"; ctx.lineWidth = 3; ctx.stroke();
    const [tl, tr, br, bl] = cell.poly;
    ctx.beginPath();
    ctx.moveTo(tl[0] * sx, tl[1] * sy); ctx.lineTo(br[0] * sx, br[1] * sy);
    ctx.moveTo(tr[0] * sx, tr[1] * sy); ctx.lineTo(bl[0] * sx, bl[1] * sy);
    ctx.strokeStyle = "rgba(255,255,255,.94)"; ctx.lineWidth = 6; ctx.stroke();
    ctx.strokeStyle = "#b42318"; ctx.lineWidth = 3; ctx.stroke();
    return;
  }
  ctx.fillStyle = "rgba(138,90,43,.12)"; ctx.fill();
  const segment = reviewFencePreviewSegment(cell, S.fenceDirection);
  if (segment) {
    ctx.beginPath(); ctx.moveTo(segment[0][0] * sx, segment[0][1] * sy);
    ctx.lineTo(segment[1][0] * sx, segment[1][1] * sy);
    ctx.strokeStyle = "rgba(255,255,255,.94)"; ctx.lineWidth = 8; ctx.stroke();
    ctx.strokeStyle = "#8a5a2b"; ctx.lineWidth = 5; ctx.stroke();
  }
}

function drawReview() {
  if (!ui.review.open || !ui.reviewCanvas.width || !ui.reviewImage.naturalWidth) return;
  const canvas = ui.reviewCanvas, ctx = canvas.getContext("2d"), ratio = window.devicePixelRatio || 1;
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  const width = canvas.width / ratio, height = canvas.height / ratio;
  ctx.clearRect(0, 0, width, height);
  const sx = width / ui.reviewImage.naturalWidth, sy = height / ui.reviewImage.naturalHeight;
  ctx.strokeStyle = "rgba(32,37,34,.16)"; ctx.lineWidth = 1;
  for (const line of S.reviewGrid?.grid || []) {
    ctx.beginPath(); ctx.moveTo(line[0][0] * sx, line[0][1] * sy); ctx.lineTo(line[1][0] * sx, line[1][1] * sy); ctx.stroke();
  }
  drawBoardFeatures(ctx, S.state, sx, sy);
  for (const piece of S.state?.pieces || []) {
    const selected = S.reviewMode === "select" && String(piece.id) === String(S.editorPieceId);
    const draft = selected && S.reviewMode === "select" && S.editorDraft ? S.editorDraft : null;
    const preview = draft ? reviewPieceWithDraft(piece, draft) : piece;
    const species = preview.species || piece.species || "sheep";
    const facing = preview.facing || piece.facing;
    const style = speciesStyles[species] || speciesStyles.sheep;
    for (const polygon of preview.polys || []) {
      path(ctx, polygon, sx, sy);
      ctx.fillStyle = selected ? style.fill : piece.review ? "rgba(161,98,7,.15)" : style.fill;
      ctx.fill(); ctx.strokeStyle = selected ? "#2563eb" : piece.review ? "#a16207" : style.color;
      ctx.lineWidth = selected ? 3 : 1.25; ctx.stroke();
    }
    drawDirectionArrow(ctx, reviewPreviewArrow(preview, facing), sx, sy, selected, species === "elephant");
    drawReviewPieceBadge(ctx, preview, species, facing, selected, sx, sy, width);
  }
  if (S.reviewMode === "add") {
    const selectedCells = (S.reviewGrid?.cells || []).filter(cell => S.draftCells.some(value => value[0] === cell.row && value[1] === cell.col));
    for (const cell of selectedCells) {
      path(ctx, cell.poly, sx, sy); ctx.fillStyle = "rgba(35,122,75,.24)"; ctx.fill();
      ctx.strokeStyle = "#237a4b"; ctx.lineWidth = 3; ctx.stroke();
    }
    if (selectedCells.length >= 2) {
      const species = $("#editorSpecies").value;
      const arrow = reviewArrowForCells(selectedCells, $("#editorFacing").value);
      drawDirectionArrow(ctx, arrow, sx, sy, true, species === "elephant");
    }
  }
  drawReviewToolPreview(ctx, sx, sy);
}
