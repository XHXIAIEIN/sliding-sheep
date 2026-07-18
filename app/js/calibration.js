// 透视校准对话框
"use strict";

let calibrationPreviewTimer = 0, calibrationPreviewToken = 0;
const calibrationTune = { x: 0, y: 0, rotation: 0, skewX: 0, skewY: 0 };
const calibrationCornerNames = { TL: "左上", TR: "右上", BR: "右下", BL: "左下" };

function updateCalibrationSelection() {
  const host = $("#calibrationSelection");
  const key = S.calibration?.selectedCorner;
  host.className = "calibration-selection";
  if (!key || !S.calibration?.corners?.[key]) {
    host.textContent = "当前支点：未选择 · 点击圆点后可用方向键微调";
    return;
  }
  const [x, y] = S.calibration.corners[key];
  const locked = S.calibration.locked.has(key);
  host.classList.add(locked ? "locked" : "selected");
  host.textContent = `当前支点：${calibrationCornerNames[key]} ${key} · x ${x.toFixed(1)} / y ${y.toFixed(1)}${locked ? " · 已锁定" : " · 方向键 1 px / Shift 10 px"}`;
}

function resetCalibrationTuning() {
  const controls = { tuneX: "x", tuneY: "y", tuneRotation: "rotation", tuneSkewX: "skewX", tuneSkewY: "skewY" };
  for (const [id, key] of Object.entries(controls)) { $("#" + id).value = 0; calibrationTune[key] = 0; }
  $("#tuneXValue").textContent = "0"; $("#tuneYValue").textContent = "0";
  $("#tuneRotationValue").textContent = "0"; $("#tuneSkewXValue").textContent = "0"; $("#tuneSkewYValue").textContent = "0";
}

async function refreshCalibrationPreview() {
  if (!S.calibration) return;
  const token = ++calibrationPreviewToken;
  const corners = JSON.parse(JSON.stringify(S.calibration.corners));
  try {
    const result = await call("calibration_preview", corners, +$("#calibrationRows").value, +$("#calibrationCols").value);
    if (token !== calibrationPreviewToken || !result.ok) return;
    S.calibration.grid = result.grid || []; drawCalibration();
  } catch (_) { /* keep the boundary interactive while an invalid intermediate quad is adjusted */ }
}

function scheduleCalibrationPreview(delay = 24) {
  clearTimeout(calibrationPreviewTimer);
  calibrationPreviewTimer = setTimeout(refreshCalibrationPreview, delay);
  drawCalibration();
}

async function openCalibration(reset = false) {
  if (!ui.image.src) { toast("请先采集一张游戏截图", true); return; }
  try {
    const seed = await call("seed_params");
    if (!seed.ok) throw new Error(seed.error);
    S.calibration = {
      corners: JSON.parse(JSON.stringify(seed.corners)), rows: seed.rows, cols: seed.cols,
      locked: new Set(seed.locked || []), grid: [], selectedCorner: null,
    };
    S.calibrationZoom = 1;
    $("#calibrationRows").value = seed.rows; $("#calibrationCols").value = seed.cols;
    $$('[data-calibration-lock]').forEach(input => input.checked = S.calibration.locked.has(input.dataset.calibrationLock));
    $("#calibrationZoom").value = 100; $("#calibrationZoomValue").textContent = "100%";
    resetCalibrationTuning();
    updateCalibrationSelection();
    if (!ui.calibration.open) ui.calibration.showModal();
    requestAnimationFrame(() => { fitCalibration(); drawCalibration(); refreshCalibrationPreview(); });
  } catch (error) { toast(errorText(error), true); }
}

function calibrationViewportCenter() {
  const image = ui.calibrationImage, canvas = ui.calibrationCanvas, host = ui.calibrationPreview;
  if (!image.naturalWidth || !canvas.clientWidth) return [image.naturalWidth / 2, image.naturalHeight / 2];
  const rect = canvas.getBoundingClientRect(), hostRect = host.getBoundingClientRect();
  return [
    Math.max(0, Math.min(image.naturalWidth, (hostRect.left + host.clientWidth / 2 - rect.left) * image.naturalWidth / rect.width)),
    Math.max(0, Math.min(image.naturalHeight, (hostRect.top + host.clientHeight / 2 - rect.top) * image.naturalHeight / rect.height)),
  ];
}

function fitCalibration(focusSource = null) {
  const image = ui.calibrationImage, canvas = ui.calibrationCanvas;
  const host = ui.calibrationPreview, stage = ui.calibrationStage;
  if (!image.naturalWidth || !host.clientWidth || !host.clientHeight) return;
  const padding = 12;
  const fitScale = Math.min(
    Math.max(1, host.clientWidth - padding * 2) / image.naturalWidth,
    Math.max(1, host.clientHeight - padding * 2) / image.naturalHeight,
  );
  const width = Math.max(1, image.naturalWidth * fitScale * S.calibrationZoom);
  const height = Math.max(1, image.naturalHeight * fitScale * S.calibrationZoom);
  const stageWidth = Math.max(host.clientWidth, width + padding * 2);
  const stageHeight = Math.max(host.clientHeight, height + padding * 2);
  const left = (stageWidth - width) / 2, top = (stageHeight - height) / 2;
  stage.style.width = `${stageWidth}px`; stage.style.height = `${stageHeight}px`;
  image.style.left = `${left}px`; image.style.top = `${top}px`;
  image.style.width = `${width}px`; image.style.height = `${height}px`;
  canvas.style.left = `${left}px`; canvas.style.top = `${top}px`;
  canvas.style.width = `${width}px`; canvas.style.height = `${height}px`;
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.round(width * ratio));
  canvas.height = Math.max(1, Math.round(height * ratio));
  if (focusSource) {
    host.scrollLeft = left + focusSource[0] / image.naturalWidth * width - host.clientWidth / 2;
    host.scrollTop = top + focusSource[1] / image.naturalHeight * height - host.clientHeight / 2;
  }
}

function setCalibrationZoom(percent) {
  const bounded = Math.max(75, Math.min(400, Math.round(+percent / 25) * 25));
  const selected = S.calibration?.selectedCorner;
  const focus = selected ? [...S.calibration.corners[selected]] : calibrationViewportCenter();
  S.calibrationZoom = bounded / 100;
  $("#calibrationZoom").value = bounded;
  $("#calibrationZoomValue").textContent = `${bounded}%`;
  fitCalibration(focus); drawCalibration();
}

function nudgeCalibrationCorner(event) {
  if (!ui.calibration.open || !S.calibration || !event.key.startsWith("Arrow")) return false;
  const key = S.calibration.selectedCorner;
  if (!key) { updateCalibrationSelection(); return false; }
  event.preventDefault();
  if (S.calibration.locked.has(key)) { updateCalibrationSelection(); return true; }
  const step = event.shiftKey ? 10 : 1;
  const deltas = { ArrowUp: [0, -step], ArrowDown: [0, step], ArrowLeft: [-step, 0], ArrowRight: [step, 0] };
  const [dx, dy] = deltas[event.key];
  const [x, y] = S.calibration.corners[key];
  S.calibration.corners[key] = [
    Math.max(0, Math.min(ui.calibrationImage.naturalWidth, x + dx)),
    Math.max(0, Math.min(ui.calibrationImage.naturalHeight, y + dy)),
  ];
  scheduleCalibrationPreview(0); updateCalibrationSelection();
  return true;
}


function drawCalibration() {
  if (!S.calibration || !ui.calibrationCanvas.width) return;
  const canvas = ui.calibrationCanvas, ctx = canvas.getContext("2d"), ratio = window.devicePixelRatio || 1;
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0); ctx.clearRect(0, 0, canvas.width / ratio, canvas.height / ratio);
  const sx = canvas.clientWidth / ui.calibrationImage.naturalWidth, sy = canvas.clientHeight / ui.calibrationImage.naturalHeight;
  const keys = ["TL", "TR", "BR", "BL"], points = keys.map(key => S.calibration.corners[key]);
  path(ctx, points, sx, sy); ctx.fillStyle = "rgba(37,99,235,.10)"; ctx.fill(); ctx.strokeStyle = "#2563eb"; ctx.lineWidth = 2; ctx.stroke();
  ctx.strokeStyle = "rgba(37,99,235,.38)"; ctx.lineWidth = 1;
  for (const line of S.calibration.grid || []) {
    ctx.beginPath(); ctx.moveTo(line[0][0] * sx, line[0][1] * sy); ctx.lineTo(line[1][0] * sx, line[1][1] * sy); ctx.stroke();
  }
  for (let index = 0; index < points.length; index++) {
    const key = keys[index], locked = S.calibration.locked.has(key), [x, y] = points[index];
    const selected = S.calibration.selectedCorner === key;
    if (selected) {
      ctx.beginPath(); ctx.arc(x * sx, y * sy, 15, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(245,158,11,.16)"; ctx.fill();
      ctx.strokeStyle = "#d97706"; ctx.lineWidth = 3; ctx.stroke();
    }
    ctx.beginPath(); ctx.arc(x * sx, y * sy, 10, 0, Math.PI * 2); ctx.fillStyle = locked ? "#4b5563" : "#fff"; ctx.fill();
    ctx.strokeStyle = selected ? "#d97706" : "#2563eb"; ctx.lineWidth = 3; ctx.stroke();
    ctx.fillStyle = selected ? "#9a6709" : locked ? "#4b5563" : "#1d4ed8"; ctx.font = "700 10px Segoe UI"; ctx.textAlign = "center";
    ctx.fillText(`${selected ? "● " : ""}${key}${locked ? " · 锁" : ""}`, x * sx, y * sy - 16);
  }
  updateCalibrationSelection();
}

function calibrationPointer(event) {
  if (!S.calibration) return null;
  const rect = ui.calibrationCanvas.getBoundingClientRect();
  return [(event.clientX - rect.left) * ui.calibrationImage.naturalWidth / rect.width, (event.clientY - rect.top) * ui.calibrationImage.naturalHeight / rect.height];
}

function pointInPolygon(point, polygon) {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const [xi, yi] = polygon[i], [xj, yj] = polygon[j];
    if (((yi > point[1]) !== (yj > point[1])) && point[0] < (xj - xi) * (point[1] - yi) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

function pointInQuad(point, corners) { return pointInPolygon(point, [corners.TL, corners.TR, corners.BR, corners.BL]); }

function unlockedCalibrationKeys() {
  return ["TL", "TR", "BR", "BL"].filter(key => !S.calibration.locked.has(key));
}

function calibrationPivot() {
  const locked = ["TL", "TR", "BR", "BL"].filter(key => S.calibration.locked.has(key));
  const keys = locked.length ? locked : ["TL", "TR", "BR", "BL"];
  return keys.reduce((sum, key) => [sum[0] + S.calibration.corners[key][0] / keys.length, sum[1] + S.calibration.corners[key][1] / keys.length], [0, 0]);
}

function translateCalibration(dx, dy) {
  for (const key of unlockedCalibrationKeys()) S.calibration.corners[key] = [S.calibration.corners[key][0] + dx, S.calibration.corners[key][1] + dy];
  scheduleCalibrationPreview();
}

function transformCalibration(transform) {
  const pivot = calibrationPivot();
  for (const key of unlockedCalibrationKeys()) {
    const point = S.calibration.corners[key], changed = transform(point[0] - pivot[0], point[1] - pivot[1]);
    S.calibration.corners[key] = [pivot[0] + changed[0], pivot[1] + changed[1]];
  }
  scheduleCalibrationPreview();
}

function applyCalibrationTune(kind, value) {
  const delta = value - calibrationTune[kind]; calibrationTune[kind] = value;
  if (!delta || !S.calibration) return;
  if (kind === "x") translateCalibration(delta, 0);
  else if (kind === "y") translateCalibration(0, delta);
  else if (kind === "rotation") {
    const angle = delta * Math.PI / 180, cos = Math.cos(angle), sin = Math.sin(angle);
    transformCalibration((x, y) => [x * cos - y * sin, x * sin + y * cos]);
  } else if (kind === "skewX") transformCalibration((x, y) => [x + delta * y, y]);
  else if (kind === "skewY") transformCalibration((x, y) => [x, y + delta * x]);
}

async function saveCalibration() {
  try {
    const result = await call("save_params", S.calibration.corners, +$("#calibrationRows").value, +$("#calibrationCols").value, [...S.calibration.locked]);
    if (!result.ok) throw new Error(result.error);
    ui.calibration.close(); toast("校准已保存，正在重新分析");
    S.reviewAfterCalibration = true;
    if (!await start("analyze")) S.reviewAfterCalibration = false;
  } catch (error) { toast(errorText(error), true); }
}
