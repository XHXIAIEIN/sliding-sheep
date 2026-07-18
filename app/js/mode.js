// 桌面/移动双模式切换与移动底部卡片
"use strict";

function preferredAppMode() {
  try {
    const saved = localStorage.getItem(appModeStorageKey);
    if (["operator", "reference"].includes(saved)) return saved;
  } catch (_error) { /* viewport default remains available */ }
  return window.matchMedia("(max-width: 720px)").matches ? "reference" : "operator";
}

function setAppMode(mode, { persist = true } = {}) {
  const next = mode === "reference" ? "reference" : "operator";
  S.appMode = next;
  document.body.dataset.appMode = next;
  $$('[data-app-mode]').forEach(button => {
    const selected = button.dataset.appMode === next;
    button.setAttribute("aria-pressed", String(selected));
  });
  if (persist) {
    try { localStorage.setItem(appModeStorageKey, next); } catch (_error) { /* optional */ }
  }
  if (next !== "reference") closeMobileSheet();
  renderMobile();
  requestAnimationFrame(() => { fitOverlay(); draw(); });
}

async function syncHostWindowMode(mode) {
  try {
    const result = await call("set_window_mode", mode);
    if (!result?.ok) throw new Error(result?.error || "窗口尺寸同步失败");
    return result;
  } catch (error) {
    toast(`窗口尺寸同步失败：${errorText(error)}`, true);
    return null;
  }
}

function openMobileSheet(tab = S.mobileTab) {
  if (S.appMode !== "reference") return;
  if (!ui.mobileSheet.classList.contains("open")) mobileSheetReturnFocus = document.activeElement;
  setMobileTab(tab);
  ui.mobileSheet.classList.add("open");
  ui.mobileSheet.setAttribute("aria-hidden", "false");
  ui.mobileSheetScrim.classList.add("visible");
  ui.mobileSheetPeek.setAttribute("aria-expanded", "true");
  requestAnimationFrame(() => ui.mobileSheetClose.focus({ preventScroll: true }));
}

function closeMobileSheet() {
  const wasOpen = ui.mobileSheet.classList.contains("open");
  ui.mobileSheet.classList.remove("open");
  ui.mobileSheet.classList.remove("dragging");
  ui.mobileSheet.style.removeProperty("--sheet-drag-y");
  ui.mobileSheet.setAttribute("aria-hidden", "true");
  ui.mobileSheetScrim.classList.remove("visible");
  ui.mobileSheetPeek.setAttribute("aria-expanded", "false");
  S.mobileSheetDrag = null;
  if (wasOpen && mobileSheetReturnFocus?.isConnected) {
    const target = mobileSheetReturnFocus;
    mobileSheetReturnFocus = null;
    requestAnimationFrame(() => target.focus({ preventScroll: true }));
  }
}

function setMobileTab(tab) {
  S.mobileTab = tab === "solution" ? "solution" : "analysis";
  $$('[data-mobile-tab]').forEach(button => {
    const selected = button.dataset.mobileTab === S.mobileTab;
    button.classList.toggle("selected", selected);
    button.setAttribute("aria-pressed", String(selected));
    button.setAttribute("aria-selected", String(selected));
    button.tabIndex = selected ? 0 : -1;
  });
  $("#mobileAnalysisPanel").hidden = S.mobileTab !== "analysis";
  $("#mobileSolutionPanel").hidden = S.mobileTab !== "solution";
  ui.mobileSheetTitle.textContent = S.mobileTab === "solution" ? "解法参考" : "分析结果";
}

function beginMobileSheetDrag(event) {
  if (!ui.mobileSheet.classList.contains("open") || event.target.closest("button")) return;
  if (event.pointerType === "mouse" && event.button !== 0) return;
  S.mobileSheetDrag = { pointerId: event.pointerId, startY: event.clientY, startAt: performance.now() };
  ui.mobileSheet.classList.add("dragging");
  event.currentTarget.setPointerCapture?.(event.pointerId);
}

function moveMobileSheetDrag(event) {
  const drag = S.mobileSheetDrag;
  if (!drag || drag.pointerId !== event.pointerId) return;
  const distance = Math.max(0, event.clientY - drag.startY);
  ui.mobileSheet.style.setProperty("--sheet-drag-y", `${distance}px`);
}

function endMobileSheetDrag(event) {
  const drag = S.mobileSheetDrag;
  if (!drag || drag.pointerId !== event.pointerId) return;
  const distance = Math.max(0, event.clientY - drag.startY);
  const velocity = distance / Math.max(1, performance.now() - drag.startAt);
  event.currentTarget.releasePointerCapture?.(event.pointerId);
  S.mobileSheetDrag = null;
  if (distance >= 84 || velocity >= .65) {
    closeMobileSheet();
    return;
  }
  ui.mobileSheet.classList.remove("dragging");
  ui.mobileSheet.style.removeProperty("--sheet-drag-y");
}

function trapMobileSheetFocus(event) {
  if (event.key !== "Tab") return false;
  const focusable = $$("#mobileSheet button:not(:disabled), #mobileSheet [tabindex]:not([tabindex='-1'])")
    .filter(element => !element.hidden && element.offsetParent !== null);
  if (!focusable.length) return false;
  const first = focusable[0];
  const last = focusable[focusable.length - 1];
  if (event.shiftKey && document.activeElement === first) {
    event.preventDefault(); last.focus(); return true;
  }
  if (!event.shiftKey && document.activeElement === last) {
    event.preventDefault(); first.focus(); return true;
  }
  return false;
}

async function screenshotBitmap(file) {
  if (window.createImageBitmap) {
    try { return await createImageBitmap(file, { imageOrientation: "from-image" }); }
    catch (_error) { /* Image decoding fallback below */ }
  }
  const source = URL.createObjectURL(file);
  try {
    const image = new Image();
    image.src = source;
    await image.decode();
    return image;
  } finally { URL.revokeObjectURL(source); }
}

async function prepareScreenshot(file) {
  const supported = /^(image\/png|image\/jpeg|image\/webp)$/i.test(file?.type || "");
  if (!file || !supported) throw new Error("请选择 PNG、JPEG 或 WebP 截图");
  if (file.size > 24 * 1024 * 1024) throw new Error("截图过大，请选择 24 MB 以内的图片");
  const bitmap = await screenshotBitmap(file);
  const sourceWidth = bitmap.naturalWidth || bitmap.width;
  const sourceHeight = bitmap.naturalHeight || bitmap.height;
  if (Math.min(sourceWidth, sourceHeight) < 320) throw new Error("截图分辨率太低，请上传完整棋盘截图");
  const maxEdge = 2600;
  const scale = Math.min(1, maxEdge / Math.max(sourceWidth, sourceHeight));
  const width = Math.max(1, Math.round(sourceWidth * scale));
  const height = Math.max(1, Math.round(sourceHeight * scale));
  const canvas = document.createElement("canvas");
  canvas.width = width; canvas.height = height;
  const context = canvas.getContext("2d", { alpha: false });
  context.fillStyle = "#fff"; context.fillRect(0, 0, width, height);
  context.drawImage(bitmap, 0, 0, width, height);
  if (typeof bitmap.close === "function") bitmap.close();
  const dataUrl = canvas.toDataURL("image/png");
  return { dataUrl, encoded: dataUrl.split(",", 2)[1], width, height };
}

async function uploadScreenshot(file) {
  if (!file || S.busy) return;
  try {
    const prepared = await prepareScreenshot(file);
    exitSimulation(false);
    S.analysis = S.solution = S.panelSolution = S.liveState = S.state = null;
    S.selectedId = null; S.processTrace = [];
    ui.image.onload = () => { fitOverlay(); draw(); };
    ui.image.src = prepared.dataUrl;
    ui.calibrationImage.src = prepared.dataUrl;
    ui.reviewImage.src = prepared.dataUrl;
    ui.viewport.classList.remove("empty");
    $("#emptyState").hidden = true;
    renderAll();
    await start("upload", {
      image_data: prepared.encoded,
      file_name: file.name || "相册截图.png",
      capture_if_missing: false,
    });
  } catch (error) { toast(errorText(error), true); }
  finally { ui.screenshotInput.value = ""; }
}

function chooseScreenshot() {
  if (!S.busy) ui.screenshotInput.click();
}
