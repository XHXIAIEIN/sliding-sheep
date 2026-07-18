// 事件绑定与应用启动
"use strict";

async function pause() {
  try { const result = await call("workflow_cancel"); if (!result.ok) throw new Error(result.error); applyJob(result.job || {}); }
  catch (error) { toast(errorText(error), true); }
}

async function hardReset() {
  try {
    const result = await call("hard_refresh"); if (!result.ok) throw new Error(result.error);
    S.pollToken++; S.jobId = null; S.jobView = null; S.panelSolution = null;
    S.analysis = S.solution = S.liveState = S.state = null; S.selectedId = null; S.editorPieceId = null;
    S.simulationIndex = null; S.backgroundMode = "dim";
    S.quickAdding = false; S.quickDraftCells = []; S.quickMessage = null; S.reviewGrid = null;
    S.boardConfirming = false; S.reviewHoverCell = null; setBusy(false);
    ui.image.removeAttribute("src"); ui.viewport.classList.add("empty"); $("#emptyState").hidden = false;
    renderAll(); applyJob({ phase: "idle", busy: false, detail: "应用已重置", elapsed_ms: 0 });
    await refreshTargets(); toast("应用已强刷；运行设置、校准与学习数据已保留");
  } catch (error) { toast(errorText(error), true); }
}

async function openReview() {
  try {
    S.quickAdding = false; S.quickDraftCells = [];
    const grid = await call("editor_grid");
    if (!grid.ok) throw new Error(grid.error);
    S.reviewGrid = grid; S.reviewMode = "select"; S.draftCells = []; S.editorDraft = null; S.editorPieceId = null;
    S.editorSaving = false; S.boardConfirming = false; S.reviewHoverCell = null;
    // editor_grid returns the screenshot, board snapshot and geometry from the
    // same backend frame. Never compose manual review from a cached preview.
    applyPayload(grid);
    const pieces = S.state?.pieces || [];
    const current = pieces.find(piece => String(piece.id) === String(S.selectedId));
    const first = current || pieces.find(piece => piece.review) || pieces[0];
    if (first) selectPiece(first.id); else renderPieces();
    $("#reviewMessage").textContent = first?.review ? `已定位待复核棋子 #${first.id}` : "直接点击棋盘中的对象开始校验。";
    if (!ui.review.open) ui.review.showModal();
    try { await ui.reviewImage.decode(); } catch (_error) { /* load event retries layout */ }
    scheduleReviewGeometrySync();
  } catch (error) { toast(errorText(error), true); }
}

function bind() {
  bindRuntimeSettings();
  $$('[data-app-mode]').forEach(button => button.addEventListener("click", async () => {
    const changed = S.appMode !== button.dataset.appMode;
    setAppMode(button.dataset.appMode);
    if (changed) await syncHostWindowMode(S.appMode);
    if (changed && S.appMode === "operator") await refreshTargets();
  }));
  ui.mobileUploadQuick.onclick = chooseScreenshot;
  ui.mobileUploadHero.onclick = chooseScreenshot;
  ui.screenshotInput.addEventListener("change", event => uploadScreenshot(event.target.files?.[0]));
  ui.mobileSolve.onclick = () => start("solve", { capture_if_missing: false });
  ui.mobileReview.onclick = () => openMobileSheet("analysis");
  ui.mobileSheetPeek.onclick = () => openMobileSheet(S.solution ? "solution" : "analysis");
  ui.mobileSheetClose.onclick = closeMobileSheet;
  ui.mobileSheetScrim.onclick = closeMobileSheet;
  ui.mobileSheetHeader.addEventListener("pointerdown", beginMobileSheetDrag);
  ui.mobileSheetHeader.addEventListener("pointermove", moveMobileSheetDrag);
  ui.mobileSheetHeader.addEventListener("pointerup", endMobileSheetDrag);
  ui.mobileSheetHeader.addEventListener("pointercancel", endMobileSheetDrag);
  $$('[data-mobile-tab]').forEach(button => button.addEventListener("click", () => setMobileTab(button.dataset.mobileTab)));
  for (const [mobile, desktop] of [
    ["#mobileShowGrid", "#showGrid"], ["#mobileShowPieces", "#showPieces"],
    ["#mobileShowDirections", "#showDirections"],
  ]) $(mobile).addEventListener("change", event => {
    $(desktop).checked = event.target.checked; draw();
  });
  $$(".workflow-button").forEach(button => button.addEventListener("click", () => start(button.dataset.action)));
  $("#refreshTargets").onclick = refreshTargets; ui.target.onchange = () => selectTarget().catch(error => toast(errorText(error), true));
  ui.pause.onclick = pause; $("#hardResetButton").onclick = hardReset;
  $("#calibrateButton").onclick = () => openCalibration();
  $("#reviewButton").onclick = focusQuickReview;
  $("#quickAddPiece").onclick = toggleQuickAdd;
  $("#quickDeletePiece").onclick = quickDeleteSelected;
  $("#quickUndo").onclick = quickUndo;
  $("#quickConfirmBoard").onclick = quickConfirmBoard;
  $("#quickAdvancedReview").onclick = openReview;
  ui.quickSpecies.addEventListener("change", event => {
    S.quickSpecies = event.target.value;
    S.quickMessage = null;
    if (S.quickAdding) { S.quickDraftCells = []; renderQuickReview(); draw(); }
    else quickUpdateSelected({ species: S.quickSpecies });
  });
  $$('[data-quick-facing]').forEach(button => button.addEventListener("click", () => {
    S.quickFacing = button.dataset.quickFacing;
    S.quickMessage = null;
    if (S.quickAdding) renderQuickReview();
    else quickUpdateSelected({ facing: S.quickFacing });
  }));
  ui.sandboxPrevious.onclick = () => stepSimulation(-1);
  ui.sandboxNext.onclick = () => stepSimulation(1);
  ui.sandboxBackground.onclick = cycleSimulationBackground;
  ui.sandboxExit.onclick = () => exitSimulation();
  $("#showGrid").onchange = draw; $("#showPieces").onchange = draw; $("#showDirections").onchange = draw;
  ui.overlay.addEventListener("pointerdown", handleMainBoardPointer);
  ui.reviewCanvas.addEventListener("pointerdown", handleReviewPointer);
  ui.reviewCanvas.addEventListener("pointermove", updatePieceDrag);
  ui.reviewCanvas.addEventListener("pointermove", updateReviewHover);
  ui.reviewCanvas.addEventListener("pointerleave", clearReviewHover);
  ui.reviewCanvas.addEventListener("pointerup", event => finishPieceDrag(event));
  ui.reviewCanvas.addEventListener("pointercancel", event => finishPieceDrag(event, true));
  ui.reviewImage.addEventListener("load", scheduleReviewGeometrySync);
  $("#selectPieceMode").onclick = enterSelectMode; $("#addPieceMode").onclick = () => enterAddMode();
  $("#emptyAddPiece").onclick = () => enterAddMode();
  $("#previousReview").onclick = () => selectRelativeReview(-1);
  $("#nextReview").onclick = () => selectRelativeReview(1);
  $("#wolfMode").onclick = () => enterObstacleMode("wolf"); $("#fenceMode").onclick = () => enterObstacleMode("fence");
  $("#clearCellMode").onclick = () => enterObstacleMode("clear");
  $$('[data-fence-direction]').forEach(button => button.addEventListener("click", () => {
    S.fenceDirection = button.dataset.fenceDirection;
    $$('[data-fence-direction]').forEach(item => item.classList.toggle("selected", item === button));
    $("#reviewMessage").textContent = `已选择${button.textContent.trim()}；点击棋盘格添加。`;
    drawReview();
  }));
  $$(".species-palette button").forEach(button => button.addEventListener("click", () => chooseEditorSpecies(button.dataset.species)));
  $$(".direction-pad button").forEach(button => button.addEventListener("click", () => chooseEditorFacing(button.dataset.facing)));
  $("#editorAwake").addEventListener("change", event => { if (S.editorDraft) { S.editorDraft.awake = event.target.checked; commitSelectedDraft(); } });
  $("#editorHitLimit").addEventListener("input", event => { if (S.editorDraft) S.editorDraft.hitLimit = +event.target.value || 3; });
  $("#editorHitsRemaining").addEventListener("input", event => { if (S.editorDraft) S.editorDraft.hitsRemaining = +event.target.value || 3; });
  $("#editorHitLimit").addEventListener("change", commitSelectedDraft);
  $("#editorHitsRemaining").addEventListener("change", commitSelectedDraft);
  $("#applyPieceEdit").onclick = () => applyPieceEdit(); $("#deletePiece").onclick = deleteSelectedPiece;
  $("#duplicatePiece").onclick = duplicateSelectedPiece; $("#cancelEditorDraft").onclick = cancelEditorAction;
  $("#editorUndo").onclick = () => editHistory("undo"); $("#editorRedo").onclick = () => editHistory("redo");
  $("#editorReset").onclick = () => { if (window.confirm("恢复本次识别结果？所有人工修正都会被清除。")) editHistory("reset"); };
  $("#confirmBoard").onclick = confirmBoard; $("#saveSample").onclick = saveSample;
  $("#resetCalibration").onclick = () => openCalibration(true); $("#saveCalibration").onclick = saveCalibration;
  $("#calibrationZoom").addEventListener("input", event => setCalibrationZoom(event.target.value));
  $("#calibrationZoomOut").onclick = () => setCalibrationZoom(S.calibrationZoom * 100 - 25);
  $("#calibrationZoomIn").onclick = () => setCalibrationZoom(S.calibrationZoom * 100 + 25);
  $("#calibrationZoomFit").onclick = () => setCalibrationZoom(100);
  ui.calibrationPreview.addEventListener("wheel", event => {
    if (!event.ctrlKey || !ui.calibration.open) return;
    event.preventDefault(); setCalibrationZoom(S.calibrationZoom * 100 + (event.deltaY < 0 ? 25 : -25));
  }, { passive: false });
  ui.calibrationImage.addEventListener("load", () => {
    if (!ui.calibration.open) return;
    requestAnimationFrame(() => { fitCalibration(); drawCalibration(); });
  });
  ui.calibrationCanvas.addEventListener("pointerdown", event => {
    const point = calibrationPointer(event); if (!point) return;
    ui.calibrationCanvas.focus({ preventScroll: true });
    const keys = ["TL", "TR", "BR", "BL"];
    const nearest = keys.reduce((best, key) => Math.hypot(S.calibration.corners[key][0] - point[0], S.calibration.corners[key][1] - point[1]) < Math.hypot(S.calibration.corners[best][0] - point[0], S.calibration.corners[best][1] - point[1]) ? key : best, keys[0]);
    const grabRadius = 20 * ui.calibrationImage.naturalWidth / ui.calibrationCanvas.clientWidth;
    if (Math.hypot(S.calibration.corners[nearest][0] - point[0], S.calibration.corners[nearest][1] - point[1]) <= grabRadius) {
      S.calibration.selectedCorner = nearest; updateCalibrationSelection(); drawCalibration();
      if (S.calibration.locked.has(nearest)) return;
      S.calibrationDrag = { kind: "corner", key: nearest };
    } else if (pointInQuad(point, S.calibration.corners)) {
      S.calibration.selectedCorner = null; updateCalibrationSelection(); drawCalibration();
      S.calibrationDrag = { kind: "board", start: point, corners: JSON.parse(JSON.stringify(S.calibration.corners)) };
    } else {
      S.calibration.selectedCorner = null; updateCalibrationSelection(); drawCalibration(); return;
    }
    ui.calibrationCanvas.setPointerCapture(event.pointerId);
  });
  ui.calibrationCanvas.addEventListener("pointermove", event => {
    if (!S.calibrationDrag) return;
    const point = calibrationPointer(event);
    if (S.calibrationDrag.kind === "corner") S.calibration.corners[S.calibrationDrag.key] = point;
    else {
      const dx = point[0] - S.calibrationDrag.start[0], dy = point[1] - S.calibrationDrag.start[1];
      for (const key of unlockedCalibrationKeys()) S.calibration.corners[key] = [S.calibrationDrag.corners[key][0] + dx, S.calibrationDrag.corners[key][1] + dy];
    }
    scheduleCalibrationPreview();
  });
  ui.calibrationCanvas.addEventListener("pointerup", () => { S.calibrationDrag = null; scheduleCalibrationPreview(0); });
  ui.calibrationCanvas.addEventListener("pointercancel", () => { S.calibrationDrag = null; scheduleCalibrationPreview(0); });
  $$('[data-calibration-lock]').forEach(input => input.addEventListener("change", () => {
    if (!S.calibration) return;
    if (input.checked) S.calibration.locked.add(input.dataset.calibrationLock);
    else S.calibration.locked.delete(input.dataset.calibrationLock);
    drawCalibration(); updateCalibrationSelection();
  }));
  const tunes = [
    ["#tuneX", "x", "#tuneXValue", 1], ["#tuneY", "y", "#tuneYValue", 1],
    ["#tuneRotation", "rotation", "#tuneRotationValue", 1],
    ["#tuneSkewX", "skewX", "#tuneSkewXValue", 3], ["#tuneSkewY", "skewY", "#tuneSkewYValue", 3],
  ];
  for (const [selector, kind, output, digits] of tunes) $(selector).addEventListener("input", event => {
    const value = +event.target.value; $(output).textContent = value.toFixed(digits); applyCalibrationTune(kind, value);
  });
  $("#calibrationRows").addEventListener("input", () => scheduleCalibrationPreview());
  $("#calibrationCols").addEventListener("input", () => scheduleCalibrationPreview());
  if (window.ResizeObserver) {
    const reviewResizeObserver = new ResizeObserver(scheduleReviewGeometrySync);
    reviewResizeObserver.observe(ui.reviewImage);
    reviewResizeObserver.observe(ui.reviewImage.parentElement);
  }
  window.addEventListener("resize", () => { fitOverlay(); draw(); fitCalibration(); drawCalibration(); scheduleReviewGeometrySync(); });
  window.addEventListener("keydown", event => {
    if (S.appMode === "reference" && ui.mobileSheet.classList.contains("open")) {
      if (event.key === "Escape") { event.preventDefault(); closeMobileSheet(); return; }
      if (trapMobileSheetFocus(event)) return;
    }
    if (!event.target.matches("input, select, textarea") && nudgeCalibrationCorner(event)) return;
    if (ui.review.open && event.ctrlKey && event.key === "Enter") { event.preventDefault(); confirmBoard(); return; }
    if (event.target.matches("input, select, textarea")) return;
    if (!ui.review.open && S.simulationIndex !== null && event.key === "ArrowLeft") { event.preventDefault(); stepSimulation(-1); }
    else if (!ui.review.open && S.simulationIndex !== null && event.key === "ArrowRight") { event.preventDefault(); stepSimulation(1); }
    else if (!ui.review.open && S.simulationIndex !== null && event.key === "Escape") { event.preventDefault(); exitSimulation(); }
    else if (ui.review.open && event.ctrlKey && event.key.toLowerCase() === "z") { event.preventDefault(); editHistory("undo"); }
    else if (ui.review.open && event.ctrlKey && event.key.toLowerCase() === "y") { event.preventDefault(); editHistory("redo"); }
    else if (ui.review.open && event.key === "Delete" && S.selectedId) { event.preventDefault(); deleteSelectedPiece(); }
    else if (ui.review.open && !event.ctrlKey && !event.metaKey && !event.altKey) {
      const key = event.key.toLowerCase();
      const digit = Number(event.key);
      if (key === "v") { event.preventDefault(); enterSelectMode(); }
      else if (key === "a") { event.preventDefault(); enterAddMode(); }
      else if (key === "w") { event.preventDefault(); enterObstacleMode("wolf"); }
      else if (key === "f") { event.preventDefault(); enterObstacleMode("fence"); }
      else if (key === "e") { event.preventDefault(); enterObstacleMode("clear"); }
      else if (key === "d" && editorTargetPiece()) { event.preventDefault(); duplicateSelectedPiece(); }
      else if (key === "r" && S.editorDraft) { event.preventDefault(); rotateEditorFacing(); }
      else if (event.key === "[") { event.preventDefault(); selectRelativeReview(-1); }
      else if (event.key === "]") { event.preventDefault(); selectRelativeReview(1); }
      else if (event.key === "Escape" && S.reviewMode !== "select") { event.preventDefault(); cancelEditorAction(); }
      else if (digit >= 1 && digit <= editorSpeciesOrder.length && S.editorDraft) {
        event.preventDefault(); chooseEditorSpecies(editorSpeciesOrder[digit - 1]);
      }
    }
  });
  window.addEventListener("app-global-hotkey", event => ({ capture: () => start("analyze"), exec: () => start("quick"), auto: () => start("auto"), replay: () => start("solve"), stop: pause }[event.detail]?.()));
}

async function ready() {
  if (S.bridgeReady) return;
  S.bridgeReady = true;
  setAppMode(preferredAppMode(), { persist: false });
  await syncHostWindowMode(S.appMode);
  await restoreRuntimeSettings();
  bind(); setBusy(false);
  if (S.appMode === "operator") await refreshTargets();
  renderAll();
}

setAppMode(preferredAppMode(), { persist: false });
window.addEventListener("pywebviewready", ready);
