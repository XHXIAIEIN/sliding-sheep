// 作业轮询、载荷投影与按钮状态
"use strict";

function options() {
  const settings = runtimeSettingsSnapshot({ normalizeFields: true });
  return {
    timeout_ms: settings.solve_timeout_s * 1000,
    elastic_timeout: settings.elastic_timeout,
    timeout_extension_ms: settings.timeout_extension_s * 1000,
    timeout_max_ms: settings.timeout_max_s * 1000,
    settle_ms: settings.settle_ms,
    max_steps: settings.max_steps,
    source_level_label: settings.source_level_label || null,
  };
}

async function refreshTargets() {
  try {
    const result = await call("list_targets");
    if (!result.ok) throw new Error(result.error);
    ui.target.replaceChildren();
    for (const win of result.windows || []) {
      const option = document.createElement("option");
      option.value = win.hwnd;
      option.textContent = `${win.title} · ${win.width}×${win.height}`;
      option.selected = String(result.selected || "") === String(win.hwnd);
      ui.target.append(option);
    }
    if (!ui.target.options.length) {
      const option = document.createElement("option");
      option.textContent = "没有找到游戏窗口";
      option.value = "";
      ui.target.append(option);
    } else if (!result.selected) {
      await selectTarget();
    }
  } catch (error) { toast(errorText(error), true); }
}

async function selectTarget() {
  if (!ui.target.value) return;
  const result = await call("select_target", ui.target.value);
  if (!result.ok) throw new Error(result.error);
}

async function start(action, extraOptions = {}) {
  if (!S.bridgeReady || S.busy) return false;
  if (S.appMode === "reference" && ["analyze", "quick", "step", "auto"].includes(action)) {
    toast("移动参考模式只提供相册分析与解法参考", true);
    return false;
  }
  // Reserve the workflow slot before the first awaited bridge call.  Without
  // this, a second click can launch another workflow during target selection.
  S.action = action;
  S.reviewGuidance = null;
  S.jobView = {
    action, busy: true,
    phase: action === "upload" ? "capturing" : action === "analyze" ? "analyzing" : "solving",
    detail: action === "upload" ? "正在读取相册截图"
      : action === "analyze" ? "正在准备采集"
      : action === "quick" ? "正在复用当前棋盘生成快速解法"
      : "正在准备求解",
    progress: {},
  };
  S.panelSolution = null;
  setBusy(true);
  try {
    if (["analyze", "upload", "solve", "quick", "step", "auto"].includes(action)) exitSimulation(false);
    const referenceWorkflow = action === "upload" || (S.appMode === "reference" && action === "solve");
    if (!referenceWorkflow) await selectTarget();
    if (["analyze", "solve"].includes(action)) {
      S.solution = null;
      S.selectedId = null;
    }
    if (action === "upload") {
      S.solution = null;
      S.selectedId = null;
    }
    if (["solve", "auto"].includes(action)) S.processTrace = [];
    renderAll();
    const result = await call("workflow_start", action, {
      ...options(), capture_if_missing: S.appMode !== "reference", ...extraOptions,
    });
    if (!result.ok) throw new Error(result.error);
    S.jobId = result.job.id;
    applyJob(result.job);
    pollJob(++S.pollToken);
    return true;
  } catch (error) {
    S.jobView = null;
    S.panelSolution = null;
    S.action = null;
    setBusy(false);
    renderAll();
    toast(errorText(error), true);
    return false;
  }
}

async function pollJob(token) {
  if (token !== S.pollToken || !S.jobId) return;
  try {
    const result = await call("workflow_status", S.jobId);
    if (!result.ok) throw new Error(result.error);
    const job = result.job;
    applyJob(job);
    if (job.busy) {
      setTimeout(() => pollJob(token), 240);
      return;
    }
    if (job.phase === "error") {
      if (!handleReviewGuidance(job.error)) toast(errorText(job.error), true);
    } else if (job.phase === "cancelled") {
      toast("已在当前点击结束后暂停");
    } else if (job.result) {
      const solutionHistory = !!job.result.solution_history;
      S.jobView = null;
      S.panelSolution = null;
      S.action = null;
      applyPayload(job.result, { solutionHistory });
      const guided = handleReviewGuidance(job.result);
      if (job.result.ok === false && !guided) toast(errorText(job.result.error), true);
    }
    if (S.reviewAfterCalibration) {
      S.reviewAfterCalibration = false;
      const blockers = job.result?.execution_blockers || [];
      const needsReview = blockers.some(item => [
        "manual_review_required", "manual_learning_confirmation_required",
        "manual_board_unconfirmed",
      ].includes(item.code));
      if (needsReview && S.state) {
        toast("校准已生效；请完成棋盘复核后再连续执行");
        setTimeout(() => focusQuickReview(), 0);
      }
    }
  } catch (error) {
    setBusy(false);
    toast(errorText(error), true);
  }
}

function setBusy(busy) {
  S.busy = !!busy;
  syncWorkflowButtons();
  ui.pause.disabled = !S.busy;
  $("#calibrateButton").disabled = S.busy;
  $("#reviewButton").disabled = S.busy || !S.liveState;
  renderQuickReview();
}

function solutionReady(solution = S.solution) {
  return !!(solution && (solution.execution_ready || solution.safe_prefix_ready));
}

function syncWorkflowButtons() {
  const blockedPlan = !!S.solution && !solutionReady();
  const referenceOnly = !!S.analysis?.reference_only;
  const safetyBlockers = (S.analysis?.execution_blockers || [])
    .filter(item => item.code !== "gesture_occlusion");
  const safetyBlocked = safetyBlockers.length > 0;
  $$(".workflow-button").forEach(button => {
    const executionAction = ["quick", "step", "auto"].includes(button.dataset.action);
    const planBlocked = executionAction && (referenceOnly || blockedPlan || safetyBlocked);
    if (!button.dataset.defaultTitle) button.dataset.defaultTitle = button.title || "";
    button.disabled = S.busy || planBlocked;
    button.classList.toggle("plan-blocked", planBlocked);
    button.title = planBlocked
      ? (referenceOnly ? "相册截图只用于分析与解法参考；重新采集桌面窗口后才可执行"
        : safetyBlockers[0]?.message || S.solution?.failure_reason
        || "当前没有可安全执行的完整解法；请先重求或复核棋盘")
      : button.dataset.defaultTitle;
  });
}

function applyJob(job) {
  S.jobView = job || null;
  if (job?.action) S.action = job.action;
  if (Object.prototype.hasOwnProperty.call(job || {}, "panel_solution")) {
    S.panelSolution = job.panel_solution || null;
  }
  if (job?.panel_analysis) {
    S.analysis = { ...(S.analysis || {}), ...job.panel_analysis };
  }
  if (Array.isArray(job?.solve_trace)) S.processTrace = job.solve_trace;
  setBusy(job.busy);
  const phase = job.phase || "idle";
  ui.runtimePill.className = `runtime-pill ${job.busy ? "busy" : phase === "error" ? "error" : phase === "done" ? "done" : "idle"}`;
  ui.runtimePill.querySelector("b").textContent = phaseNames[phase] || phase;
  ui.runtimePhase.textContent = phaseNames[phase] || phase;
  ui.runtimeTime.textContent = `${((job.elapsed_ms || 0) / 1000).toFixed(1)}s`;
  ui.runtimeDetail.textContent = job.detail || (job.error ? errorText(job.error) : "准备就绪");
  const progress = job.progress || {};
  const percent = progress.total ? Math.min(100, (progress.completed || progress.steps || 0) / progress.total * 100) : job.busy ? 35 : phase === "done" ? 100 : 0;
  ui.runtimeProgress.style.width = `${percent}%`;
  if (job.preview_state) {
    S.liveState = job.preview_state;
    if (S.simulationIndex === null) S.state = job.preview_state;
  }
  renderAll();
}

function applyPayload(payload, { solutionHistory = false } = {}) {
  if (!payload) return;
  let receivedSolution = false;
  if (payload.img) {
    ui.image.onload = () => { fitOverlay(); draw(); };
    ui.image.src = `data:image/png;base64,${payload.img}`;
    ui.calibrationImage.src = ui.image.src;
    ui.reviewImage.src = ui.image.src;
    ui.viewport.classList.remove("empty");
    $("#emptyState").hidden = true;
  }
  const hasAnalysis = payload.scene_state || payload.state || Number.isFinite(payload.rows);
  if (hasAnalysis) {
    S.analysis = { ...(S.analysis || {}), ...payload };
    if (payload.state) {
      S.liveState = payload.state;
      if (S.simulationIndex === null) S.state = payload.state;
      if (!(payload.state.pieces || []).some(piece => piece.review)) {
        S.reviewGuidance = null;
      }
    }
  }
  if (payload.solution) {
    S.solution = payload.solution;
    if (Array.isArray(payload.solution.process_trace)) S.processTrace = payload.solution.process_trace;
    S.backgroundMode = "dim";
    receivedSolution = true;
  } else if (Array.isArray(payload.moves) && Array.isArray(payload.states)) {
    S.solution = payload;
    if (Array.isArray(payload.process_trace)) S.processTrace = payload.process_trace;
    S.backgroundMode = "dim";
    receivedSolution = true;
  }
  if (S.solution?.states?.length && !S.liveState) S.liveState = S.solution.states[0];
  if (receivedSolution && !solutionHistory) selectFirstSolutionStep();
  else if (receivedSolution && solutionHistory) {
    S.simulationIndex = null;
    S.state = S.liveState || S.solution.states?.[0] || null;
    S.selectedId = null;
  }
  else if (S.simulationIndex === null && S.liveState) S.state = S.liveState;
  if (payload.level_label) $("#sourceLevel").placeholder = payload.level_label;
  renderAll();
  if (receivedSolution) {
    reflectSolutionOutcome(S.solution);
    if (S.appMode === "reference") openMobileSheet("solution");
  }
}

function renderAll() {
  syncWorkflowButtons();
  renderSafety();
  renderSearchProcess();
  renderPlan();
  renderWorkflowSummary();
  renderPieces();
  renderTypeLegend();
  draw();
  renderSimulation();
  renderQuickReview();
  renderMobile();
  $("#reviewButton").disabled = S.busy || !S.liveState;
}

function mobileNoticeItem(icon, title, detail, tone = "") {
  const item = document.createElement("article");
  item.className = `mobile-reference-item${tone ? ` ${tone}` : ""}`;
  const mark = document.createElement("i"); mark.textContent = icon;
  const copy = document.createElement("div");
  const heading = document.createElement("b"); heading.textContent = title;
  const description = document.createElement("small"); description.textContent = detail;
  copy.append(heading, description); item.append(mark, copy);
  return item;
}

function listNotices(value) {
  return Array.isArray(value) ? value.filter(item => item && typeof item === "object") : [];
}

function noticeDetail(notice) {
  const detail = notice?.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) return detail.map(item => String(item)).join(" · ");
  if (detail && typeof detail === "object") {
    try { return Object.values(detail).flat().slice(0, 3).map(item => String(item)).join(" · "); }
    catch (_error) { return ""; }
  }
  return notice?.location || notice?.reason || "";
}
