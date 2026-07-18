(() => {
  "use strict";

  const $ = (selector) => document.querySelector(selector);
  const $$ = (selector) => [...document.querySelectorAll(selector)];
  const ui = {
    target: $("#targetSelect"), runtimePill: $("#runtimePill"), runtimePhase: $("#runtimePhase"),
    runtimeTime: $("#runtimeTime"), runtimeProgress: $("#runtimeProgress"), runtimeDetail: $("#runtimeDetail"),
    processSummary: $("#processSummary"), processTimeline: $("#processTimeline"),
    pause: $("#pauseButton"), image: $("#gameImage"), overlay: $("#overlay"), viewport: $("#viewport"),
    surface: $("#boardSurface"),
    sceneBadge: $("#sceneBadge"), blockers: $("#blockers"), moveList: $("#moveList"),
    sandbox: $("#sandboxControls"), sandboxPosition: $("#sandboxPosition"),
    sandboxPrevious: $("#sandboxPrevious"), sandboxNext: $("#sandboxNext"),
    sandboxBackground: $("#sandboxBackground"), sandboxExit: $("#sandboxExit"),
    quickBar: $("#quickReviewBar"), quickStatus: $("#quickReviewStatus"),
    quickSpecies: $("#quickSpecies"), quickDirections: $("#quickDirections"),
    calibration: $("#calibrationDialog"), calibrationPreview: $("#calibrationPreview"),
    calibrationStage: $("#calibrationStage"), calibrationImage: $("#calibrationImage"),
    calibrationCanvas: $("#calibrationCanvas"), review: $("#reviewDialog"),
    reviewImage: $("#reviewImage"), reviewCanvas: $("#reviewCanvas"), toast: $("#toast"),
    screenshotInput: $("#screenshotInput"), mobileSceneTitle: $("#mobileSceneTitle"),
    mobileBoardMeta: $("#mobileBoardMeta"), mobileUploadQuick: $("#mobileUploadQuick"),
    mobileUploadHero: $("#mobileUploadHero"), mobileSolve: $("#mobileSolveButton"),
    mobileReview: $("#mobileReviewButton"), mobileSheet: $("#mobileSheet"),
    mobileSheetScrim: $("#mobileSheetScrim"), mobileSheetPeek: $("#mobileSheetPeek"),
    mobileSheetClose: $("#mobileSheetClose"), mobileSheetHeader: $(".mobile-sheet-header"),
    mobileSheetTitle: $("#mobileSheetTitle"), mobileInsightTitle: $("#mobileInsightTitle"),
    mobileInsightDetail: $("#mobileInsightDetail"), mobileInsightDot: $("#mobileInsightDot"),
    mobileSafetyHero: $("#mobileSafetyHero"), mobileSafetyList: $("#mobileSafetyList"),
    mobileSolutionHero: $("#mobileSolutionHero"), mobileMoveList: $("#mobileMoveList"),
  };
  const S = {
    bridgeReady: false, busy: false, jobId: null, pollToken: 0, action: null,
    jobView: null, panelSolution: null,
    processTrace: [],
    analysis: null, solution: null, liveState: null, state: null, selectedId: null,
    simulationIndex: null, backgroundMode: "dim",
    quickAdding: false, quickDraftCells: [], quickSpecies: "sheep", quickFacing: "R",
    quickSaving: false, quickMessage: null,
    calibration: null, calibrationDrag: null, calibrationZoom: 1,
    reviewGrid: null, reviewMode: "select", draftCells: [], fenceDirection: "H", editorDraft: null, editorPieceId: null,
    editorSaving: false, boardConfirming: false, reviewDrag: null, reviewHoverCell: null, reviewAfterCalibration: false,
    reviewGuidance: null, appMode: "operator", mobileTab: "analysis", mobileSheetDrag: null,
  };

  const phaseNames = {
    idle: "空闲", capturing: "采集画面", analyzing: "分析棋盘", solving: "求解中",
    "opening-coarse": "开局粗解", "quick-exit": "快速解法", "exit-closure": "清理直出", "exact-a*": "最优搜索", "macro-beam": "宏步搜索",
    "randomized-macro": "多路搜索", "weighted-a*": "加权搜索", beam: "扩展搜索",
    "online-greedy": "快速候选", greedy: "整理提示",
    "solve-budget": "分配预算", "budget-extension": "自动续时",
    "cache-hit": "读取缓存",
    executing: "执行点击", verifying: "核对棋盘", pausing: "等待当前点击结束",
    done: "已完成", cancelled: "已暂停", error: "失败",
  };
  const sceneNames = { gameplay: "棋盘", victory: "过关", transition: "转场", popup: "弹窗", loading: "加载", unknown: "未知" };
  const speciesNames = { sheep: "普通羊", goat: "山羊", rocket: "火箭羊", cattle: "牛", elephant: "大象", pink_sheep: "粉羊", black_sheep: "黑羊", pig: "猪", bomb: "炸弹羊" };
  const facingNames = { U: "↑", D: "↓", L: "←", R: "→" };
  const reviewReasonNames = {
    low_occupancy_confidence: "占格置信度偏低",
    detector_facing_disagreement: "方向识别结果冲突",
    detector_axis_disagreement: "横竖方向识别冲突",
    weak_cattle_cell_only: "牛棋子证据偏弱",
    manual_learning_single_observation: "学习样本只有一次观察",
    manual_direction_single_observation: "方向学习只有一次观察",
  };
  const editorSpeciesOrder = ["sheep", "rocket", "bomb", "pink_sheep", "black_sheep", "pig", "cattle", "goat", "elephant"];
  const editorModeCopy = {
    select: { name: "选择与移动", hint: "点击选择，拖动可换格；类型和方向修改会立即保存", pill: "选择模式", inspector: "当前棋子", canvas: "点击选择 · 拖动换格" },
    add: { name: "补充棋子", hint: "先选类型和方向，再点击棋子占用的棋盘格", pill: "新增模式", inspector: "新增棋子", canvas: "点击空格 · 完成后添加" },
    wolf: { name: "添加狼危险格", hint: "单击空格添加狼格；已有标记不会被误触删除", pill: "狼格工具", inspector: "添加狼格", canvas: "点击空格添加 · 清除请按 E" },
    fence: { name: "添加栅栏", hint: "先在右侧选栅栏位置，再点击棋盘格添加", pill: "栅栏工具", inspector: "添加栅栏", canvas: "选择栅栏位置 · 点击添加" },
    clear: { name: "清除误判", hint: "点击格子，一次清掉其中的棋子、狼格和栅栏", pill: "清除工具", inspector: "清除误判", canvas: "点击误判格清除 · Ctrl Z 撤销" },
  };
  const simulationSelectionColor = "#e11d48";
  const speciesStyles = {
    sheep: { color: "#2563eb", fill: "rgba(37,99,235,.08)", mark: "羊" },
    rocket: { color: "#7c3aed", fill: "rgba(124,58,237,.16)", mark: "火" },
    bomb: { color: "#dc2626", fill: "rgba(220,38,38,.16)", mark: "炸" },
    pink_sheep: { color: "#db2777", fill: "rgba(219,39,119,.16)", mark: "粉" },
    black_sheep: { color: "#111827", fill: "rgba(17,24,39,.18)", mark: "黑" },
    pig: { color: "#c2410c", fill: "rgba(194,65,12,.15)", mark: "猪" },
    cattle: { color: "#92400e", fill: "rgba(146,64,14,.16)", mark: "牛" },
    goat: { color: "#0f766e", fill: "rgba(15,118,110,.15)", mark: "山" },
    elephant: { color: "#475569", fill: "rgba(71,85,105,.16)", mark: "象" },
  };
  const runtimeSettingsStorageKey = "sheep-solver:runtime-settings:v1";
  const appModeStorageKey = "sheep-solver:app-mode:v1";
  let runtimeSettingsClock = 0;
  let runtimeSettingsBackendWarning = false;
  let mobileSheetReturnFocus = null;

  function boundedSetting(value, fallback, minimum, maximum) {
    const number = Number(value);
    const normalized = Number.isFinite(number) ? Math.round(number) : fallback;
    return Math.max(minimum, Math.min(maximum, normalized));
  }

  function normalizeRuntimeSettings(data = {}) {
    const initial = boundedSetting(data.solve_timeout_s, 10, 1, 60);
    return {
      solve_timeout_s: initial,
      timeout_extension_s: boundedSetting(data.timeout_extension_s, 5, 1, 60),
      timeout_max_s: Math.max(initial, boundedSetting(data.timeout_max_s, 30, 1, 300)),
      elastic_timeout: typeof data.elastic_timeout === "boolean" ? data.elastic_timeout : true,
      settle_ms: boundedSetting(data.settle_ms, 60, 20, 3000),
      max_steps: boundedSetting(data.max_steps, 200, 1, 500),
      source_level_label: String(data.source_level_label || "").trim().slice(0, 120),
      updated_at_ms: Math.max(0, boundedSetting(
        data.updated_at_ms, 0, 0, Date.now() + 86400000)),
    };
  }

  function applyRuntimeSettings(data) {
    const settings = normalizeRuntimeSettings(data);
    $("#solveTimeout").value = settings.solve_timeout_s;
    $("#timeoutExtension").value = settings.timeout_extension_s;
    $("#timeoutMax").value = settings.timeout_max_s;
    $("#elasticTimeout").checked = settings.elastic_timeout;
    $("#settleMs").value = settings.settle_ms;
    $("#maxSteps").value = settings.max_steps;
    $("#sourceLevel").value = settings.source_level_label;
    runtimeSettingsClock = Math.max(runtimeSettingsClock, settings.updated_at_ms);
    return settings;
  }

  function runtimeSettingsSnapshot({ normalizeFields = false, touch = false } = {}) {
    const settings = normalizeRuntimeSettings({
      solve_timeout_s: $("#solveTimeout").value,
      timeout_extension_s: $("#timeoutExtension").value,
      timeout_max_s: $("#timeoutMax").value,
      elastic_timeout: $("#elasticTimeout").checked,
      settle_ms: $("#settleMs").value,
      max_steps: $("#maxSteps").value,
      source_level_label: $("#sourceLevel").value,
      updated_at_ms: runtimeSettingsClock,
    });
    if (touch) {
      settings.updated_at_ms = Math.max(Date.now(), runtimeSettingsClock + 1);
      runtimeSettingsClock = settings.updated_at_ms;
    }
    if (normalizeFields) applyRuntimeSettings(settings);
    return settings;
  }

  function readRuntimeSettingsCache() {
    try {
      const payload = JSON.parse(localStorage.getItem(runtimeSettingsStorageKey) || "null");
      return payload?.schema === 1 ? normalizeRuntimeSettings(payload.settings) : null;
    } catch (_error) { return null; }
  }

  function writeRuntimeSettingsCache(settings) {
    try {
      localStorage.setItem(runtimeSettingsStorageKey, JSON.stringify({
        schema: 1, settings: normalizeRuntimeSettings(settings),
      }));
    } catch (_error) { /* backend JSON remains the durable fallback */ }
  }

  async function persistRuntimeSettings({ normalizeFields = false } = {}) {
    const settings = runtimeSettingsSnapshot({ normalizeFields, touch: true });
    writeRuntimeSettingsCache(settings);
    try {
      const result = await call("save_runtime_settings", settings);
      if (!result.ok) throw new Error(result.error);
      const saved = normalizeRuntimeSettings(result.settings);
      if (saved.updated_at_ms >= settings.updated_at_ms) writeRuntimeSettingsCache(saved);
    } catch (error) {
      if (!runtimeSettingsBackendWarning) {
        runtimeSettingsBackendWarning = true;
        toast(`运行设置后端副本保存失败：${errorText(error)}`, true);
      }
    }
  }

  async function restoreRuntimeSettings() {
    const cached = readRuntimeSettingsCache();
    if (cached) applyRuntimeSettings(cached);
    try {
      const result = await call("load_runtime_settings");
      if (!result.ok) throw new Error(result.error);
      const backend = normalizeRuntimeSettings(result.settings);
      if (cached && cached.updated_at_ms > backend.updated_at_ms) {
        applyRuntimeSettings(cached);
        const synced = await call("save_runtime_settings", cached);
        if (!synced.ok) throw new Error(synced.error);
      } else {
        applyRuntimeSettings(backend);
        writeRuntimeSettingsCache(backend);
      }
    } catch (_error) {
      // Keep the local cache or HTML defaults; startup must remain available.
    }
  }

  function bindRuntimeSettings() {
    for (const selector of [
      "#solveTimeout", "#timeoutExtension", "#timeoutMax",
      "#settleMs", "#maxSteps", "#sourceLevel",
    ]) {
      $(selector).addEventListener("input", () => { void persistRuntimeSettings(); });
      $(selector).addEventListener("change", () => { void persistRuntimeSettings({ normalizeFields: true }); });
    }
    $("#elasticTimeout").addEventListener("change", () => {
      void persistRuntimeSettings({ normalizeFields: true });
    });
  }

  function bridge() {
    if (!window.pywebview || !window.pywebview.api) throw new Error("后端桥尚未就绪");
    return window.pywebview.api;
  }

  async function call(method, ...args) {
    const api = bridge();
    const fn = api[method];
    if (typeof fn !== "function") throw new Error(`后端缺少接口 ${method}`);
    return await fn(...args);
  }

  function errorText(value) {
    if (!value) return "未知错误";
    if (typeof value === "string") return value;
    return value.message || value.error || JSON.stringify(value);
  }

  let toastTimer = 0;
  function toast(message, error = false) {
    clearTimeout(toastTimer);
    ui.toast.textContent = message;
    ui.toast.className = `toast show${error ? " error" : ""}`;
    toastTimer = setTimeout(() => ui.toast.className = "toast", 3200);
  }

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

  function renderMobileAnalysis() {
    const data = S.analysis || {};
    const blockers = listNotices(data.execution_blockers);
    const advisories = listNotices(data.advisories);
    const scene = data.scene_state;
    const ready = scene === "gameplay" && !!S.liveState;
    const heroStrong = ui.mobileSafetyHero.querySelector("strong");
    const heroSmall = ui.mobileSafetyHero.querySelector("small");
    heroStrong.textContent = !scene ? "等待截图" : ready
      ? blockers.length ? "建议先复核" : "识别完成"
      : sceneNames[scene] || "未识别为棋盘";
    heroSmall.textContent = !scene ? "上传后自动分析"
      : `${Number(data.rows) || "?"} × ${Number(data.cols) || "?"} 网格 · ${Number(data.count ?? S.liveState?.pieces?.length) || 0} 只棋子`;
    ui.mobileSafetyList.replaceChildren();
    if (!scene) {
      ui.mobileSafetyList.append(mobileNoticeItem("＋", "选择游戏截图", "建议保留完整棋盘边界，不要裁掉出口。"));
      return;
    }
    if (ready && !blockers.length) {
      ui.mobileSafetyList.append(mobileNoticeItem("✓", "棋盘结构可用", "可以生成解法，也可以展开标注核对识别结果。"));
    }
    for (const notice of blockers.slice(0, 4)) {
      ui.mobileSafetyList.append(mobileNoticeItem("!", notice.message || notice.code || "需要处理", noticeDetail(notice) || "生成解法前建议检查识别结果。", "error"));
    }
    for (const notice of advisories.slice(0, Math.max(0, 5 - blockers.length))) {
      ui.mobileSafetyList.append(mobileNoticeItem("·", notice.message || notice.code || "识别提示", noticeDetail(notice) || "这不会阻止查看解法参考。", "warning"));
    }
    if (!ready && !blockers.length) {
      ui.mobileSafetyList.append(mobileNoticeItem("?", sceneNames[scene] || "无法识别", data.scene_reason || "请确认上传的是包含完整棋盘的游戏截图。", "warning"));
    }
  }

  function renderMobileSolution() {
    const solution = S.panelSolution || S.solution;
    const heroStrong = ui.mobileSolutionHero.querySelector("strong");
    const heroSmall = ui.mobileSolutionHero.querySelector("small");
    ui.mobileMoveList.replaceChildren();
    if (!solution) {
      heroStrong.textContent = S.jobView?.busy && S.action === "solve" ? "正在搜索" : "尚未生成";
      heroSmall.textContent = S.jobView?.busy && S.action === "solve"
        ? (S.jobView.detail || "正在计算解法") : "分析成功后可生成完整计划";
      return;
    }
    const moves = solution.moves || [];
    heroStrong.textContent = solution.solved ? `${moves.length} 步可清盘`
      : moves.length ? `${moves.length} 步参考前缀` : "暂无可用步骤";
    heroSmall.textContent = solution.solved ? `计划完成后剩 0 只 · ${solution.kind || "求解器"}`
      : `计划后剩 ${solution.remaining ?? "?"} 只 · 可逐步查看沙盘`;
    for (const [index, move] of moves.slice(0, 160).entries()) {
      const item = document.createElement("li");
      item.classList.toggle("selected", S.simulationIndex === index);
      item.tabIndex = 0; item.setAttribute("role", "button");
      const number = document.createElement("span"); number.textContent = move.step || index + 1;
      const copy = document.createElement("div");
      const title = document.createElement("b");
      title.textContent = `#${move.piece} · ${speciesNames[move.species] || "羊"}`;
      const detail = document.createElement("small");
      detail.textContent = move.desc || `${move.result === "EXIT" ? "离场" : "移动"}${move.distance ? ` ${move.distance} 格` : ""}`;
      const direction = document.createElement("em"); direction.textContent = facingNames[move.direction] || move.direction || "→";
      copy.append(title, detail); item.append(number, copy, direction);
      const select = () => { closeMobileSheet(); enterSimulation(index); };
      item.addEventListener("click", select);
      item.addEventListener("keydown", event => {
        if (event.key !== "Enter" && event.key !== " ") return;
        event.preventDefault(); select();
      });
      ui.mobileMoveList.append(item);
    }
  }

  function renderMobile() {
    if (!ui.mobileSceneTitle) return;
    const data = S.analysis || {};
    const solution = S.panelSolution || S.solution;
    const blockers = listNotices(data.execution_blockers);
    const pieceCount = Number(data.count ?? S.liveState?.pieces?.length);
    const busy = !!S.jobView?.busy;
    document.body.classList.toggle("mobile-has-board", !!S.liveState);
    document.body.classList.toggle("mobile-has-insight", busy || !!data.scene_state || !!solution);
    ui.mobileSheet.setAttribute("aria-busy", String(busy));
    ui.mobileSceneTitle.textContent = busy ? phaseNames[S.jobView.phase] || "处理中"
      : solution ? solution.solved ? "解法已就绪" : "参考步骤已就绪"
        : data.scene_state === "gameplay" ? blockers.length ? "识别待复核" : "棋盘已识别"
          : data.scene_state ? sceneNames[data.scene_state] || "未识别为棋盘" : "等待截图";
    ui.mobileBoardMeta.textContent = busy ? S.jobView.detail || "请稍候"
      : data.scene_state ? `${Number(data.rows) || "?"}×${Number(data.cols) || "?"} · ${Number.isFinite(pieceCount) ? pieceCount : 0} 只`
        : "从相册选择游戏截图";
    ui.mobileSolve.disabled = busy || !S.liveState;
    ui.mobileReview.disabled = busy || !data.scene_state;
    ui.mobileUploadQuick.disabled = busy;
    ui.mobileUploadHero.disabled = busy;
    ui.mobileInsightDot.className = busy ? "warning" : solution ? "ready"
      : blockers.length ? "error" : data.scene_state === "gameplay" ? "ready" : "";
    ui.mobileInsightTitle.textContent = busy ? phaseNames[S.jobView.phase] || "处理中"
      : solution ? `${solution.total || solution.moves?.length || 0} 步解法参考`
        : data.scene_state === "gameplay" ? blockers.length ? `${blockers.length} 项需要留意` : "识别结果可用"
          : "尚未分析";
    ui.mobileInsightDetail.textContent = busy ? S.jobView.detail || "正在处理截图"
      : solution ? `点击展开步骤 · 计划后剩 ${solution.remaining ?? 0} 只`
        : data.scene_state === "gameplay" ? "点击查看棋盘识别报告" : "上传截图后查看识别与解法";
    $("#mobileShowGrid").checked = $("#showGrid").checked;
    $("#mobileShowPieces").checked = $("#showPieces").checked;
    $("#mobileShowDirections").checked = $("#showDirections").checked;
    renderMobileAnalysis(); renderMobileSolution();
  }

  function renderWorkflowSummary() {
    const track = ui.runtimeProgress.parentElement;
    const job = S.jobView;
    if (job?.busy || ["error", "cancelled"].includes(job?.phase)) {
      track.hidden = !job.busy;
      return;
    }
    track.hidden = true;
    if (S.reviewGuidance) {
      ui.runtimePhase.textContent = "等待复核";
      ui.runtimeTime.textContent = S.reviewGuidance.location || "";
      ui.runtimeDetail.textContent = S.reviewGuidance.review_message
        || "请检查或修正已选中的低置信度棋子，然后继续执行。";
      return;
    }
    const solution = S.panelSolution || S.solution;
    if (solution) {
      const failure = solutionFailure(solution);
      const total = Number(solution.total) || (solution.moves || []).length;
      ui.runtimeTime.textContent = total ? `${total} 步` : "";
      if (solution.quick_exit) {
        ui.runtimePhase.textContent = solution.solved ? "棋盘已清空" : total ? "快速计划" : "没有直出步骤";
        ui.runtimeDetail.textContent = total
          ? `${solution.exit_layers || 0} 层可直接离场，计划后剩 ${solution.remaining ?? "?"} 只。`
          : "当前棋盘没有可直接离场的普通羊。";
      } else if (failure) {
        ui.runtimePhase.textContent = failure.title;
        ui.runtimeDetail.textContent = failure.detail;
      } else {
        ui.runtimePhase.textContent = "完整计划";
        ui.runtimeDetail.textContent = solution.bomb_count
          ? "可清盘；特殊棋子将在执行时逐步核验。"
          : "可清盘并连续执行。";
      }
      return;
    }
    const data = S.analysis || {};
    const hardBlockers = (data.execution_blockers || []).filter(item => item.code !== "gesture_occlusion");
    const pieces = Number.isFinite(S.liveState?.pieces?.length)
      ? S.liveState.pieces.length : Number.isFinite(data.count) ? data.count : null;
    ui.runtimeTime.textContent = pieces === null ? "" : `${pieces} 只`;
    if (data.scene_state === "gameplay") {
      ui.runtimePhase.textContent = hardBlockers.length ? "需要复核" : "等待生成解法";
      ui.runtimeDetail.textContent = hardBlockers[0]?.message || "棋盘已分析，可以生成解题计划。";
    } else {
      ui.runtimePhase.textContent = "等待分析";
      ui.runtimeDetail.textContent = "选择游戏窗口，然后采集并分析。";
    }
  }

  function renderTypeLegend() {
    const host = $("#typeLegend"); host.replaceChildren();
    const present = new Set((S.state?.pieces || []).map(piece => piece.species || "sheep"));
    for (const species of Object.keys(speciesStyles)) {
      if (!present.has(species)) continue;
      const item = document.createElement("span"); item.style.setProperty("--type-color", speciesStyles[species].color);
      item.textContent = speciesNames[species] || species; host.append(item);
    }
    if ((S.state?.dynamic_hazards?.length || 0) + (S.state?.hazards?.length || 0)) {
      const item = document.createElement("span"); item.style.setProperty("--type-color", "#b42318"); item.textContent = "狼"; host.append(item);
    }
    if (S.state?.fences?.length) {
      const item = document.createElement("span"); item.style.setProperty("--type-color", "#8a5a2b"); item.textContent = "栅栏"; host.append(item);
    }
  }

  function reviewNoticePieces(notice) {
    const detailed = notice?.detail?.pieces;
    if (Array.isArray(detailed) && detailed.length) return detailed;
    return (S.liveState?.pieces || S.state?.pieces || [])
      .filter(piece => piece.review)
      .map(piece => ({
        id: String(piece.id),
        cells: piece.cells || [],
        location: (piece.cells || []).map(cellName).join("–"),
        reason: piece.review_reason,
      }));
  }

  function focusReviewPiece(info, message = null) {
    const pieceId = String(info?.id ?? info?.piece_id ?? "");
    if (!pieceId) return false;
    if (S.simulationIndex !== null) exitSimulation(false);
    const piece = (S.state?.pieces || []).find(item => String(item.id) === pieceId);
    if (!piece) {
      toast(`当前棋盘中找不到待复核棋子 #${pieceId}，请重新分析`, true);
      return false;
    }
    const location = info.location || (piece.cells || []).map(cellName).join("–");
    const reason = reviewReasonNames[info.reason] || info.reason;
    const prompt = message || `请检查或修正棋子 #${pieceId}（${location}）`;
    S.reviewGuidance = {
      ...info,
      piece_id: pieceId,
      location,
      review_message: prompt,
    };
    S.quickMessage = `${prompt}${reason ? `；原因：${reason}` : ""}。无误可直接点“完成复核”`;
    selectPiece(pieceId, true);
    renderQuickReview();
    ui.quickBar.scrollIntoView({ block: "nearest", behavior: "smooth" });
    toast(`已选中 #${pieceId}（${location}），请检查或修正`);
    return true;
  }

  function handleReviewGuidance(payload) {
    if (!payload || !(payload.review_required
        || payload.error_code === "manual_review_required")) return false;
    return focusReviewPiece({
      id: payload.piece_id,
      cells: payload.cells || [],
      location: payload.location,
      reason: payload.review_reason,
    }, payload.review_message);
  }

  function renderSafety() {
    const data = S.analysis || {};
    const scene = data.scene_state || "unknown";
    const blockers = data.execution_blockers || [];
    const hardBlockers = blockers.filter(item => item.code !== "gesture_occlusion");
    const advisories = [...(data.advisories || []),
      ...blockers.filter(item => item.code === "gesture_occlusion")]
      .filter((item, index, all) => all.findIndex(other => other.code === item.code
        && other.message === item.message) === index);
    const executionAllowed = scene === "gameplay" && !hardBlockers.length;
    ui.sceneBadge.textContent = sceneNames[scene] || scene;
    ui.sceneBadge.className = `scene-badge ${scene === "gameplay" ? (hardBlockers.length ? "blocked" : "gameplay") : scene === "victory" ? "victory" : hardBlockers.length ? "blocked" : "unknown"}`;
    const livePieceCount = S.liveState?.pieces?.length;
    $("#pieceCount").textContent = Number.isFinite(livePieceCount)
      ? livePieceCount : Number.isFinite(data.count) ? data.count : "—";
    const health = data.metrics?.health_score;
    $("#healthScore").textContent = Number.isFinite(health) ? `${Math.round(health * (health <= 1 ? 100 : 1))}` : "—";
    $("#executableValue").textContent = scene !== "gameplay" ? "未检查" : executionAllowed ? "可执行" : "已阻止";
    $("#safetyMeta").hidden = scene !== "gameplay";
    $("#boardTitle").textContent = scene === "gameplay" ? `${data.rows || S.state?.rows || "?"}×${data.cols || S.state?.cols || "?"} 棋盘` : (data.scene_reason || "等待采集");
    $("#captureInfo").textContent = data.capture ? `${data.capture.win.w}×${data.capture.win.h}` : "未采集";
    ui.blockers.replaceChildren();
    if (!hardBlockers.length && scene === "gameplay") {
      const ok = document.createElement("div");
      ok.className = "blocker safe";
      const liveGuard = !!(S.solution?.bomb_live_control || S.solution?.wolf_track?.length || S.solution?.wolf_zone?.length);
      const title = document.createElement("b");
      title.textContent = "安全检查通过";
      ok.append(title);
      if (liveGuard) {
        const detail = document.createElement("small");
        detail.textContent = "特殊棋子将在执行时逐步核验。";
        ok.append(detail);
      }
      ui.blockers.append(ok);
    } else if (!hardBlockers.length) {
      ui.blockers.innerHTML = '<p class="muted">尚无安全检查结果。</p>';
    }
    for (const blocker of hardBlockers) {
      const item = document.createElement("div");
      item.className = `blocker${blocker.code === "manual_learning_confirmation_required" ? " warning" : ""}`;
      const title = document.createElement("b");
      title.textContent = blocker.message || blocker.code || "已阻止执行";
      item.append(title);
      const blockerHelp = {
        manual_review_required: "完成识别结果复核后才能执行。",
        manual_learning_confirmation_required: "确认学习候选后才能执行。",
        manual_board_unconfirmed: "完成整盘复核后才能执行。",
      }[blocker.code];
      if (blockerHelp) {
        const detail = document.createElement("small");
        detail.textContent = blockerHelp;
        item.append(detail);
      }
      if (["manual_review_required", "manual_learning_confirmation_required", "manual_board_unconfirmed"].includes(blocker.code)) {
        const action = document.createElement("button");
        action.type = "button";
        action.className = "blocker-action";
        action.textContent = "在主棋盘复核";
        action.disabled = S.busy || !S.state;
        action.addEventListener("click", focusQuickReview);
        item.append(action);
      }
      ui.blockers.append(item);
    }
    for (const advisory of advisories) {
      const item = document.createElement("div");
      item.className = "blocker warning advisory";
      const title = document.createElement("b");
      const reviewPieces = advisory.code === "manual_review_required"
        ? reviewNoticePieces(advisory) : [];
      title.textContent = reviewPieces.length
        ? `低置信度棋子：${reviewPieces.map(piece => `#${piece.id}（${piece.location || (piece.cells || []).map(cellName).join("–")}）`).join("、")}`
        : advisory.message || advisory.code || "采集提示";
      const detail = document.createElement("small");
      detail.textContent = reviewPieces.length
        ? "请检查类型、方向和占格；自动执行会先处理其他棋子，不会把它当作整盘门禁。"
        : "不阻止执行";
      item.append(title, detail);
      if (reviewPieces.length) {
        const actions = document.createElement("div");
        actions.className = "blocker-actions";
        for (const piece of reviewPieces) {
          const action = document.createElement("button");
          action.type = "button";
          action.className = "blocker-action";
          action.textContent = `选中并复核 #${piece.id}`;
          action.disabled = S.busy || !S.state;
          action.addEventListener("click", () => focusReviewPiece(piece));
          actions.append(action);
        }
        item.append(actions);
      }
      ui.blockers.append(item);
    }
    $("#diagnosticText").textContent = JSON.stringify({
      scene_state: scene, scene_reason: data.scene_reason,
      metrics: data.metrics, blockers: hardBlockers, advisories,
      warnings: data.warnings,
      board_revision: data.board_revision, cache: data.cache,
      workflow: S.jobView ? {
        action: S.jobView.action,
        phase: S.jobView.phase,
        detail: S.jobView.detail,
        progress: S.jobView.progress,
      } : null,
    }, null, 2);
  }

  function solutionFailure(solution) {
    if (solution?.quick_exit) return null;
    if (!solution || solution.solved) return null;
    const remaining = solution.remaining ?? "?";
    const reason = solution.failure_reason || solution.suspicion?.message;
    if (solution.result_type === "structural_conflict") return {
      tone: "failed", badge: "方向冲突", title: "当前棋盘无法求解",
      detail: reason || "检测到迎头相向的棋子，请复核位置和朝向后重新求解。",
    };
    if (solution.timeout || solution.result_type === "timeout") return {
      tone: "partial", badge: "求解超时", title: "搜索超时，未找到完整解",
      detail: reason || `搜索在 ${Math.max(1, Math.round((solution.timeout_ms || 0) / 1000))} 秒后停止，仍剩 ${remaining} 只。`,
    };
    if ((solution.moves || []).length) return {
      tone: "partial", badge: "部分计划", title: "只找到部分计划",
      detail: reason || `当前步骤走完仍剩 ${remaining} 只，仅供沙盘检查，已禁止自动执行。`,
    };
    return {
      tone: "failed", badge: "未解出", title: "没有找到可用解法",
      detail: reason || `当前棋盘仍剩 ${remaining} 只；请复核识别结果或提高求解上限。`,
    };
  }

  function reflectSolutionOutcome(solution) {
    const failure = solutionFailure(solution);
    if (!failure) return;
    ui.runtimePill.className = `runtime-pill ${failure.tone === "failed" ? "error" : "warning"}`;
    ui.runtimePill.querySelector("b").textContent = failure.badge;
    renderWorkflowSummary();
  }

  function renderPlanFeedback(failure) {
    const host = $("#planFeedback");
    host.replaceChildren();
    host.hidden = !failure;
    host.className = `plan-feedback${failure?.tone === "partial" ? " partial" : ""}`;
    if (!failure) return;
    const copy = document.createElement("div");
    copy.className = "plan-feedback-copy";
    const title = document.createElement("h3"); title.textContent = failure.title;
    const detail = document.createElement("p"); detail.textContent = failure.detail;
    copy.append(title, detail);
    const actions = document.createElement("div");
    actions.className = "plan-feedback-actions";
    const retry = document.createElement("button");
    retry.type = "button"; retry.textContent = "重新求解"; retry.disabled = S.busy;
    retry.addEventListener("click", () => start("solve"));
    actions.append(retry);
    if (S.state) {
      const review = document.createElement("button");
      review.type = "button"; review.textContent = "复核棋盘"; review.disabled = S.busy;
      review.addEventListener("click", focusQuickReview);
      actions.append(review);
    }
    host.append(copy, actions);
  }

  function planJobSummary(job) {
    if (!job) return "";
    const progress = job.progress || {};
    const parts = [job.detail || phaseNames[job.phase] || job.phase].filter(Boolean);
    if (Number.isFinite(progress.steps)) parts.push(`已规划 ${progress.steps} 步`);
    if (Number.isFinite(progress.completed) && Number.isFinite(progress.total) && progress.total > 0) {
      parts.push(`进度 ${Math.min(progress.completed, progress.total)} / ${progress.total}`);
    }
    const liveRemaining = S.liveState?.pieces?.length;
    const remaining = Number.isFinite(liveRemaining) ? liveRemaining : progress.remaining;
    if (Number.isFinite(remaining)) parts.push(`棋盘剩 ${remaining} 只`);
    return [...new Set(parts)].join(" · ");
  }

  function formatProcessDuration(milliseconds) {
    const value = Math.max(0, Number(milliseconds) || 0);
    return value >= 1000 ? `${(value / 1000).toFixed(value >= 10000 ? 0 : 1)}s` : `${Math.round(value)}ms`;
  }

  function processEventDetail(event) {
    if (event.event === "budget-start") {
      const initial = formatProcessDuration(event.initial_ms);
      const maximum = formatProcessDuration(event.max_ms || event.initial_ms);
      return event.elastic ? `初始 ${initial} · 最多 ${maximum}` : `固定 ${initial}`;
    }
    if (event.event === "extension") {
      return `自动补时 ${formatProcessDuration(event.added_ms)} · 已分配 ${formatProcessDuration(event.allocated_ms)} / ${formatProcessDuration(event.max_ms)}`;
    }
    const parts = [];
    if (event.attempt) parts.push(`第 ${event.attempt} 轮`);
    if (event.event === "start" && event.budget_ms) parts.push(`预算 ${formatProcessDuration(event.budget_ms)}`);
    if (event.event === "finish" && Number.isFinite(Number(event.elapsed_ms))) parts.push(`用时 ${formatProcessDuration(event.elapsed_ms)}`);
    if (Number.isFinite(Number(event.steps))) parts.push(`已规划 ${event.steps} 步`);
    if (Number.isFinite(Number(event.remaining))) parts.push(`剩 ${event.remaining} 只`);
    if (Number.isFinite(Number(event.expanded))) parts.push(`扩展 ${Number(event.expanded).toLocaleString()} 节点`);
    if (Number.isFinite(Number(event.restarts))) parts.push(`${event.restarts} 次重启`);
    return parts.join(" · ") || "正在运行";
  }

  function renderSearchProcess() {
    const trace = Array.isArray(S.processTrace) ? S.processTrace : [];
    const solution = S.panelSolution || S.solution;
    const budget = solution?.budget || {};
    ui.processTimeline.replaceChildren();
    if (!trace.length) {
      ui.processSummary.textContent = "等待求解";
      const empty = document.createElement("li");
      empty.className = "process-empty";
      empty.textContent = "生成解法后，这里会显示搜索策略、进度与自动续时。";
      ui.processTimeline.append(empty);
      return;
    }
    const elapsed = S.jobView?.busy ? S.jobView.elapsed_ms : solution?.elapsed_ms ?? trace.at(-1)?.at_ms;
    const extensionEvent = [...trace].reverse().find(item => item.event === "extension");
    const extensions = Number(budget.extensions ?? extensionEvent?.extensions ?? 0);
    ui.processSummary.textContent = `${formatProcessDuration(elapsed)}${extensions ? ` · 续时 ${extensions} 次` : ""}`;
    const visible = trace.length > 20 ? [trace[0], ...trace.slice(-19)] : trace;
    for (const [index, event] of visible.entries()) {
      const item = document.createElement("li");
      const isLast = index === visible.length - 1;
      item.className = `process-event ${event.event || "progress"}`;
      if (isLast && S.jobView?.busy && event.event !== "finish") item.classList.add("active");
      if (event.solved) item.classList.add("solved");
      const marker = document.createElement("i");
      const copy = document.createElement("div");
      const title = document.createElement("b");
      title.textContent = phaseNames[event.phase] || event.phase;
      const detail = document.createElement("small");
      detail.textContent = processEventDetail(event);
      copy.append(title, detail);
      item.append(marker, copy);
      ui.processTimeline.append(item);
    }
  }

  function groupPlanMoves(moves) {
    const groups = [];
    for (const [moveIndex, move] of (moves || []).slice(0, 120).entries()) {
      const exit = move.result === "EXIT";
      const layer = exit && Number.isFinite(Number(move.exit_layer))
        ? Number(move.exit_layer) : null;
      const previous = groups.at(-1);
      if (exit && previous?.exit && previous.layer === layer) {
        previous.entries.push({ move, moveIndex });
      } else {
        groups.push({ exit, layer, entries: [{ move, moveIndex }] });
      }
    }
    return groups;
  }

  function createPlanMoveItem(move, moveIndex, completed, phaseLabels, resultLabels) {
    const item = document.createElement("li");
    const pieceId = String(move.piece);
    item.tabIndex = S.busy ? -1 : 0;
    item.setAttribute("role", "button");
    item.setAttribute("aria-label", `沙盘推演第 ${move.step} 步，棋子 ${pieceId}`);
    item.setAttribute("aria-disabled", S.busy ? "true" : "false");
    item.classList.toggle("selected", moveIndex === S.simulationIndex);
    item.classList.toggle("simulated", S.simulationIndex !== null && moveIndex < S.simulationIndex);
    item.classList.toggle("executed", moveIndex < completed);
    item.classList.toggle("working", S.busy);
    item.classList.toggle("exit-step", move.result === "EXIT");
    const number = document.createElement("span"); number.textContent = move.step;
    const copy = document.createElement("div"); copy.className = "move-copy";
    const title = document.createElement("div"); title.className = "move-title";
    const identity = document.createElement("b");
    identity.textContent = `#${pieceId} · ${speciesNames[move.species] || "羊"}`;
    const action = document.createElement("span");
    action.textContent = `${resultLabels[move.result] || "移动"}${move.result !== "EXIT" && move.distance ? ` ${move.distance} 格` : ""}`;
    title.append(identity, action);
    const meta = document.createElement("small");
    const cell = Array.isArray(move.cell) ? `R${move.cell[0] + 1} C${move.cell[1] + 1}` : "定位已记录";
    const bombChange = (move.bomb_changes || [])[0];
    const bombNote = bombChange ? ` · 炸弹 #${bombChange.piece} ${bombChange.before}→${bombChange.after ?? "离场"}` : "";
    const phaseLabel = move.exit_layer ? `第 ${move.exit_layer} 层直出` : phaseLabels[move.phase] || "规划";
    meta.textContent = `${phaseLabel} · ${cell}${bombNote}`;
    copy.append(title, meta);
    const direction = document.createElement("em"); direction.className = "move-direction";
    if (move.bomb_changes?.length || move.wolf_risk) direction.classList.add("risk");
    direction.textContent = move.bomb_changes?.length ? "炸弹\n核验" : move.wolf_risk ? "狼道\n核验" : facingNames[move.direction] || move.direction;
    item.append(number, copy, direction);
    item.addEventListener("click", () => enterSimulation(moveIndex));
    item.addEventListener("keydown", event => {
      if (S.busy || (event.key !== "Enter" && event.key !== " ")) return;
      event.preventDefault(); enterSimulation(moveIndex);
    });
    return item;
  }

  function renderPlan() {
    const job = S.jobView;
    const active = !!job?.busy && ["solve", "quick", "step", "auto"].includes(job.action);
    const executing = active && ["executing", "verifying", "pausing"].includes(job.phase);
    const solution = S.panelSolution || S.solution;
    const panel = $("#planPanel"), badge = $("#planBadge"), summary = $("#planSummary");
    ui.moveList.replaceChildren();
    panel.classList.remove("failed", "partial", "running");
    badge.className = "plan-badge";
    if (!solution) {
      if (active) {
        panel.classList.add("running");
        badge.classList.add("running");
        badge.textContent = phaseNames[job.phase] || "处理中";
        summary.textContent = planJobSummary(job) || "正在准备解题信息。";
      } else {
        badge.textContent = "未求解";
        summary.textContent = "分析完成后再求解。";
      }
      renderPlanFeedback(null);
      return;
    }
    const failure = solutionFailure(solution);
    const ready = solutionReady(solution);
    if (executing) {
      panel.classList.add("running");
      badge.textContent = job.phase === "verifying" ? "核对中" : job.phase === "pausing" ? "暂停中" : "执行中";
      badge.classList.add("running");
    } else if (active) {
      panel.classList.add("running");
      badge.textContent = phaseNames[job.phase] || "方案更新中";
      badge.classList.add("running");
    } else if (solution.quick_exit) {
      badge.textContent = solution.solved ? "已清空" : (solution.total ? "快速清理" : "无直出");
      badge.classList.add(solution.solved ? "complete" : "partial");
      if (!solution.solved) panel.classList.add("partial");
    } else if (solution.solved) {
      badge.textContent = "完整解"; badge.classList.add("complete");
    } else if (ready) {
      badge.textContent = "安全前缀"; badge.classList.add("partial"); panel.classList.add("partial");
    } else {
      badge.textContent = failure.badge; badge.classList.add(failure.tone); panel.classList.add(failure.tone);
    }
    const bombSummary = solution.bomb_count ? ` · ${solution.bomb_count} 只炸弹羊逐步核验` : "";
    const planSummary = solution.quick_exit
      ? `${solution.exit_layers || 0} / ${solution.max_exit_layers || 3} 层 · ${solution.total || 0} 只离场 · 剩 ${solution.remaining ?? "?"} 只`
      : `${solution.total || 0} 步 · ${solution.solved ? "可清盘" : `剩 ${solution.remaining ?? "?"} 只`}${bombSummary}`;
    summary.textContent = active ? `${planJobSummary(job)} · ${planSummary}` : planSummary;
    renderPlanFeedback(active || ready || solution.quick_exit ? null : failure);
    const phaseLabels = { coarse: "直出", refine: "规划", cli: "规划", plan: "规划" };
    const resultLabels = { EXIT: "离场", MOVE: "滑动", STEP: "前进", BOUNCE: "借位" };
    const completed = executing && Number.isFinite(job.progress?.completed)
      ? Math.max(0, job.progress.completed - Math.max(0, Number(job.plan_completed_base) || 0)) : 0;
    for (const group of groupPlanMoves(solution.moves)) {
      if (!group.exit || group.entries.length === 1) {
        const entry = group.entries[0];
        ui.moveList.append(createPlanMoveItem(
          entry.move, entry.moveIndex, completed, phaseLabels, resultLabels));
        continue;
      }
      const wrapper = document.createElement("li");
      wrapper.className = "move-group exit-group";
      const details = document.createElement("details");
      details.open = group.entries.some(entry => entry.moveIndex === S.simulationIndex);
      const groupSummary = document.createElement("summary");
      const icon = document.createElement("span"); icon.textContent = "↗";
      const title = document.createElement("b");
      title.textContent = group.layer ? `第 ${group.layer} 层离场` : "连续离场";
      const done = group.entries.filter(entry => entry.moveIndex < completed).length;
      const meta = document.createElement("small");
      meta.textContent = `${group.entries.length} 步${done ? ` · 已执行 ${done}` : ""}`;
      groupSummary.append(icon, title, meta);
      const nested = document.createElement("ol");
      nested.className = "move-group-list";
      for (const entry of group.entries) nested.append(createPlanMoveItem(
        entry.move, entry.moveIndex, completed, phaseLabels, resultLabels));
      details.append(groupSummary, nested);
      wrapper.append(details);
      ui.moveList.append(wrapper);
    }
  }

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
  if (window.pywebview?.api) ready();
})();
