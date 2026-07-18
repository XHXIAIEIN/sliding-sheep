// 共享状态、运行设置与桥接工具
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
