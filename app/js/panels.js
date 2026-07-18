// 安全报告、算法过程与方案列表渲染
"use strict";

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
