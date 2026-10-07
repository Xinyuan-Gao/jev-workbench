/* JEV Lab workbench: browser-only UI. Secrets stay in the backend environment. */
(function () {
  "use strict";

  const API_BASE = (window.JEV_API_BASE || "/api").replace(/\/$/, "");
  const POLL_MS = 900;
  const LAST_RUN_KEY = "jev-workbench:last-run";
  let experiments = [
    { id: "intent-classification", code: "01 / CLASSIFICATION", title: "中文意图分类", meta: "216 samples · 2 modes", sampleCount: 216, description: "把日常客服消息分到 billing、technical 或 complaint，比较模拟与 JEV 判断。", dataset: "intent-classification" },
    { id: "rag-relevance", code: "02 / RAG", title: "RAG 片段相关性", meta: "200 samples · 2 modes", sampleCount: 200, description: "判断检索到的知识片段是否能回答当前问题，观察相关性与置信度。", dataset: "rag-relevance" },
    { id: "agent-routing", code: "03 / AGENT", title: "Agent 路由选择", meta: "216 samples · 2 modes", sampleCount: 216, description: "为一条任务选择最合适的处理路径，比较搜索、代码和写作路由。", dataset: "agent-routing" },
  ];

  const state = {
    experiment: experiments[0],
    runId: null,
    runState: "idle",
    pollTimer: null,
    eventCursor: 0,
    localTimer: null,
    localMode: false,
    events: [],
    results: { models: [], samples: [] },
  };

  const $ = (id) => document.getElementById(id);
  const els = {
    experimentList: $("experiment-list"), title: $("experiment-title"), kicker: $("experiment-kicker"), description: $("experiment-description"),
    start: $("start-run"), stop: $("stop-run"), reset: $("reset-view"), status: $("run-status"), runId: $("run-id"), progress: $("progress-value"), progressPercent: $("progress-percent"),
    sampleProgress: $("sample-progress"), throughput: $("throughput"), fastest: $("fastest-model"), fastestLatency: $("fastest-latency"), totalCost: $("total-cost"),
    eventFeed: $("event-feed"), liveChip: $("live-chip"), resultCount: $("result-count"), resultTable: $("result-table"), sampleTable: $("sample-table"),
    mode: $("mode-select"), dataset: $("dataset-select"), concurrency: $("concurrency"), concurrencyValue: $("concurrency-value"), filter: $("sample-filter"),
    exportJson: $("export-json"), exportCsv: $("export-csv"), toast: $("toast"), connectionDot: $("connection-dot"), connectionLabel: $("connection-label"), chart: $("model-chart"), chartFallback: $("chart-fallback"), clock: $("clock"),
  };

  function nowTime() { return new Date().toLocaleTimeString("zh-CN", { hour12: false }); }
  function esc(value) { return String(value ?? "").replace(/[&<>\"]/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[char])); }
  function toast(message) { els.toast.textContent = message; els.toast.classList.add("show"); clearTimeout(toast.timer); toast.timer = setTimeout(() => els.toast.classList.remove("show"), 3000); }

  async function request(path, options = {}) {
    const headers = { "Content-Type": "application/json", ...(options.headers || {}) };
    const response = await fetch(`${API_BASE}${path}`, { ...options, headers });
    if (!response.ok) throw new Error(`API ${response.status}`);
    const type = response.headers.get("content-type") || "";
    return type.includes("json") ? response.json() : response.text();
  }

  async function loadExperiments() {
    const remote = await request("/experiments");
    if (!Array.isArray(remote) || !remote.length) return;
    experiments = remote.map((item, index) => ({
      id: item.id,
      code: `${String(index + 1).padStart(2, "0")} / ${item.task ? String(item.task).toUpperCase() : "EXPERIMENT"}`,
      title: item.name || item.title || item.id,
      meta: `${Array.isArray(item.records) ? item.records.length : (item.sample_count || 0)} samples · ${Array.isArray(item.labels) ? item.labels.length : 0} labels`,
      sampleCount: Array.isArray(item.records) ? item.records.length : Number(item.sample_count || 0),
      description: item.description || "本地实验数据集。",
      dataset: item.id,
    }));
    state.experiment = experiments.find((item) => item.id === state.experiment.id) || experiments[0];
    els.title.textContent = state.experiment.title;
    els.kicker.textContent = `EXPERIMENT ${state.experiment.code}`;
    els.description.textContent = state.experiment.description;
    els.dataset.innerHTML = experiments.map((item) => `<option value="${esc(item.dataset)}">${esc(item.title)} · ${esc(item.meta.split(" · ")[0])}</option>`).join("");
    els.dataset.value = state.experiment.dataset;
    renderExperiments();
  }

  function setConnection(kind, label) { els.connectionDot.className = `connection-dot ${kind || ""}`; els.connectionLabel.textContent = label; }

  function renderExperiments() {
    els.experimentList.innerHTML = experiments.map((item) => `<button class="experiment-item ${item.id === state.experiment.id ? "active" : ""}" data-experiment="${esc(item.id)}"><span class="item-code">${esc(item.code)}</span><div class="item-title">${esc(item.title)}</div><div class="item-meta">${esc(item.meta)}</div><span class="item-state ready">ready</span></button>`).join("");
    els.experimentList.querySelectorAll("[data-experiment]").forEach((button) => button.addEventListener("click", () => selectExperiment(button.dataset.experiment)));
  }

  function selectExperiment(id) {
    if (state.runState === "running") { toast("当前运行中，先停止运行再切换实验"); return; }
    state.experiment = experiments.find((item) => item.id === id) || experiments[0];
    els.title.textContent = state.experiment.title; els.kicker.textContent = `EXPERIMENT ${state.experiment.code}`; els.description.textContent = state.experiment.description; els.dataset.value = state.experiment.dataset;
    renderExperiments(); resetView(false, true);
  }

  function appendEvents(items) {
    if (!Array.isArray(items) || !items.length) return;
    state.events.push(...items);
    els.eventFeed.innerHTML = state.events.slice(-100).map((item) => `<div class="event-line ${item.level === "error" ? "error" : item.level === "warn" ? "warn" : ""}"><span class="event-time">${esc(item.time || nowTime())}</span><span class="event-bullet"></span><div class="event-detail"><strong>${esc(item.message || item.type || "event")}</strong>${item.detail ? `<span>${esc(item.detail)}</span>` : ""}</div></div>`).join("");
    els.eventFeed.scrollTop = els.eventFeed.scrollHeight;
  }

  function addLocalEvent(message, detail, level) { appendEvents([{ time: nowTime(), message, detail, level }]); }

  function setRunStatus(status, stateLabel) {
    state.runState = status; els.status.textContent = stateLabel || ({ idle: "待运行", running: "运行中", complete: "已完成", stopped: "已停止", error: "异常" }[status] || status);
    els.liveChip.textContent = status === "running" ? "LIVE" : status === "complete" ? "DONE" : status === "error" ? "ERROR" : "IDLE"; els.liveChip.className = `live-chip ${status === "running" ? "active" : status === "complete" ? "done" : ""}`;
    els.start.disabled = status === "running"; els.start.innerHTML = status === "running" ? "运行中 <span aria-hidden=\"true\">…</span>" : status === "complete" || status === "stopped" ? "重新运行 <span aria-hidden=\"true\">→</span>" : "开始实验 <span aria-hidden=\"true\">→</span>";
    els.stop.disabled = status !== "running"; els.exportJson.disabled = !state.results.models.length; els.exportCsv.disabled = !state.results.models.length;
  }

  function updateProgress(done, total, throughput) {
    const amount = Math.max(0, Math.min(100, total ? done / total * 100 : 0)); els.progress.style.width = `${amount}%`; els.progressPercent.textContent = `${Math.round(amount)}%`; els.sampleProgress.innerHTML = `${done} <small>/ ${total || "—"}</small>`; els.throughput.textContent = throughput || (done ? "处理中" : "等待开始");
  }

  function rememberRun(run) {
    try { localStorage.setItem(LAST_RUN_KEY, JSON.stringify({ runId: run.run_id || run.id, experimentId: run.experiment_id, mode: run.mode })); } catch (_) { /* private browsing may disable storage */ }
  }

  function forgetRun() {
    try { localStorage.removeItem(LAST_RUN_KEY); } catch (_) { /* ignore storage errors */ }
  }

  async function restoreLatestRun() {
    const summary = await request("/state");
    const runs = Array.isArray(summary.runs) ? summary.runs : [];
    if (!runs.length) return;
    let saved = null;
    try { saved = JSON.parse(localStorage.getItem(LAST_RUN_KEY) || "null"); } catch (_) { saved = null; }
    const selected = (saved && runs.find((run) => run.run_id === saved.runId)) || runs[runs.length - 1];
    if (!selected) return;
    const experiment = experiments.find((item) => item.id === selected.experiment_id);
    if (experiment) {
      state.experiment = experiment;
      els.title.textContent = experiment.title;
      els.kicker.textContent = `EXPERIMENT ${experiment.code}`;
      els.description.textContent = experiment.description;
      els.dataset.value = experiment.dataset;
      renderExperiments();
    }
    state.runId = selected.run_id;
    state.eventCursor = 0;
    els.runId.textContent = `run ${selected.run_id}`;
    if (["simulation", "xgboost", "lightgbm", "jev", "gemini"].includes(selected.mode)) {
      els.mode.value = selected.mode;
      document.querySelectorAll("#model-toggles input").forEach((input) => { input.checked = input.value === selected.mode; });
    }
    rememberRun(selected);
    setConnection("connected", "已恢复后端运行");
    setRunStatus(["queued", "running"].includes(selected.status) ? "running" : selected.status === "stopped" ? "stopped" : ["completed", "partial"].includes(selected.status) ? "complete" : "error");
    beginPolling();
  }

  function renderResults(payload) {
    const data = payload || {};
    const rawResults = Array.isArray(data.results) ? data.results : [];
    const samples = Array.isArray(data.samples) ? data.samples : rawResults.map((record) => {
      const input = record.input || {};
      const gold = input.label || input.gold_label || input.gold;
      const prediction = record.prediction || record.label;
      return { ...record, sample_id: record.id, text: input.text || input.passage || input.task || input.query || JSON.stringify(input), gold_label: gold, correct: gold ? gold === prediction : undefined };
    });
    let models = Array.isArray(data.models) ? data.models : [];
    if (!models.length && rawResults.length) {
      const valid = rawResults.filter((record) => record.input && record.input.label);
      const accuracy = valid.length ? valid.filter((record) => record.prediction === record.input.label).length / valid.length : null;
      const latencies = rawResults.map((record) => Number(record.latency_ms)).filter(Number.isFinite).sort((a, b) => a - b);
      const modelName = data.mode === "jev" ? "JEV" : data.mode === "gemini" ? "Gemini 3.8 Flash" : data.mode === "xgboost" ? "TF-IDF + XGBoost" : data.mode === "lightgbm" ? "TF-IDF + LightGBM" : "规则模拟";
      models = [{ model: modelName, accuracy, macro_f1: null, p95_latency_ms: latencies[Math.min(latencies.length - 1, Math.floor(latencies.length * .95))] || 0, estimated_cost_usd: null }];
    }
    state.results = { models, samples };
    models = state.results.models;
    els.resultCount.textContent = `${models.length} model${models.length === 1 ? "" : "s"}`;
    els.resultTable.innerHTML = models.length ? models.map((row) => `<tr><td class="model-name">${esc(row.model || row.name)}</td><td class="num">${formatPct(row.accuracy)}</td><td class="num">${formatPct(row.macro_f1 ?? row.macroF1)}</td><td class="num">${formatNum(row.p95_latency_ms ?? row.p95)}</td><td class="num">${formatCost(row.estimated_cost_usd ?? row.cost_usd ?? row.cost)}</td></tr>`).join("") : `<tr><td colspan="5" class="empty-cell">暂无结果</td></tr>`;
    const fastest = models.filter((row) => Number(row.p95_latency_ms ?? row.p95) > 0).sort((a, b) => Number(a.p95_latency_ms ?? a.p95) - Number(b.p95_latency_ms ?? b.p95))[0]; els.fastest.textContent = fastest ? (fastest.model || fastest.name) : "—"; els.fastestLatency.textContent = fastest ? `p95 ${formatNum(fastest.p95_latency_ms ?? fastest.p95)} ms` : "—";
    const knownCosts = models.map((row) => row.estimated_cost_usd ?? row.cost_usd ?? row.cost).filter((value) => value !== null && value !== undefined); els.totalCost.textContent = knownCosts.length ? `$${knownCosts.reduce((sum, value) => sum + Number(value), 0).toFixed(4)}` : "—";
    renderChart(models); renderSamples();
  }

  function formatPct(value) { if (value === null || value === undefined) return "—"; const num = Number(value); return Number.isFinite(num) ? `${(num <= 1 ? num * 100 : num).toFixed(1)}%` : "—"; }
  function formatNum(value) { if (value === null || value === undefined) return "—"; const num = Number(value); return Number.isFinite(num) ? num.toFixed(0) : "—"; }
  function formatCost(value) { if (value === null || value === undefined) return "—"; const num = Number(value); return Number.isFinite(num) ? `$${num.toFixed(4)}` : "—"; }

  function renderSamples() {
    const filter = els.filter.value; const samples = state.results.samples.filter((sample) => filter === "errors" ? sample.correct === false : filter === "boundary" ? Number(sample.confidence) >= .3 && Number(sample.confidence) <= .7 : true);
    els.sampleTable.innerHTML = samples.length ? samples.slice(0, 80).map((sample) => `<tr><td>${esc(sample.sample_id || sample.id || "sample")}</td><td title="${esc(sample.text || sample.input)}">${esc(sample.text || sample.input || "—")}</td><td>${esc(sample.gold_label || sample.gold || "—")}</td><td class="${sample.correct === false ? "sample-error" : "sample-ok"}">${esc(sample.prediction || "—")}</td><td class="num">${formatPct(sample.confidence)}</td><td class="num">${formatNum(sample.latency_ms)} ms</td><td><button class="detail-button" data-sample="${esc(sample.sample_id || sample.id || "sample")}">详情</button></td></tr>`).join("") : `<tr><td colspan="7" class="empty-cell">暂无匹配样本</td></tr>`;
    els.sampleTable.querySelectorAll("[data-sample]").forEach((button) => button.addEventListener("click", () => { const item = state.results.samples.find((sample) => String(sample.sample_id || sample.id) === button.dataset.sample); if (item) toast(`${item.sample_id || item.id}: ${item.prediction || "暂无预测"} · ${formatNum(item.latency_ms)} ms`); }));
  }

  function renderChart(models) {
    const canvas = els.chart; const ctx = canvas.getContext && canvas.getContext("2d"); if (!ctx || !models.length) { els.chartFallback.classList.remove("hidden"); return; } els.chartFallback.classList.add("hidden");
    const width = canvas.clientWidth || 430; const height = 145; const dpr = window.devicePixelRatio || 1; canvas.width = width * dpr; canvas.height = height * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, width, height); const max = Math.max(100, ...models.map((item) => Number(item.accuracy || 0) * (Number(item.accuracy || 0) <= 1 ? 100 : 1))); const gap = 9; const barWidth = Math.max(18, (width - gap * (models.length + 1)) / models.length);
    models.forEach((item, index) => { const score = Number(item.accuracy || 0) * (Number(item.accuracy || 0) <= 1 ? 100 : 1); const barHeight = Math.max(4, score / max * 102); const x = gap + index * (barWidth + gap); const y = 111 - barHeight; ctx.fillStyle = index === 0 ? "#12766c" : index === 1 ? "#72afa0" : index === 2 ? "#cb6b36" : "#a7b3ae"; ctx.fillRect(x, y, barWidth, barHeight); ctx.fillStyle = "#6e7b76"; ctx.font = "9px IBM Plex Mono, monospace"; ctx.textAlign = "center"; ctx.fillText(`${score.toFixed(0)}%`, x + barWidth / 2, y - 5); ctx.save(); ctx.translate(x + barWidth / 2, 126); ctx.rotate(-.25); ctx.fillText(String(item.model || item.name || "model").slice(0, 13), 0, 0); ctx.restore(); }); ctx.strokeStyle = "#d9e1dc"; ctx.beginPath(); ctx.moveTo(0, 111.5); ctx.lineTo(width, 111.5); ctx.stroke();
  }

  function collectConfig() { return { experiment_id: state.experiment.id, dataset: els.dataset.value, mode: els.mode.value, models: [...document.querySelectorAll("#model-toggles input:checked")].map((input) => input.value), concurrency: Number(els.concurrency.value), seed: 42 }; }

  async function startRun() {
    if (state.runState === "running") return; resetView(false, true); const config = collectConfig(); setRunStatus("running", "准备中"); addLocalEvent("创建运行", `${state.experiment.title} · ${config.mode}`); els.runId.textContent = "正在创建 run…";
    try {
      const payload = await request("/runs", { method: "POST", body: JSON.stringify(config) }); state.runId = payload.run_id || payload.id || payload.runId; if (!state.runId) throw new Error("missing run id"); state.localMode = false; rememberRun(payload); els.runId.textContent = `run ${state.runId}`; setConnection("connected", "已连接后端"); addLocalEvent("运行已创建", `run ${state.runId}`); beginPolling();
    } catch (error) {
      if (config.mode === "simulation") { state.localMode = true; state.runId = `local-${Date.now().toString(36)}`; els.runId.textContent = `run ${state.runId}`; setConnection("error", "后端不可用 · 本地模拟"); addLocalEvent("切换本地模拟", "后端未连接，使用内置数据演示"); beginLocalSimulation(config); }
      else { setRunStatus("error", "连接失败"); setConnection("error", "后端不可用"); addLocalEvent("无法创建运行", error.message, "error"); toast("JEV 模式需要先启动本地后端"); }
    }
  }

  function beginPolling() { clearInterval(state.pollTimer); state.pollTimer = setInterval(pollRun, POLL_MS); pollRun(); }
  async function pollRun() {
    if (!state.runId || state.localMode) return; try { const [statePayload, eventsPayload, resultPayload] = await Promise.all([request(`/runs/${encodeURIComponent(state.runId)}/state`), request(`/runs/${encodeURIComponent(state.runId)}/events?after=${state.eventCursor}`), request(`/runs/${encodeURIComponent(state.runId)}/results`)]); const incomingEvents = (eventsPayload.events || eventsPayload || []).filter((item) => item.seq === undefined || Number(item.seq) >= state.eventCursor); appendEvents(incomingEvents.map((item) => ({ ...item, time: item.time || (item.ts ? new Date(item.ts * 1000).toLocaleTimeString("zh-CN", { hour12: false }) : undefined), message: item.message || item.type, detail: item.detail || item.record_id || item.run_id || (item.record && item.record.id) }))); const maxSeq = incomingEvents.reduce((max, item) => Math.max(max, Number(item.seq ?? -1)), state.eventCursor - 1); state.eventCursor = maxSeq + 1; renderResults({ ...resultPayload, mode: statePayload.mode }); const done = Number(statePayload.count ?? statePayload.completed ?? statePayload.done ?? 0); const total = Number(statePayload.total ?? statePayload.total_samples ?? 0); updateProgress(done, total, statePayload.throughput ? `${statePayload.throughput} samples/s` : undefined); const status = statePayload.status || "running"; if (["completed", "complete", "partial", "stopped", "failed", "error"].includes(status)) { clearInterval(state.pollTimer); const finalStatus = status === "failed" || status === "error" ? "error" : status === "stopped" ? "stopped" : "complete"; setRunStatus(finalStatus, status === "partial" ? "已结束，含无效输出" : undefined); if (status === "partial") els.throughput.textContent = `运行结束 · ${total - done} 条无效输出`; addLocalEvent(finalStatus === "stopped" ? "运行已停止" : finalStatus === "error" ? "运行失败" : "运行已完成", `${done}/${total || done} samples`); } } catch (error) { setConnection("error", "后端连接异常"); addLocalEvent("轮询失败", error.message, "warn"); }
  }

  function beginLocalSimulation(config) { clearInterval(state.localTimer); const total = Number(state.experiment?.sampleCount) || 200; let done = 0; const modelNames = config.models.length ? config.models : ["rules", "jev"]; const start = performance.now(); state.localTimer = setInterval(() => { if (state.runState !== "running") return; done += Math.max(1, Math.round(total / 13)); const current = Math.min(done, total); updateProgress(current, total, `${(current / Math.max(.5, (performance.now() - start) / 1000)).toFixed(1)} samples/s`); if (current === 1) addLocalEvent("数据集已加载", `${total} samples · seed 42`); if (current % 18 < 8) addLocalEvent("批次完成", `batch ${Math.ceil(current / 8)} · ${current}/${total}`); if (current >= total) { clearInterval(state.localTimer); state.results = { models: modelNames.map((model, index) => ({ model, accuracy: .82 + index * .035, macro_f1: .8 + index * .04, p95_latency_ms: 18 + index * 112, estimated_cost_usd: index === 1 ? .0006 : 0 })), samples: Array.from({ length: 8 }, (_, index) => ({ sample_id: `sample-${String(index + 1).padStart(3, "0")}`, text: ["支付失败，已经影响销售", "如何修改登录邮箱？", "服务恢复了吗？", "我要申请退款"][index % 4], gold_label: ["billing", "account", "technical", "refund"][index % 4], prediction: ["billing", "account", "technical", index === 3 ? "billing" : "refund"][index % 4], confidence: .55 + (index % 4) * .1, latency_ms: 17 + index * 8, correct: index !== 3 })) }; renderResults(state.results); setRunStatus("complete"); addLocalEvent("模拟运行完成", `${total} samples · ${((performance.now() - start) / 1000).toFixed(1)} s`); } }, 220); }

  async function stopRun() { if (state.runState !== "running") return; clearInterval(state.pollTimer); clearInterval(state.localTimer); if (!state.localMode && state.runId) { try { await request(`/runs/${encodeURIComponent(state.runId)}/stop`, { method: "POST" }); } catch (error) { addLocalEvent("停止请求失败", error.message, "warn"); } } setRunStatus("stopped"); addLocalEvent("运行已停止", "可以调整配置后重新运行"); }
  function resetView(showToast = true, clearSaved = false) { clearInterval(state.pollTimer); clearInterval(state.localTimer); if (clearSaved) forgetRun(); state.runId = null; state.eventCursor = 0; state.localMode = false; state.events = []; state.results = { models: [], samples: [] }; els.runId.textContent = "尚未创建 run"; els.eventFeed.innerHTML = `<div class="event-empty">开始运行后，事件会显示在这里</div>`; updateProgress(0, Number(state.experiment?.sampleCount) || 0); renderResults(state.results); setRunStatus("idle"); if (showToast) toast("视图已重置"); }

  function download(name, content, type) { const blob = new Blob([content], { type }); const anchor = document.createElement("a"); anchor.href = URL.createObjectURL(blob); anchor.download = name; anchor.click(); setTimeout(() => URL.revokeObjectURL(anchor.href), 300); }
  function exportJson() { download(`jev-${state.runId || "run"}.json`, JSON.stringify({ run_id: state.runId, experiment: state.experiment.id, results: state.results }, null, 2), "application/json"); }
  function exportCsv() { const rows = [["model", "accuracy", "macro_f1", "p95_latency_ms", "estimated_cost_usd"], ...state.results.models.map((row) => [row.model || row.name, row.accuracy ?? "", row.macro_f1 ?? row.macroF1 ?? "", row.p95_latency_ms ?? row.p95 ?? "", row.estimated_cost_usd ?? row.cost_usd ?? row.cost ?? ""])]; download(`jev-${state.runId || "run"}.csv`, rows.map((row) => row.map((cell) => `"${String(cell).replace(/"/g, '""')}"`).join(",")).join("\n"), "text/csv;charset=utf-8"); }

  async function checkConnection() { try { await loadExperiments(); setConnection("connected", "后端已连接"); await restoreLatestRun(); } catch (_) { setConnection("", "等待连接后端"); } }
  function bind() { els.start.addEventListener("click", startRun); els.stop.addEventListener("click", stopRun); els.reset.addEventListener("click", () => resetView(true, true)); $("refresh-experiments").addEventListener("click", () => { renderExperiments(); toast("实验目录已刷新"); }); els.concurrency.addEventListener("input", () => { els.concurrencyValue.textContent = els.concurrency.value; }); els.filter.addEventListener("change", renderSamples); els.exportJson.addEventListener("click", exportJson); els.exportCsv.addEventListener("click", exportCsv); document.querySelectorAll("#model-toggles input").forEach((input) => input.addEventListener("change", () => { if (input.checked) { document.querySelectorAll("#model-toggles input").forEach((other) => { if (other !== input) other.checked = false; }); if (["xgboost", "lightgbm", "jev", "gemini"].includes(input.value)) els.mode.value = input.value; } })); els.mode.addEventListener("change", () => { document.querySelectorAll("#model-toggles input").forEach((input) => { input.checked = input.value === els.mode.value; }); }); setInterval(() => { els.clock.textContent = nowTime(); }, 1000); }
  function init() { renderExperiments(); els.eventFeed.innerHTML = `<div class="event-empty">开始运行后，事件会显示在这里</div>`; bind(); resetView(false, false); checkConnection(); }
  init();
})();
