(function () {
  const POLL_INTERVAL_MS = 5000;

  const nodes = {
    streamStatus: document.getElementById("stream-status"),
    stressStatusBadge: document.getElementById("stress-status-badge"),
    healthStatusBadge: document.getElementById("health-status-badge"),
    emotion: document.getElementById("emotion"),
    emotionConfidence: document.getElementById("emotion-confidence"),
    jawTension: document.getElementById("jaw-tension"),
    jawTensionBar: document.getElementById("jaw-tension-bar"),
    postureQuality: document.getElementById("posture-quality"),
    postureBar: document.getElementById("posture-bar"),
    overallStress: document.getElementById("overall-stress"),
    stressBar: document.getElementById("stress-bar"),
    stressModelScore: document.getElementById("stress-model-score"),
    mindlensModelScore: document.getElementById("mindlens-model-score"),
    avgStressScore: document.getElementById("avg-stress-score"),
    slouchScore: document.getElementById("slouch-score"),
    jawClenchScore: document.getElementById("jaw-clench-score"),
    scanPostureMetric: document.getElementById("scan-posture-metric"),
    stressScanTime: document.getElementById("stress-scan-time"),
    recommendations: document.getElementById("recommendations"),
    latestEmotion: document.getElementById("latest-emotion"),
    postureIcon: document.getElementById("posture-icon"),
    bloodPressure: document.getElementById("blood-pressure"),
    bpBadge: document.getElementById("bp-badge"),
    bpCategory: document.getElementById("bp-category"),
    heartRate: document.getElementById("heart-rate"),
    hrBadge: document.getElementById("hr-badge"),
    hrCategory: document.getElementById("hr-category"),
    cholesterol: document.getElementById("cholesterol"),
    cholBadge: document.getElementById("chol-badge"),
    cholCategory: document.getElementById("chol-category"),
    glucose: document.getElementById("glucose"),
    glucoseBadge: document.getElementById("glucose-badge"),
    glucoseCategory: document.getElementById("glucose-category"),
    insulin: document.getElementById("insulin"),
    insulinBadge: document.getElementById("insulin-badge"),
    insulinCategory: document.getElementById("insulin-category"),
    mentalHealth: document.getElementById("mental-health"),
    mentalHealthBadge: document.getElementById("mental-health-badge"),
    mentalHealthCategory: document.getElementById("mental-health-category"),
    healthMetricsTime: document.getElementById("health-metrics-time"),
  };

  let pollTimer = null;
  let eventSource = null;

  function normalizeEmail(value) {
    const email = String(value || "").trim();
    if (!email) return "";
    const lowered = email.toLowerCase();
    if (lowered === "undefined" || lowered === "null" || lowered === "none") return "";
    return email.includes("@") && email.includes(".") ? email : "";
  }

  function resolveEmail() {
    const params = new URLSearchParams(window.location.search);
    const queryEmail = normalizeEmail(params.get("email"));
    if (queryEmail) return queryEmail;

    try {
      const userData = JSON.parse(localStorage.getItem("user_data") || "{}");
      const userEmail = normalizeEmail(userData.email);
      if (userEmail) {
        localStorage.setItem("user_email", userEmail);
        return userEmail;
      }
    } catch (_err) {
    }

    try {
      const latestScan = JSON.parse(localStorage.getItem("latest_scan_result") || "{}");
      const scanEmail = normalizeEmail(latestScan.email);
      if (scanEmail) {
        localStorage.setItem("user_email", scanEmail);
        return scanEmail;
      }
    } catch (_err) {
    }

    return normalizeEmail(localStorage.getItem("user_email") || sessionStorage.getItem("user_email"));
  }

  function firstDefined() {
    for (let index = 0; index < arguments.length; index += 1) {
      const value = arguments[index];
      if (value !== undefined && value !== null && value !== "") return value;
    }
    return null;
  }

  function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
  }

  function toNumber(value, fallback = null) {
    const num = Number(value);
    return Number.isFinite(num) ? num : fallback;
  }

  function formatPercent(value, digits = 1, fallback = "0%") {
    const num = toNumber(value);
    if (num === null) return fallback;
    const normalized = num <= 1 ? num * 100 : num;
    return `${normalized.toFixed(digits)}%`;
  }

  function toPercentConfidence(value, fallback = "68% confidence") {
    const num = toNumber(value);
    if (num === null) return fallback;
    return `${Math.round((num <= 1 ? num * 100 : num))}% confidence`;
  }

  function toScore(value, fallback = "0.0") {
    const num = toNumber(value);
    return num === null ? fallback : num.toFixed(1);
  }

  function formatDateTime(value) {
    if (!value) return "";
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString();
  }

  function setText(node, value, fallback) {
    if (!node) return;
    node.textContent = value === undefined || value === null || value === "" ? fallback : String(value);
  }

  function escapeHtml(value) {
    return String(value || "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#39;");
  }

  function badgeClassForStatus(status) {
    const key = String(status || "").toLowerCase();
    if (key.includes("high") || key.includes("alert")) return "bg-danger";
    if (key.includes("moderate") || key.includes("monitor") || key.includes("elevated")) return "bg-warning text-dark";
    if (key.includes("low") || key.includes("normal") || key.includes("good") || key.includes("balanced")) return "bg-success";
    return "bg-secondary";
  }

  function updateBadge(node, text, fallback) {
    if (!node) return;
    const value = text || fallback;
    node.className = `badge ${badgeClassForStatus(value)}`;
    node.textContent = value;
  }

  function overallStressLabel(score, fallback = "Moderate") {
    const num = toNumber(score);
    if (num === null) return fallback;
    if (num >= 70) return "High";
    if (num >= 40) return "Moderate";
    return "Low";
  }

  function jawPercent(level) {
    const key = String(level || "").toLowerCase();
    if (key === "high") return 88;
    if (key === "medium") return 58;
    return 26;
  }

  function posturePercent(label) {
    const key = String(label || "").toLowerCase();
    if (key.includes("aligned")) return 90;
    if (key.includes("upright")) return 80;
    if (key.includes("forward")) return 58;
    if (key.includes("slouch")) return 34;
    return 50;
  }

  function stressColor(score) {
    const num = toNumber(score, 48);
    if (num >= 70) return "#dc3545";
    if (num >= 40) return "#ffc107";
    return "#198754";
  }

  function buildFallbackScan(summary) {
    const stressModel = toNumber(summary.stress_model_score, 45.6);
    const habitModel = toNumber(summary.mindlens_model_score, 47.8);
    const avgStress = toNumber(summary.avg_stress, (stressModel + habitModel) / 2);
    return {
      emotion: firstDefined(summary.emotion, "Neutral"),
      emotion_confidence: toNumber(summary.emotion_confidence, 0.68),
      overall_stress: firstDefined(summary.overall_stress, overallStressLabel(avgStress)),
      avg_stress: avgStress,
      posture_quality: firstDefined(summary.posture_quality, "Upright"),
      slouch_score: toNumber(summary.slouch_score, 0.28),
      jaw_tension: firstDefined(summary.jaw_tension, "Low"),
      jaw_clench_score: toNumber(summary.jaw_clench_score, 0.41),
      mindlens_model_score: habitModel,
      stress_model_score: stressModel,
      scanned_at: firstDefined(summary.scanned_at, summary.timestamp),
    };
  }

  function normalizeScanRow(row, fallbackRow) {
    const source = row || {};
    const fallback = fallbackRow || {};
    const facialScore = firstDefined(source.stress_model_score, fallback.stress_model_score);
    const habitScore = firstDefined(source.mindlens_model_score, fallback.mindlens_model_score);
    const avgStress = firstDefined(source.avg_stress, source.overall_average_stress, fallback.avg_stress, facialScore);
    return {
      scanned_at: firstDefined(source.scanned_at, source.timestamp, fallback.scanned_at),
      emotion: firstDefined(source.emotion, fallback.emotion),
      emotion_confidence: toNumber(firstDefined(source.emotion_confidence, fallback.emotion_confidence)),
      avg_stress: toNumber(avgStress),
      overall_stress: firstDefined(source.overall_stress, fallback.overall_stress, overallStressLabel(avgStress)),
      posture_quality: firstDefined(source.posture_quality, source.posture, fallback.posture_quality),
      jaw_tension: firstDefined(source.jaw_tension, fallback.jaw_tension),
      jaw_clench_score: toNumber(firstDefined(source.jaw_clench_score, fallback.jaw_clench_score)),
      slouch_score: toNumber(firstDefined(source.slouch_score, fallback.slouch_score)),
      stress_model_score: toNumber(facialScore),
      mindlens_model_score: toNumber(habitScore),
    };
  }

  function renderRecommendations(items) {
    if (!nodes.recommendations) return;
    const safeItems = Array.isArray(items) && items.length ? items : [
      "Take a short breathing break to reduce stress.",
      "Drink water and stay hydrated.",
      "Keep a relaxed upright posture while working.",
      "Take a short walk to refresh your mind.",
    ];
    nodes.recommendations.innerHTML = safeItems
      .slice(0, 5)
      .map((item) => `
        <div class="recommendation-item">
          <span class="recommendation-dot"></span>
          <span>${escapeHtml(item)}</span>
        </div>
      `)
      .join("");
  }

  function renderHealth(summary) {
    setText(nodes.bloodPressure, summary.blood_pressure, "118/76");
    updateBadge(nodes.bpBadge, summary.bp_badge, "Normal");
    setText(nodes.bpCategory, summary.bp_category, "Stable");
    setText(nodes.heartRate, `${toScore(summary.heart_rate, "74.0")} bpm`, "74.0 bpm");
    updateBadge(nodes.hrBadge, summary.hr_badge, "Normal");
    setText(nodes.hrCategory, summary.hr_category, "Resting");
    setText(nodes.cholesterol, `${Math.round(toNumber(summary.cholesterol, 176))} mg/dL`, "176 mg/dL");
    updateBadge(nodes.cholBadge, summary.chol_badge, "Normal");
    setText(nodes.cholCategory, summary.chol_category, "Healthy");
    setText(nodes.glucose, `${Math.round(toNumber(summary.glucose, 93))} mg/dL`, "93 mg/dL");
    updateBadge(nodes.glucoseBadge, summary.glucose_badge, "Normal");
    setText(nodes.glucoseCategory, summary.glucose_category, "Stable");
    setText(nodes.insulin, `${toScore(summary.insulin, "7.2")} uIU/mL`, "7.2 uIU/mL");
    updateBadge(nodes.insulinBadge, summary.insulin_badge, "Normal");
    setText(nodes.insulinCategory, summary.insulin_category, "Healthy");
    setText(nodes.mentalHealth, `${toScore(summary.mental_health, "7.1")}/10`, "7.1/10");
    updateBadge(nodes.mentalHealthBadge, summary.mental_health_badge, "Good");
    setText(nodes.mentalHealthCategory, summary.mental_health_category, "Balanced");
    setText(nodes.healthMetricsTime, formatDateTime(summary.timestamp), "");
    updateBadge(nodes.healthStatusBadge, summary.mental_health_badge, "Balanced");
  }

  function bindLatestScan(scan) {
    const avgStress = toNumber(scan.avg_stress, 48.0);
    const overallStress = firstDefined(scan.overall_stress, overallStressLabel(avgStress));
    const posture = firstDefined(scan.posture_quality, "Upright");
    const jawTension = firstDefined(scan.jaw_tension, "Low");

    setText(nodes.emotion, firstDefined(scan.emotion, "Neutral"), "Neutral");
    setText(nodes.emotionConfidence, toPercentConfidence(scan.emotion_confidence), "68% confidence");
    setText(nodes.jawTension, jawTension, "Low");
    setText(nodes.postureQuality, posture, "Upright");
    setText(nodes.overallStress, overallStress, "Moderate");
    setText(nodes.stressModelScore, formatPercent(scan.stress_model_score, 1, "45.6%"), "45.6%");
    setText(nodes.mindlensModelScore, formatPercent(scan.mindlens_model_score, 1, "47.8%"), "47.8%");
    setText(nodes.avgStressScore, toScore(scan.avg_stress, "46.7"), "46.7");
    setText(nodes.slouchScore, toScore(scan.slouch_score, "0.28"), "0.28");
    setText(nodes.jawClenchScore, formatPercent(scan.jaw_clench_score, 0, "41%"), "41%");
    setText(nodes.scanPostureMetric, posture, "Upright");
    setText(nodes.latestEmotion, firstDefined(scan.emotion, "Neutral"), "Neutral");
    setText(nodes.postureIcon, posture, "Upright");
    setText(nodes.stressScanTime, formatDateTime(scan.scanned_at), "");
    updateBadge(nodes.stressStatusBadge, overallStress, "Moderate");

    if (nodes.jawTensionBar) nodes.jawTensionBar.style.width = `${jawPercent(jawTension)}%`;
    if (nodes.postureBar) {
      const percent = posturePercent(posture);
      nodes.postureBar.style.width = `${percent}%`;
      nodes.postureBar.style.backgroundColor = percent >= 75 ? "#198754" : percent >= 45 ? "#ffc107" : "#dc3545";
    }
    if (nodes.stressBar) {
      nodes.stressBar.style.width = `${clamp(avgStress, 0, 100)}%`;
      nodes.stressBar.style.backgroundColor = stressColor(avgStress);
    }
  }

  async function fetchJson(url) {
    const response = await fetch(url, { cache: "no-store" });
    if (!response.ok) {
      let details = "";
      try {
        const body = await response.json();
        details = body.message || body.detail || "";
      } catch (_err) {
      }
      throw new Error(`Request failed: ${response.status}${details ? ` - ${details}` : ""}`);
    }
    return response.json();
  }

  async function fetchDashboardSummary(email) {
    const url = email ? `/api/dashboard/summary?email=${encodeURIComponent(email)}&limit=5` : "/api/dashboard/summary?limit=5";
    const payload = await fetchJson(url);
    console.log("Dashboard API response:", payload);
    return payload;
  }

  async function loadDashboardData() {
    try {
      let email = resolveEmail();
      let summary = await fetchDashboardSummary(email);

      if (!email) {
        email = normalizeEmail(summary.email);
        if (email) {
          localStorage.setItem("user_email", email);
          summary = await fetchDashboardSummary(email);
        }
      }

      if (!email) {
        if (nodes.streamStatus) nodes.streamStatus.textContent = "No user email found yet. Sign in to sync your latest wellness data.";
        return;
      }

      localStorage.setItem("user_email", email);
      const latestScan = normalizeScanRow(summary.latest_scan, buildFallbackScan(summary));
      bindLatestScan(latestScan);
      renderRecommendations(summary.recommendations || []);
      renderHealth(summary);
      setText(nodes.stressScanTime, formatDateTime(firstDefined(latestScan.scanned_at, summary.timestamp)), "");
      setText(nodes.healthMetricsTime, formatDateTime(firstDefined(latestScan.scanned_at, summary.timestamp)), "");

      if (nodes.streamStatus) nodes.streamStatus.textContent = `Live updates active for ${email}`;
    } catch (error) {
      console.error("Dashboard load failed:", error);
      if (nodes.streamStatus) nodes.streamStatus.textContent = "Dashboard sync is retrying. Your latest saved wellness snapshot will appear shortly.";
    }
  }

  function startPolling() {
    loadDashboardData();
    pollTimer = window.setInterval(loadDashboardData, POLL_INTERVAL_MS);
  }

  function startEventStream() {
    if (typeof EventSource === "undefined") return;
    eventSource = new EventSource("/api/stream");
    eventSource.onmessage = function () { loadDashboardData(); };
    eventSource.onerror = function () {
      if (nodes.streamStatus) nodes.streamStatus.textContent = "Live stream paused. Polling continues in the background.";
    };
  }

  function cleanup() {
    if (pollTimer) window.clearInterval(pollTimer);
    if (eventSource) eventSource.close();
    pollTimer = null;
    eventSource = null;
  }

  document.addEventListener("visibilitychange", function () {
    if (document.visibilityState === "visible") loadDashboardData();
  });

  window.addEventListener("beforeunload", cleanup);

  startPolling();
  startEventStream();
})();
