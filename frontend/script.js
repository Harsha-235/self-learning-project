
const API_BASE_URL = "http://127.0.0.1:8000";

const NORMAL_EXAMPLE = {
    temperature: 50,
    pressure: 100,
    vibration: 0.5,
    rotational_speed: 1500,
    operating_hours: 50,
    ambient_temperature: 25,
    load_percent: 40
};

const RISK_EXAMPLE = {
    temperature: 75,
    pressure: 101.5,
    vibration: 2,
    rotational_speed: 1470,
    operating_hours: 300,
    ambient_temperature: 27,
    load_percent: 68
};

const INPUTS = [
    ["tempInput", "temperature"],
    ["pressureInput", "pressure"],
    ["vibrationInput", "vibration"],
    ["speedInput", "rotational_speed"],
    ["hoursInput", "operating_hours"],
    ["ambientInput", "ambient_temperature"],
    ["loadInput", "load_percent"]
];

document.addEventListener("DOMContentLoaded", () => {
    setupNavigation();
    setupPrediction();
    setupExamples();
    setupLeakageScan();
   
    setupSplitVerification();
    setupInterlock();
    setupStatusRefresh();
    refreshSystemStatus();

    setInterval(refreshSystemStatus, 10000);
});

// -------------------- COMMON HELPERS --------------------

async function apiRequest(path, options = {}) {
    const response = await fetch(`${API_BASE_URL}${path}`, {
        ...options,
        headers: {
            ...(options.body ? { "Content-Type": "application/json" } : {}),
            ...options.headers
        }
    });

    const data = await response.json().catch(() => ({}));

    if (!response.ok) {
        const detail = data.detail || data.message || `HTTP ${response.status}`;
        throw new Error(
            typeof detail === "string" ? detail : JSON.stringify(detail)
        );
    }

    return data;
}

function setText(id, value) {
    const element = document.getElementById(id);
    if (element) {
        element.textContent = value ?? "—";
    }
}

function showMessage(id, message, type = "") {
    const element = document.getElementById(id);
    if (!element) return;

    element.textContent = message;
    element.className = "tool-result";

    if (type === "error") {
        element.style.color = "#b91c1c";
        element.style.background = "#fef2f2";
    } else if (type === "success") {
        element.style.color = "#15803d";
        element.style.background = "#f0fdf4";
    } else {
        element.style.color = "#475569";
        element.style.background = "#f8fafc";
    }
}

function setButtonLoading(button, loading, loadingText, normalText) {
    if (!button) return;

    button.disabled = loading;
    button.textContent = loading ? loadingText : normalText;
}

function safeDisplay(value) {
    if (value === null || value === undefined || value === "") {
        return "Not available";
    }

    if (Array.isArray(value)) {
        return value.join(", ");
    }

    if (typeof value === "object") {
        return JSON.stringify(value);
    }

    return String(value);
}

function makePill(element, text, state) {
    if (!element) return;

    element.textContent = text;
    element.className = "pill";

    if (state === true) {
        element.classList.add("good");
    } else if (state === false) {
        element.classList.add("bad");
    } else {
        element.classList.add("warn");
    }
}

// -------------------- NAVIGATION --------------------

function setupNavigation() {
    const links = document.querySelectorAll(".nav-link");

    links.forEach(link => {
        link.addEventListener("click", () => {
            links.forEach(item => item.classList.remove("active"));
            link.classList.add("active");
        });
    });
}

// -------------------- EXAMPLE INPUTS --------------------

function fillInputs(example) {
    INPUTS.forEach(([elementId, key]) => {
        const element = document.getElementById(elementId);

        if (element) {
            element.value = example[key];
        }
    });
}

function setupExamples() {
    document.getElementById("presetNormal")?.addEventListener("click", () => {
        fillInputs(NORMAL_EXAMPLE);
        clearPredictionResult();
    });

    document.getElementById("presetRisk")?.addEventListener("click", () => {
        fillInputs(RISK_EXAMPLE);
        clearPredictionResult();
    });

    document.getElementById("resetInputs")?.addEventListener("click", () => {
        fillInputs(NORMAL_EXAMPLE);
        clearPredictionResult();
    });
}

// -------------------- MACHINE PREDICTION --------------------

function clearPredictionResult() {
    document.getElementById("emptyResult")?.classList.remove("hidden");
    document.getElementById("detailedResult")?.classList.add("hidden");
    document.getElementById("errorMessage")?.classList.add("hidden");
}

function setupPrediction() {
    const form = document.getElementById("predictionForm");

    form?.addEventListener("submit", async event => {
        event.preventDefault();

        const errorBox = document.getElementById("errorMessage");
        errorBox?.classList.add("hidden");

        const payload = {};

        for (const [elementId, key] of INPUTS) {
            const input = document.getElementById(elementId);
            const rawValue = input?.value.trim();

            if (rawValue === "" || rawValue === undefined) {
                showPredictionError(`Please enter a value for ${key.replaceAll("_", " ")}.`);
                input?.focus();
                return;
            }

            const value = Number(rawValue);

            if (!Number.isFinite(value)) {
                showPredictionError(`Please enter a valid number for ${key.replaceAll("_", " ")}.`);
                input?.focus();
                return;
            }

            payload[key] = value;
        }

        const button = document.getElementById("btnPredict");
        setButtonLoading(
            button,
            true,
            "Checking machine...",
            "Predict machine condition"
        );

        try {
            const result = await apiRequest("/predict", {
                method: "POST",
                body: JSON.stringify(payload)
            });

            displayPrediction(result);
            await refreshSystemStatus();
        } catch (error) {
            showPredictionError(error.message);
            await refreshSystemStatus();
        } finally {
            setButtonLoading(
                button,
                false,
                "",
                "Predict machine condition"
            );
        }
    });
}

function showPredictionError(message) {
    document.getElementById("emptyResult")?.classList.add("hidden");
    document.getElementById("detailedResult")?.classList.add("hidden");

    const box = document.getElementById("errorMessage");

    if (box) {
        box.textContent = `Prediction could not be completed: ${message}`;
        box.classList.remove("hidden");
    }
}

function displayPrediction(data) {
    document.getElementById("emptyResult")?.classList.add("hidden");
    document.getElementById("errorMessage")?.classList.add("hidden");
    document.getElementById("detailedResult")?.classList.remove("hidden");

    const isRisk = Number(data.prediction) === 1;
    const isDegraded = Boolean(data.is_degraded);

    const probability = Number(data.failure_probability);
    const validProbability = Number.isFinite(probability);
    const probabilityPercent = validProbability
        ? Math.min(100, Math.max(0, probability * 100))
        : null;

    const banner = document.getElementById("resultBanner");
    banner?.classList.toggle("risk", isRisk);
    banner?.classList.toggle("fallback", isDegraded);

    setText(
        "resultIcon",
        isDegraded ? "!" : isRisk ? "!" : "✓"
    );

    setText(
        "metricClass",
        isDegraded
            ? "Safe fallback"
            : isRisk
                ? "Failure risk"
                : "Normal"
    );

    setText(
        "resultExplanation",
        isDegraded
            ? "The safety interlock is active. Read the response carefully."
            : isRisk
                ? "The model classified these readings as failure risk."
                : "The model classified these readings as normal."
    );

    setText(
        "gaugeProbText",
        probabilityPercent === null
            ? "Not available"
            : `${probabilityPercent.toFixed(1)}%`
    );

    const bar = document.getElementById("gaugeBarFill");

    if (bar) {
        bar.style.width = `${probabilityPercent ?? 0}%`;
        bar.classList.toggle("risk", isRisk);
    }

    setText(
        "metricEnvelope",
        data.safety_envelope_checked
            ? "Passed"
            : "Not confirmed"
    );

    setText("metricInterlock", safeDisplay(data.interlock_state));
    setText(
        "metricMode",
        isDegraded
            ? "Conservative fallback"
            : "Gradient Boosting model"
    );

    document.getElementById("degradedNotice")
        ?.classList.toggle("hidden", !isDegraded);
}

// -------------------- BACKEND STATUS --------------------

async function refreshSystemStatus() {
    const dot = document.getElementById("apiDot");

    try {
        const [health, status] = await Promise.all([
            apiRequest("/health"),
            apiRequest("/system/status")
        ]);

        if (dot) {
            dot.className = "dot online";
        }

        setText("apiStatus", "Backend connected");
        setText("statusBackend", health.status || "Connected");
        setText(
            "statusModel",
            health.model_loaded ? "Available" : "Not loaded"
        );
        setText(
            "statusInterlock",
            status.interlock_state || health.interlock_state || "Unknown"
        );
        setText("statusEvents", status.events_count ?? 0);
        setText(
            "modelStatus",
            health.model_loaded ? "Model available" : "Model unavailable"
        );
        setText(
            "footerState",
            status.is_degraded
                ? "Safety interlock active"
                : "Backend connected"
        );

        const modelStatus = document.getElementById("modelStatus");

        if (modelStatus) {
            modelStatus.className = "pill";
            modelStatus.classList.add(health.model_loaded ? "good" : "bad");
        }

        await refreshTelemetry();
    } catch (error) {
        if (dot) {
            dot.className = "dot offline";
        }

        setText("apiStatus", "Backend not connected");
        setText("statusBackend", "Offline");
        setText("statusModel", "Unknown");
        setText("statusInterlock", "Unavailable");
        setText("statusEvents", "—");
        setText("modelStatus", "Backend unavailable");
        setText("footerState", "Start the backend to use predictions");

        const modelStatus = document.getElementById("modelStatus");

        if (modelStatus) {
            modelStatus.className = "pill bad";
        }
    }
}

async function refreshTelemetry() {
    try {
        const data = await apiRequest("/telemetry");
        const feed = document.getElementById("telemetryLogFeed");

        if (!feed) return;

        feed.replaceChildren();

        if (!data.events || data.events.length === 0) {
            feed.textContent = "No safety events have been recorded.";
            return;
        }

        data.events.slice().reverse().forEach(event => {
            const entry = document.createElement("div");
            entry.className = "event-entry";

            const title = document.createElement("strong");
            title.textContent = `${event.state_after || "Event"} — ${event.trigger || "Unknown trigger"}`;

            const details = document.createElement("div");
            details.textContent = event.safe_action_taken || "No action details supplied.";

            entry.append(title, details);
            feed.appendChild(entry);
        });
    } catch (error) {
        setText("telemetryLogFeed", "Unable to load event history.");
    }
}

function setupStatusRefresh() {
    document.getElementById("btnRefreshStatus")?.addEventListener("click", async event => {
        const button = event.currentTarget;

        setButtonLoading(
            button,
            true,
            "Refreshing...",
            "Refresh system status"
        );

        await refreshSystemStatus();

        setButtonLoading(
            button,
            false,
            "",
            "Refresh system status"
        );
    });
}

// -------------------- LEAKAGE DETECTION --------------------

function setupLeakageScan() {
    const button = document.getElementById("btnRunO3Scan");

    button?.addEventListener("click", async () => {
        setButtonLoading(button, true, "Scanning features...", "Run leakage scan");
        showMessage("leakageSummary", "Checking the dataset for suspicious features...");

        try {
            const data = await apiRequest("/leakage/analyze", {
                method: "POST"
            });

            renderLeakageResults(data.features_analyzed || []);
        } catch (error) {
            showMessage(
                "leakageSummary",
                `Leakage scan failed: ${error.message}`,
                "error"
            );
        } finally {
            setButtonLoading(button, false, "", "Run leakage scan");
        }
    });
}
function renderLeakageResults(features) {
    const tbody = document.getElementById("leakageTableBody");
    if (!tbody) return;

    tbody.replaceChildren();

    let suspiciousCount = 0;

    if (features.length === 0) {
        const row = document.createElement("tr");
        const cell = document.createElement("td");

        cell.colSpan = 3;
        cell.textContent = "The API returned no feature results.";
        row.appendChild(cell);
        tbody.appendChild(row);

        showMessage("leakageSummary", "No feature results were returned.", "error");
        return;
    }

    features.forEach(feature => {
        const row = document.createElement("tr");
        const nameCell = document.createElement("td");
        const scoreCell = document.createElement("td");
        const findingCell = document.createElement("td");

        nameCell.textContent = feature.feature || "Unknown feature";

        const score = Number(feature.rubric_score);
        scoreCell.textContent = Number.isFinite(score)
            ? `${score}/100`
            : "Not available";

        const verdict = String(feature.verdict || "UNSPECIFIED");
        const isSuspicious =
            verdict.toUpperCase().includes("QUARANTINE") ||
            verdict.toUpperCase().includes("LEAK");

        if (isSuspicious) suspiciousCount++;

        const badge = document.createElement("span");
        badge.className = `finding ${isSuspicious ? "bad" : "good"}`;
        badge.textContent = verdict;

        findingCell.appendChild(badge);
        row.append(nameCell, scoreCell, findingCell);
        tbody.appendChild(row);
    });

    showMessage(
        "leakageSummary",
        `${features.length} features checked. ${suspiciousCount} feature(s) marked suspicious by the detector.`,
        suspiciousCount > 0 ? "error" : "success"
    );
}

// -------------------- TEMPORAL AND GROUP SPLITS --------------------

function setupSplitVerification() {
    const button = document.getElementById("btnVerifySplits");

    button?.addEventListener("click", async () => {
        setButtonLoading(
            button,
            true,
            "Verifying splits...",
            "Verify validation splits"
        );

        try {
            const data = await apiRequest("/validation/verify-split", {
                method: "POST"
            });

            const temporal = data.temporal_split;
            const group = data.group_split;

            if (temporal) {
                makePill(
                    document.getElementById("badgeTemporalSplit"),
                    temporal.valid ? "PASS" : "FAIL",
                    Boolean(temporal.valid)
                );

                setText(
                    "tokenTemporal",
                    `Training: ${safeDisplay(temporal.train_range)} | Testing: ${safeDisplay(temporal.test_range)} | Details: ${safeDisplay(temporal.token)}`
                );
            }

            if (group) {
                makePill(
                    document.getElementById("badgeGroupSplit"),
                    group.valid ? "PASS" : "FAIL",
                    Boolean(group.valid)
                );

                setText(
                    "tokenGroup",
                    `Training machines: ${safeDisplay(group.train_groups)} | Testing machines: ${safeDisplay(group.test_groups)} | Details: ${safeDisplay(group.token)}`
                );
            }
        } catch (error) {
            makePill(
                document.getElementById("badgeTemporalSplit"),
                "ERROR",
                false
            );

            makePill(
                document.getElementById("badgeGroupSplit"),
                "ERROR",
                false
            );

            setText("tokenTemporal", error.message);
            setText("tokenGroup", "Split verification could not be completed.");
        } finally {
            setButtonLoading(
                button,
                false,
                "",
                "Verify validation splits"
            );
        }
    });
}

// -------------------- SAFETY INTERLOCK AND RECOVERY --------------------

function setupInterlock() {
    document.getElementById("btnTripInterlock")?.addEventListener("click", async () => {
        const trigger = document.getElementById("faultSelect")?.value || "MANUAL";
        const confirmed = window.confirm(
            `Run ${trigger} against the backend?\n\nThis intentionally changes the safety-interlock state. You may need to use recovery afterward.`
        );

        if (!confirmed) return;

        const button = document.getElementById("btnTripInterlock");

        setButtonLoading(button, true, "Requesting safety hold...", "Test safety hold");

        try {
            const data = await apiRequest("/interlock/trip", {
                method: "POST",
                body: JSON.stringify({
                    trigger,
                    reason: "User-initiated dashboard test"
                })
            });

            showMessage(
                "interlockMessage",
                `Safety hold activated. Trigger: ${safeDisplay(data.trigger)}. Action: ${safeDisplay(data.safe_action)}. Use Attempt recovery after reviewing the test.`,
                "success"
            );

            await refreshSystemStatus();
        } catch (error) {
            showMessage(
                "interlockMessage",
                `Safety request failed: ${error.message}`,
                "error"
            );
        } finally {
            setButtonLoading(button, false, "", "Test safety hold");
        }
    });

    document.getElementById("btnRecoverInterlock")?.addEventListener("click", async () => {
        const confirmed = window.confirm(
            "Request recovery from the backend safety interlock?"
        );

        if (!confirmed) return;

        const button = document.getElementById("btnRecoverInterlock");

        setButtonLoading(button, true, "Checking recovery...", "Attempt recovery");

        try {
            const data = await apiRequest("/interlock/recover", {
                method: "POST"
            });

            const succeeded = data.status === "RECOVERED";

            showMessage(
                "interlockMessage",
                `${data.status || "Recovery response"}: ${data.message || "No details supplied."}` +
                (data.recovery_time_ms != null
                    ? ` Recovery time: ${data.recovery_time_ms} ms.`
                    : ""),
                succeeded ? "success" : "error"
            );

            await refreshSystemStatus();
        } catch (error) {
            showMessage(
                "interlockMessage",
                `Recovery request failed: ${error.message}`,
                "error"
            );
        } finally {
            setButtonLoading(button, false, "", "Attempt recovery");
        }
    });
}
