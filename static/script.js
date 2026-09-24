// script.js
// Handles the threshold slider and talks to the FastAPI /api/metrics endpoint.

const thresholdSlider = document.getElementById("threshold-slider");
const thresholdValueEl = document.getElementById("threshold-value");

const accuracyEl = document.getElementById("accuracy-value");
const precisionEl = document.getElementById("precision-value");
const recallEl = document.getElementById("recall-value");

const tnEl = document.getElementById("tn-value");
const fpEl = document.getElementById("fp-value");
const fnEl = document.getElementById("fn-value");
const tpEl = document.getElementById("tp-value");

const matrixTn = document.getElementById("matrix-tn");
const matrixFp = document.getElementById("matrix-fp");
const matrixFn = document.getElementById("matrix-fn");
const matrixTp = document.getElementById("matrix-tp");

const errorBanner = document.getElementById("error-banner");

function showError(message) {
  errorBanner.textContent = message;
  errorBanner.classList.remove("hidden");
}

function clearError() {
  errorBanner.textContent = "";
  errorBanner.classList.add("hidden");
}

function formatPercent(value) {
  return (value * 100).toFixed(2) + "%";
}

async function fetchMetrics(threshold) {
  try {
    const response = await fetch(`/api/metrics?threshold=${threshold}`);

    if (!response.ok) {
      const errData = await response.json().catch(() => ({}));
      throw new Error(errData.detail || `Request failed with status ${response.status}`);
    }

    const data = await response.json();
    updateDashboard(data);
    clearError();
  } catch (err) {
    showError(`Could not load metrics: ${err.message}`);
  }
}

function updateDashboard(data) {
  // Metric cards
  accuracyEl.textContent = formatPercent(data.accuracy);
  precisionEl.textContent = formatPercent(data.precision);
  recallEl.textContent = formatPercent(data.recall);

  // Count cards
  tnEl.textContent = data.tn;
  fpEl.textContent = data.fp;
  fnEl.textContent = data.fn;
  tpEl.textContent = data.tp;

  // Confusion matrix table
  matrixTn.textContent = data.tn;
  matrixFp.textContent = data.fp;
  matrixFn.textContent = data.fn;
  matrixTp.textContent = data.tp;
}

// Update the displayed threshold immediately while dragging,
// and fetch new metrics from the backend.
thresholdSlider.addEventListener("input", () => {
  const threshold = parseFloat(thresholdSlider.value).toFixed(2);
  thresholdValueEl.textContent = threshold;
  fetchMetrics(threshold);
});

// Load initial metrics at the default threshold (0.50) on page load.
window.addEventListener("DOMContentLoaded", () => {
  const initialThreshold = parseFloat(thresholdSlider.value).toFixed(2);
  thresholdValueEl.textContent = initialThreshold;
  fetchMetrics(initialThreshold);
});
