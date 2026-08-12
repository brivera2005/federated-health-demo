const stepper = document.getElementById("stepper");
const nextBtn = document.getElementById("nextBtn");
const resetBtn = document.getElementById("resetBtn");
const stepLabel = document.getElementById("stepLabel");
const sitesGrid = document.getElementById("sitesGrid");
const resultsArea = document.getElementById("resultsArea");
const auditLog = document.getElementById("auditLog");

function metricsHtml(m) {
 if (!m) return "";
 return `<div class="metric-grid">
 <div class="metric"><span>Accuracy</span><strong>${m.accuracy}</strong></div>
 <div class="metric"><span>Precision</span><strong>${m.precision}</strong></div>
 <div class="metric"><span>Recall</span><strong>${m.recall}</strong></div>
 <div class="metric"><span>AUC proxy</span><strong>${m.auc_proxy}</strong></div>
 </div>`;
}

function weightsHtml(w) {
 if (!w) return "";
 return `<div class="weights-box">
 weights: [${w.weights.join(", ")}]<br/>
 bias: ${w.bias}<br/>
 samples: ${w.sample_count} · raw_data_exported: ${w.raw_data_exported}
 </div>`;
}

function render(state) {
 stepLabel.textContent = state.step_label;
 nextBtn.disabled = state.step >= 3;
 nextBtn.textContent = state.step >= 3 ? "Complete" : "Next step";

 stepper.querySelectorAll(".step").forEach((el) => {
 const n = Number(el.dataset.step);
 el.classList.toggle("active", n === state.step);
 el.classList.toggle("done", n < state.step);
 });

 sitesGrid.innerHTML = state.sites
 .map(
 (s) => `
 <div class="site-card">
 <h3>${s.name}</h3>
 <div class="site-meta">${s.patient_count} synthetic discharges · features: ${s.features.join(", ")}</div>
 <span class="privacy-badge">PHI stays local · data_stays_local=${s.data_stays_local}</span>
 </div>`
 )
 .join("");

 let html = "";
 if (state.site_a_result) {
 const r = state.site_a_result;
 html += `<div class="result-block"><h3>Site A - local model</h3>
 <div class="msg">${r.message}</div>${metricsHtml(r.local_metrics)}${weightsHtml(r.local_weights)}</div>`;
 }
 if (state.site_b_result) {
 const r = state.site_b_result;
 html += `<div class="result-block"><h3>Site B - local model</h3>
 <div class="msg">${r.message}</div>${metricsHtml(r.local_metrics)}${weightsHtml(r.local_weights)}</div>`;
 }
 if (state.federated_result) {
 const f = state.federated_result;
 html += `<div class="result-block"><h3>Global federated model (round ${f.round_number})</h3>
 <div class="msg">${f.privacy_note}</div>
 <div class="msg">Contributors: ${f.contributing_sites.join(", ")}</div>
 ${metricsHtml(f.global_metrics)}
 <div class="weights-box">aggregated weights: [${f.aggregated_weights.join(", ")}]<br/>bias: ${f.aggregated_bias}</div>
 </div>`;
 }
 if (!html) {
 html = '<p class="msg">Click <strong>Next step</strong> to train Riverview Medical Center locally. Patient rows never leave the hospital - only coefficients upload to the coordinator.</p>';
 }
 resultsArea.innerHTML = html;

 auditLog.innerHTML = state.audit_log.map((line) => `<li>${line}</li>`).join("");
}

async function fetchState() {
 const res = await fetch("/api/state");
 render(await res.json());
}

nextBtn.addEventListener("click", async () => {
 const res = await fetch("/api/step", { method: "POST" });
 render(await res.json());
});

resetBtn.addEventListener("click", async () => {
 const res = await fetch("/api/reset", { method: "POST" });
 render(await res.json());
});

fetchState();
