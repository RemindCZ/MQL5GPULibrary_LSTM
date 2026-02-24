const runtimeStats = [
  { value: "v1.4.0", title: "Verze runtime", text: "Aktuální hlavička v kernel.cu deklaruje LSTM DLL v1.4.0." },
  { value: "column-major", title: "Maticový layout", text: "Všechny GEMM operace jsou orientované pro nativní cuBLAS pořadí." },
  { value: "std::shared_mutex + atomics", title: "Thread safety", text: "Modely kombinují zámky pro mutace a lock-free čtení stavu/progressu." },
  { value: "Progress API", title: "Telemetry", text: "DN_GetProgress* a DN_GetProgressAll zpřístupňují živé metriky tréninku." }
];

const featureCards = [
  { title: "Multi-layer LSTM", text: "Přidávání vrstev přes DN_AddLayerEx + samostatná výstupní projekce DN_SetOutputDim." },
  { title: "Asynchronní trénink", text: "DN_TrainAsync spustí worker thread a MT5 může mezitím pokračovat v OnTimer/OnCalculate." },
  { title: "Progress reporting", text: "Epocha, mini-batch, LR, MSE, best-MSE, grad norm, ETA i procenta dokončení bez blokace." },
  { title: "Checkpointing", text: "DN_SnapshotWeights a DN_RestoreWeights chrání model před degradací během experimentů." },
  { title: "Serializace stavu", text: "DN_SaveState / DN_GetState / DN_LoadState umožňují přenos modelu ve textové podobě LSTM_V1." },
  { title: "CUDA kontext", text: "Per-thread GPUContext drží stream, cuBLAS handle i cuRAND generator v RAII režimu." }
];

const trainingStates = [
  { code: "0", label: "TS_IDLE", description: "Model netrénuje nebo doběhl a čeká na další instrukce." },
  { code: "1", label: "TS_TRAINING", description: "Asynchronní worker aktivně učí model. Progress lze průběžně pollovat." },
  { code: "2", label: "TS_COMPLETED", description: "Trénink dokončen. Výsledek lze načíst přes DN_GetTrainingResult." },
  { code: "-1", label: "TS_ERROR", description: "Došlo k chybě. Detail vrací DN_GetError." }
];

const apiRows = [
  ["DN_Create", "Životní cyklus", "sync", "Vytvoří instanci modelu, vrací handle > 0."],
  ["DN_Free", "Životní cyklus", "sync", "Bezpečně zastaví trénink, synchronizuje worker a uvolní model."],
  ["DN_SetSequenceLength", "Konfigurace", "sync", "Nastaví délku sekvence (min. 1)."],
  ["DN_SetMiniBatchSize", "Konfigurace", "sync", "Nastaví mini-batch velikost (min. 1)."],
  ["DN_AddLayerEx", "Architektura", "sync", "Přidá LSTM vrstvu včetně dropout parametru."],
  ["DN_SetGradClip", "Optimalizace", "sync", "Nastaví gradient clipping threshold."],
  ["DN_SetOutputDim", "Architektura", "sync", "Inicializuje/rebuildne výstupní lineární projekci."],
  ["DN_LoadBatch", "Data", "sync", "Načte X/T dataset do persistentních GPU bufferů."],
  ["DN_PredictBatch", "Inference", "sync", "Vrací batch predikce do host bufferu Y."],
  ["DN_SnapshotWeights", "Checkpoint", "sync", "Uloží snapshot vah v paměti modelu."],
  ["DN_RestoreWeights", "Checkpoint", "sync", "Obnoví poslední snapshot vah."],
  ["DN_TrainAsync", "Trénink", "async", "Spustí background trénink s lr, wd, epochs a target_mse."],
  ["DN_GetTrainingStatus", "Trénink", "async", "Lock-free polling stavu běhu tréninku."],
  ["DN_GetTrainingResult", "Trénink", "async", "Vrátí finální MSE a počet epoch."],
  ["DN_StopTraining", "Trénink", "async", "Nastaví stop flag a nechá worker bezpečně doběhnout."],
  ["DN_GetProgressEpoch", "Progress", "async", "Aktuální epocha."],
  ["DN_GetProgressTotalEpochs", "Progress", "async", "Celkový počet epoch v běžícím tréninku."],
  ["DN_GetProgressMiniBatch", "Progress", "async", "Aktuální mini-batch v epoše."],
  ["DN_GetProgressTotalMiniBatches", "Progress", "async", "Počet mini-batchů na epochu."],
  ["DN_GetProgressLR", "Progress", "async", "Aktuální learning rate."],
  ["DN_GetProgressMSE", "Progress", "async", "Poslední spočtené MSE."],
  ["DN_GetProgressBestMSE", "Progress", "async", "Nejlepší dosud naměřené MSE."],
  ["DN_GetProgressGradNorm", "Progress", "async", "Aktuální gradient norm."],
  ["DN_GetProgressTotalSteps", "Progress", "async", "Celkem trénovacích kroků."],
  ["DN_GetProgressPercent", "Progress", "async", "Procento dokončení tréninku."],
  ["DN_GetProgressElapsedSec", "Progress", "async", "Uplynulý čas tréninku v sekundách."],
  ["DN_GetProgressETASec", "Progress", "async", "Odhad zbývajícího času v sekundách."],
  ["DN_GetProgressAll", "Progress", "async", "Bulk getter hlavních progress metrik jedním voláním."],
  ["DN_GetLayerCount", "Diagnostika", "sync", "Počet aktivních vrstev (LSTM + output layer)."],
  ["DN_GetLayerWeightNorm", "Diagnostika", "sync", "L2 norma vah vybrané vrstvy."],
  ["DN_GetGradNorm", "Diagnostika", "sync", "Agregovaná L2 norma gradient bufferů."],
  ["DN_SaveState", "Serializace", "sync", "Spočítá velikost textové serializace."],
  ["DN_GetState", "Serializace", "sync", "Zkopíruje serializovaný model do caller bufferu."],
  ["DN_LoadState", "Serializace", "sync", "Načte model ze serializovaného textu."],
  ["DN_GetError", "Diagnostika", "sync", "Vrátí poslední chybovou hlášku v short* bufferu."]
];

const workflowSteps = [
  "DN_Create + ověření handle.",
  "Nastavení seq_len, mini-batch a LSTM vrstev.",
  "DN_SetOutputDim a DN_LoadBatch.",
  "Volitelně snapshot vah před tréninkem.",
  "DN_TrainAsync + polling DN_GetTrainingStatus.",
  "Průběžné čtení DN_GetProgressAll (nebo DN_GetProgress*).",
  "DN_GetTrainingResult po doběhu nebo stopu.",
  "DN_PredictBatch v runtime smyčce indikátoru.",
  "DN_Free při deinitu bez dangling vláken."
];

function renderStats() {
  const root = document.getElementById("runtime-stats");
  root.innerHTML = runtimeStats.map((s) => `
    <article class="panel">
      <div class="value">${s.value}</div>
      <h3>${s.title}</h3>
      <p>${s.text}</p>
    </article>
  `).join("");
}

function renderFeatures() {
  const root = document.getElementById("feature-cards");
  root.innerHTML = featureCards.map((f) => `
    <article class="panel">
      <h3>${f.title}</h3>
      <p>${f.text}</p>
    </article>
  `).join("");
}

function renderStates() {
  const root = document.getElementById("state-cards");
  root.innerHTML = trainingStates.map((s) => `
    <article class="panel">
      <h3><code>${s.label}</code> (${s.code})</h3>
      <p>${s.description}</p>
    </article>
  `).join("");
}

function renderApi() {
  const table = document.getElementById("api-table");
  table.innerHTML = `
    <thead>
      <tr>
        <th>Funkce</th>
        <th>Oblast</th>
        <th>Režim</th>
        <th>Popis</th>
      </tr>
    </thead>
    <tbody>
      ${apiRows.map(([fn, domain, mode, desc]) => `
        <tr>
          <td><code>${fn}</code></td>
          <td>${domain}</td>
          <td><span class="tag ${mode}">${mode}</span></td>
          <td>${desc}</td>
        </tr>
      `).join("")}
    </tbody>
  `;
}

function renderWorkflow() {
  const root = document.getElementById("workflow");
  root.innerHTML = workflowSteps.map((step, idx) => `
    <div class="step">
      <div class="n">${idx + 1}</div>
      <p>${step}</p>
    </div>
  `).join("");
}

renderStats();
renderFeatures();
renderStates();
renderApi();
renderWorkflow();
