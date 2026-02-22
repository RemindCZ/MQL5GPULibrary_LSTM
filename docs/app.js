const metricData = [
  { title: "Zapamatování kontextu", value: "C(t-1) → C(t)", text: "LSTM buňka nese dlouhodobou paměť napříč časem." },
  { title: "Gating mechanismus", value: "forget / input / output", text: "Brány řídí, co se zahodí, co se uloží a co se pošle dál." },
  { title: "Asynchronní trénink", value: "DN_TrainAsync", text: "Učení běží na pozadí; stav se čte přes DN_GetTrainingStatus." },
  { title: "GPU akcelerace", value: "CUDA kernels", text: "Operace v síti běží na GPU kvůli vyšší propustnosti." }
];

const container = document.getElementById("lstm-metrics");

metricData.forEach((metric) => {
  const node = document.createElement("article");
  node.className = "metric";
  node.innerHTML = `
    <h3>${metric.title}</h3>
    <p><strong>${metric.value}</strong></p>
    <p>${metric.text}</p>
  `;
  container.appendChild(node);
});
