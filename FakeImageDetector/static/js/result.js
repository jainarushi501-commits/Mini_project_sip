(function () {
  if (typeof Chart === "undefined") return;           // CDN blocked: page still works without charts
  const data = JSON.parse(document.getElementById("chartdata").textContent);
  Chart.defaults.color = "#93a4bd";
  Chart.defaults.borderColor = "rgba(125,211,252,.12)";
  const labels = {ela: "ELA", noise: "Noise", edge: "Edge", copy_move: "Copy-move", fft: "FFT", freq: "Frequency", stats: "Statistics"};
  const keys = Object.keys(labels);
  new Chart(document.getElementById("scoreChart"), {
    type: "bar",
    data: {labels: keys.map((k) => labels[k]), datasets: [{data: keys.map((k) => data.scores[k]), backgroundColor: "#22d3ee"}]},
    options: {plugins: {legend: {display: false}}, scales: {y: {min: 0, max: 1}}}
  });
  const h = data.hist || {};
  if (h.r) {
    const x = h.r.map((_, i) => i * 4);
    new Chart(document.getElementById("histChart"), {
      type: "line",
      data: {labels: x, datasets: [
        {label: "R", data: h.r, borderColor: "#f87171", pointRadius: 0},
        {label: "G", data: h.g, borderColor: "#34d399", pointRadius: 0},
        {label: "B", data: h.b, borderColor: "#60a5fa", pointRadius: 0}]},
      options: {scales: {x: {ticks: {maxTicksLimit: 8}}}}
    });
  }
  if (data.radial && data.radial.length) {
    new Chart(document.getElementById("radialChart"), {
      type: "line",
      data: {labels: data.radial.map((_, i) => (i / data.radial.length).toFixed(2)),
             datasets: [{label: "Energy share", data: data.radial, borderColor: "#a78bfa", fill: false, pointRadius: 0}]},
      options: {scales: {y: {type: "logarithmic"}, x: {title: {display: true, text: "normalised frequency radius"}}}}
    });
  }
})();
