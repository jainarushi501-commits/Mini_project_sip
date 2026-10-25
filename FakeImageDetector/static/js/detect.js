(function () {
  const $ = (id) => document.getElementById(id);
  const drop = $("drop"), file = $("file"), go = $("go"), err = $("err");
  let chosen = null;
  const OK = ["image/jpeg", "image/png", "image/webp", "image/bmp"];
  function showError(m) { err.textContent = m; err.hidden = false; }
  function pick(f) {
    err.hidden = true;
    if (!f) return;
    if (!OK.includes(f.type)) { showError("Unsupported file type. Use JPEG, PNG, WebP or BMP."); return; }
    chosen = f;
    $("preview").src = URL.createObjectURL(f);
    $("fname").textContent = f.name + " (" + (f.size / 1024).toFixed(0) + " KB)";
    $("previewBox").hidden = false;
    go.disabled = false;
  }
  $("browse").addEventListener("click", (e) => { e.stopPropagation(); file.click(); });
  drop.addEventListener("click", () => file.click());
  drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); file.click(); } });
  file.addEventListener("change", () => pick(file.files[0]));
  ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("drag"); }));
  drop.addEventListener("drop", (e) => pick(e.dataTransfer.files[0]));

  go.addEventListener("click", () => {
    if (!chosen) return;
    err.hidden = true; go.disabled = true; $("prog").hidden = false;
    const fd = new FormData(); fd.append("image", chosen);
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/analyze");
    xhr.upload.onprogress = (e) => {
      if (!e.lengthComputable) return;
      const p = Math.round((e.loaded / e.total) * 100);
      $("bar").style.width = p + "%"; $("pct").textContent = p + "%";
      if (p >= 100) $("stage").innerHTML = '<span class="spinner"></span> Running forensic analysis...';
    };
    xhr.onload = () => {
      let data = {};
      try { data = JSON.parse(xhr.responseText); } catch (_) {}
      if (xhr.status === 200 && data.url) { window.location = data.url; return; }
      $("prog").hidden = true; go.disabled = false;
      showError(data.error || "Analysis failed. Please try again.");
    };
    xhr.onerror = () => { $("prog").hidden = true; go.disabled = false; showError("Network error. Please try again."); };
    xhr.send(fd);
  });
})();
