"""Runs every forensic module and the rule engine for one image."""
import hashlib
import os
import time
import traceback
import logging

import cv2
import numpy as np

import config
from modules import copy_move, edge, ela, fft_analysis, noise, preprocessing, rule_engine
from modules import statistics as img_stats
from modules.utils import new_id, save_image, write_json

log = logging.getLogger("forensics")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe(name, fn, warnings):
    """Run a module; a failure is recorded, never silently turned into a clean result."""
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001
        log.error("module %s failed: %s\n%s", name, exc, traceback.format_exc())
        warnings.append("Module '%s' failed (%s)." % (name, type(exc).__name__))
        return {"status": "failed", "score": 0.0, "error": type(exc).__name__, "metrics": {}, "images": {}}


def compute_features(pil_rgb, cfg):
    """Return (scores, metrics, modules, images, histograms, warnings)."""
    sc = cfg["scales"]
    pre = preprocessing.preprocess(pil_rgb)
    warnings = []
    m = {
        "ela": _safe("ela", lambda: ela.analyze(pre["pil"], sc), warnings),
        "noise": _safe("noise", lambda: noise.analyze(pre["gray"], sc), warnings),
        "edge": _safe("edge", lambda: edge.analyze(pre["enhanced_gray"], sc), warnings),
        "copy_move": _safe("copy_move", lambda: copy_move.analyze(pre["enhanced_gray"], sc, pre["bgr"]), warnings),
        "fft": _safe("fft", lambda: fft_analysis.analyze(pre["gray"], sc), warnings),
        "stats": _safe("stats", lambda: img_stats.analyze(pre["bgr"], pre["gray"], sc), warnings),
    }
    scores = {
        "ela": m["ela"]["score"], "noise": m["noise"]["score"], "noise_low": m["noise"].get("noise_low", 0.0),
        "edge": m["edge"]["score"], "copy_move": m["copy_move"]["score"], "fft": m["fft"]["score"],
        "freq": m["fft"].get("freq", 0.0), "stats": m["stats"]["score"],
    }
    metrics = {k: v.get("metrics", {}) for k, v in m.items()}
    images = {}
    for v in m.values():
        images.update(v.get("images", {}))
    return scores, metrics, m, images, m["stats"].get("histograms", {}), warnings, pre


def run_analysis(path, original_name, result_dir, cfg=None, analysis_id=None):
    """Full analysis of one uploaded file; writes visuals + result.json to result_dir."""
    cfg = cfg or config.load_thresholds()
    t0 = time.perf_counter()
    analysis_id = analysis_id or new_id()
    sha = sha256_file(path)
    pil, fmt = preprocessing.load_image(path)
    ow, oh = pil.size
    pil_small = preprocessing.resize_keep_aspect(pil)
    scores, metrics, modules, images, hist, warnings, pre = compute_features(pil_small, cfg)
    decision = rule_engine.classify(scores, metrics, {k: {"status": v["status"]} for k, v in modules.items()}, cfg)

    os.makedirs(result_dir, exist_ok=True)
    cv2.imwrite(
        os.path.join(result_dir, "original.jpg"), pre["bgr"], [cv2.IMWRITE_JPEG_QUALITY, 92])
    for name, img in images.items():
        save_image(os.path.join(result_dir, name + ".png"), img)

    result = {
        "id": analysis_id,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "processing_time_s": round(time.perf_counter() - t0, 3),
        "image": {"filename": os.path.basename(original_name or "image"), "format": fmt, "sha256": sha,
                  "file_size_kb": round(os.path.getsize(path) / 1024.0, 1), "orig_width": ow, "orig_height": oh,
                  "analysis_width": pil_small.size[0], "analysis_height": pil_small.size[1]},
        "scores": scores, "metrics": metrics,
        "modules": {k: {"status": v["status"], "error": v.get("error"), "note": v.get("note")} for k, v in modules.items()},
        "decision": decision, "histograms": hist, "warnings": warnings,
        "config": {"version": cfg.get("version"), "calibrated": cfg.get("calibrated"), "thresholds": cfg["thresholds"]},
    }
    write_json(os.path.join(result_dir, "result.json"), result)
    return result
