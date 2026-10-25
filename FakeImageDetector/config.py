"""Central configuration.

Decision thresholds, weights and score scales live in ``thresholds.json`` so they
can be edited (or overwritten by ``python -m modules.validation --write``)
without touching Python code.
"""
import copy
import json
import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
RESULTS_FOLDER = os.path.join(BASE_DIR, "results")
REPORTS_FOLDER = os.path.join(BASE_DIR, "reports")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
THRESHOLD_FILE = os.path.join(BASE_DIR, "thresholds.json")

MAX_UPLOAD_MB = 16
MAX_CONTENT_LENGTH = MAX_UPLOAD_MB * 1024 * 1024
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "bmp"}
MAX_PIXELS = 40_000_000      # refuse images larger than this (decompression-bomb guard)
MIN_SIDE = 64                # refuse images smaller than this
ANALYSIS_MAX_DIM = 1280      # longest side used for analysis (never upscaled)
HISTORY_LIMIT = 200          # keep at most this many saved analyses

LABELS = {
    "REAL": "LIKELY AUTHENTIC",
    "TAMPERED": "TAMPERED IMAGE",
    "AI": "AI GENERATED IMAGE",
}

DEFAULT_SCALES = {"ela_cv": 1.5, "ela_abnormal_ratio": 0.1, "noise_cv": 1.5, "noise_ref_std": 4.0, "edge_cv": 1.5, "edge_broken": 0.3, "copy_move_inliers": 20, "fft_peak_z_low": 3.0, "fft_peak_z_high": 8.0, "freq_log_low": 1.5, "freq_log_high": 3.5, "stats_uniformity_low": 0.1, "stats_uniformity_high": 0.25, "stats_entropy_max": 7.5, "stats_sharpness_ref": 500.0}

DEFAULT_THRESHOLDS = {
    "version": "builtin-defaults",
    "calibrated": False,
    "thresholds": {"copy_move": 0.60, "tamper": 0.55, "ai": 0.60},
    "weights": {
        "tamper_core": {"ela": 0.45, "noise": 0.35, "edge": 0.20},
        "tamper": {"ela": 0.30, "noise": 0.25, "edge": 0.15, "copy_move": 0.30},
        "ai_core": {"fft": 0.40, "freq": 0.30, "stats": 0.30},
        "ai": {"fft": 0.30, "freq": 0.25, "noise_low": 0.20, "stats": 0.25},
    },
    "scales": DEFAULT_SCALES,
}


def load_thresholds(path=THRESHOLD_FILE):
    """Load thresholds.json, falling back to built-in defaults if unreadable."""
    cfg = copy.deepcopy(DEFAULT_THRESHOLDS)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            user = json.load(fh)
        for key, val in user.items():
            if isinstance(val, dict) and isinstance(cfg.get(key), dict):
                cfg[key].update(val)
            else:
                cfg[key] = val
    except (OSError, ValueError):
        pass  # keep defaults; app still works
    return cfg


def ensure_dirs():
    for d in (UPLOAD_FOLDER, RESULTS_FOLDER, REPORTS_FOLDER):
        os.makedirs(d, exist_ok=True)
