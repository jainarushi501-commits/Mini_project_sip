"""Configurable weighted rule engine (no trained classifier).

Decision cascade (thresholds come from thresholds.json):
  1. copy-move score  > thresholds.copy_move      -> TAMPERED
  2. weighted(ELA, noise, edge) > thresholds.tamper -> TAMPERED
  3. weighted(FFT, frequency, statistics) > thresholds.ai -> AI
  4. otherwise                                      -> REAL
``classify_batch`` is vectorised and is the single source of truth, also used by
the calibration script.
"""
import numpy as np

import config

SCORE_KEYS = ("ela", "noise", "edge", "copy_move", "fft", "freq", "stats", "noise_low")
CODES = np.array(["REAL", "TAMPERED", "AI"])
RULES = np.array(["none", "copy_move", "tamper_core", "ai_core"])


def _weighted(S, weights):
    tot = float(sum(weights.values())) or 1.0
    return sum(S[k] * (w / tot) for k, w in weights.items())


def classify_batch(S, cfg):
    """S: dict of equal-length numpy arrays keyed by SCORE_KEYS."""
    S = {k: np.asarray(v, dtype=np.float64) for k, v in S.items()}
    t, w = cfg["thresholds"], cfg["weights"]
    tamper_core = _weighted(S, w["tamper_core"])
    ai_core = _weighted(S, w["ai_core"])
    tamper = _weighted(S, w["tamper"])
    ai = _weighted(S, w["ai"])
    r1 = S["copy_move"] > t["copy_move"]
    r2 = ~r1 & (tamper_core > t["tamper"])
    r3 = ~r1 & ~r2 & (ai_core > t["ai"])
    rule = np.zeros(len(tamper), dtype=int)
    rule[r1], rule[r2], rule[r3] = 1, 2, 3
    label = np.where(r1 | r2, 1, np.where(r3, 2, 0))
    return {"label_idx": label, "rule_idx": rule, "tamper_core": tamper_core, "ai_core": ai_core,
            "tamper_score": tamper, "ai_score": ai, "real_score": 1.0 - np.maximum(tamper, ai)}


def _margin(rule, S, o, t):
    if rule == "copy_move":
        return (S["copy_move"] - t["copy_move"]) / max(1e-6, 1 - t["copy_move"])
    if rule == "tamper_core":
        return (o["tamper_core"] - t["tamper"]) / max(1e-6, 1 - t["tamper"])
    if rule == "ai_core":
        return (o["ai_core"] - t["ai"]) / max(1e-6, 1 - t["ai"])
    return min((t["copy_move"] - S["copy_move"]) / t["copy_move"],
               (t["tamper"] - o["tamper_core"]) / t["tamper"],
               (t["ai"] - o["ai_core"]) / t["ai"])


def classify(scores, metrics, modules, cfg):
    """Classify one image and build the human-readable evidence list."""
    S = {k: [float(scores.get(k, 0.0))] for k in SCORE_KEYS}
    out = classify_batch(S, cfg)
    label = str(CODES[out["label_idx"][0]])
    rule = str(RULES[out["rule_idx"][0]])
    o = {k: float(out[k][0]) for k in ("tamper_core", "ai_core", "tamper_score", "ai_score", "real_score")}
    s1 = {k: v[0] for k, v in S.items()}
    t = cfg["thresholds"]
    margin = float(np.clip(_margin(rule, s1, o, t), 0, 1))
    confidence = int(round(100 * (0.5 + 0.5 * margin)))   # decision margin, NOT a probability

    ev = []
    cm = metrics.get("copy_move", {})
    if rule == "copy_move":
        ev.append("Copy-move score %.2f exceeds threshold %.2f: %d geometrically consistent SIFT matches between "
                  "separated regions (out of %d candidate pairs, %d keypoints)." % (
                      s1["copy_move"], t["copy_move"], cm.get("inliers", 0), cm.get("good_matches", 0), cm.get("keypoints", 0)))
    elif rule == "tamper_core":
        ev.append("Combined ELA + noise + edge score %.2f exceeds threshold %.2f (ELA %.2f, noise %.2f, edge %.2f)." % (
            o["tamper_core"], t["tamper"], s1["ela"], s1["noise"], s1["edge"]))
        e = metrics.get("ela", {})
        ev.append("ELA: %d of the analysed blocks have an unusually high error level (block CV %.2f)." % (
            e.get("abnormal_blocks", 0), e.get("block_cv", 0)))
        n = metrics.get("noise", {})
        ev.append("Noise: %d blocks deviate from the median local noise level (noise consistency %.2f)." % (
            n.get("abnormal_blocks", 0), n.get("local_consistency", 0)))
    elif rule == "ai_core":
        f = metrics.get("fft", {})
        ev.append("Combined FFT + frequency + statistics score %.2f exceeds threshold %.2f (FFT %.2f, frequency %.2f, stats %.2f)." % (
            o["ai_core"], t["ai"], s1["fft"], s1["freq"], s1["stats"]))
        ev.append("Spectrum: high/low frequency energy ratio %.2e, %d isolated high-frequency peaks." % (
            f.get("frequency_ratio", 0), f.get("high_freq_peaks", 0)))
    else:
        ev.append("No rule fired: copy-move %.2f (<= %.2f), ELA+noise+edge %.2f (<= %.2f), FFT+frequency+stats %.2f (<= %.2f)." % (
            s1["copy_move"], t["copy_move"], o["tamper_core"], t["tamper"], o["ai_core"], t["ai"]))
        ev.append("'Likely authentic' means no anomaly crossed the configured thresholds. It does not prove authenticity.")
    failed = [k for k, m in modules.items() if m.get("status") != "ok"]
    if failed:
        ev.append("Warning: module(s) %s did not complete, so this result is less reliable." % ", ".join(failed))
    if not cfg.get("calibrated"):
        ev.append("Thresholds are uncalibrated defaults; run the calibration script on your dataset for measured accuracy.")
    return {"label": label, "display": config.LABELS[label], "rule_fired": rule, "confidence": confidence,
            "confidence_note": "Decision margin from the rule engine, not a calibrated probability.",
            "evidence": ev, **o}
