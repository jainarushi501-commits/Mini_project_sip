"""Dataset evaluation and automatic threshold calibration (NOT machine learning).

    python -m modules.validation --dataset dataset --seed 42 [--write]

1. Runs every forensic module on dataset/<class>/ images and caches the scores (CSV).
2. Splits the data (stratified, seeded) into a development half and a held-out test half.
3. Tunes thresholds by exhaustive grid search on the DEVELOPMENT half only.
4. Reports accuracy / precision / recall / F1 / confusion matrix on the TEST half.
5. With --write, stores the tuned thresholds in thresholds.json (calibrated=true).
Accuracy is whatever it measures; nothing is adjusted to reach a target.
"""
import argparse
import csv
import itertools
import json
import os
import random
import sys
import time

import numpy as np

import config
from modules import preprocessing, rule_engine
from modules.pipeline import compute_features
from modules.rule_engine import SCORE_KEYS

CLASSES = ["real", "tampered", "ai_generated"]
CODE_OF = {"real": 0, "tampered": 1, "ai_generated": 2}
EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def list_dataset(root):
    items = []
    for cls in CLASSES:
        d = os.path.join(root, cls)
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                if f.lower().endswith(EXTS):
                    items.append((os.path.join(d, f), cls))
    return items


def extract_scores(items, cfg, cache_csv=None, log=print):
    """Compute (or load cached) per-image scores. Images that fail are listed, never silently dropped."""
    cache = {}
    if cache_csv and os.path.exists(cache_csv):
        with open(cache_csv, newline="") as fh:
            for row in csv.DictReader(fh):
                cache[row["path"]] = row
    rows, failed = [], []
    for n, (path, cls) in enumerate(items, 1):
        if path in cache:
            rows.append(cache[path])
            continue
        try:
            pil, _ = preprocessing.load_image(path)
            scores = compute_features(preprocessing.resize_keep_aspect(pil), cfg)[0]
            rows.append({"path": path, "label": cls, **{k: scores[k] for k in SCORE_KEYS}})
        except Exception as exc:  # noqa: BLE001
            failed.append((path, "%s: %s" % (type(exc).__name__, exc)))
        if n % 50 == 0:
            log("  processed %d / %d" % (n, len(items)))
    if cache_csv and rows:
        with open(cache_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["path", "label", *SCORE_KEYS])
            w.writeheader()
            w.writerows(rows)
    return rows, failed


def to_arrays(rows):
    S = {k: np.array([float(r[k]) for r in rows]) for k in SCORE_KEYS}
    y = np.array([CODE_OF[r["label"]] for r in rows])
    return S, y


def split(rows, seed, dev_fraction=0.5):
    rng = random.Random(seed)
    dev, test = [], []
    for cls in CLASSES:
        grp = [r for r in rows if r["label"] == cls]
        rng.shuffle(grp)
        k = int(round(len(grp) * dev_fraction))
        dev += grp[:k]
        test += grp[k:]
    return dev, test


# ---------------------------------------------------------------- metrics
def confusion(y, p, n=3):
    m = np.zeros((n, n), dtype=int)
    for a, b in zip(y, p):
        m[a, b] += 1
    return m


def report(y, p, names):
    cm = confusion(y, p, len(names))
    out = {"confusion_matrix": cm.tolist(), "labels": names, "per_class": {}}
    for i, nme in enumerate(names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp) if tp + fp else None
        rec = tp / (tp + fn) if tp + fn else None
        f1 = 2 * prec * rec / (prec + rec) if prec and rec else (0.0 if prec is not None and rec is not None else None)
        out["per_class"][nme] = {"precision": prec, "recall": rec, "f1": f1, "support": int(cm[i, :].sum())}
    out["accuracy"] = float(np.trace(cm) / cm.sum()) if cm.sum() else None
    f1s = [v["f1"] for v in out["per_class"].values() if v["f1"] is not None and v["support"] > 0]
    out["macro_f1"] = float(np.mean(f1s)) if f1s else None
    out["n"] = int(cm.sum())
    return out


def text_report(rep):
    f = lambda v: "  n/a" if v is None else "%5.3f" % v
    lines = ["%-14s %6s %6s %6s %8s" % ("class", "prec", "recall", "f1", "support")]
    for k, v in rep["per_class"].items():
        lines.append("%-14s %6s %6s %6s %8d" % (k, f(v["precision"]), f(v["recall"]), f(v["f1"]), v["support"]))
    lines.append("accuracy %s | macro-F1 %s | n=%d" % (f(rep["accuracy"]), f(rep["macro_f1"]), rep["n"]))
    return "\n".join(lines)


# ---------------------------------------------------------------- tuning
def predict(S, cfg):
    return rule_engine.classify_batch(S, cfg)["label_idx"]


def tune(S, y, base_cfg, present, steps=21):
    """Grid-search the three decision thresholds on development data (maximise macro-F1)."""
    import copy
    grid = np.round(np.linspace(0.0, 1.0, steps), 3)
    names = [CLASSES[i] for i in sorted(present)]
    best = (-1.0, None)
    ai_grid = grid if 2 in present else [1.01]            # no AI samples -> AI rule disabled
    for ct, tt, at in itertools.product(grid, grid, ai_grid):
        cfg = copy.deepcopy(base_cfg)
        cfg["thresholds"].update({"copy_move": float(ct), "tamper": float(tt), "ai": float(at)})
        rep = report(y, predict(S, cfg), [CLASSES[i] for i in range(3)])
        sc = (rep["macro_f1"] or 0.0) + 1e-3 * (rep["accuracy"] or 0.0)
        if sc > best[0]:
            best = (sc, cfg["thresholds"])
    return best[1]


def single_feature_report(S, y, grid=np.linspace(0, 1, 41)):
    """Best single-threshold real-vs-tampered split per score (threshold tuning report)."""
    mask = (y == 0) | (y == 1)
    out = {}
    for k in ("ela", "noise", "edge", "copy_move"):
        best = None
        for t in grid:
            pred = (S[k][mask] > t).astype(int)
            yy = y[mask]
            tp = int(((pred == 1) & (yy == 1)).sum()); fp = int(((pred == 1) & (yy == 0)).sum())
            fn = int(((pred == 0) & (yy == 1)).sum())
            p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * p * r / (p + r) if p + r else 0.0
            acc = float((pred == yy).mean()) if len(yy) else 0.0
            if best is None or f1 > best["f1"]:
                best = {"threshold": float(t), "f1": f1, "precision": p, "recall": r, "accuracy": acc}
        out[k] = best
    if (y == 2).any() and (y != 2).any():
        for k in ("fft", "freq", "stats"):
            best = None
            for t in grid:
                pred = (S[k] > t).astype(int); yy = (y == 2).astype(int)
                tp = int(((pred == 1) & (yy == 1)).sum()); fp = int(((pred == 1) & (yy == 0)).sum())
                fn = int(((pred == 0) & (yy == 1)).sum())
                p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0
                f1 = 2 * p * r / (p + r) if p + r else 0.0
                if best is None or f1 > best["f1"]:
                    best = {"threshold": float(t), "f1": f1, "precision": p, "recall": r}
            out[k] = best
    else:
        for k in ("fft", "freq", "stats"):
            out[k] = {"note": "no ai_generated samples in dataset, cannot tune"}
    return out


def save_confusion_png(rep, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = np.array(rep["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(rep["labels"], rotation=30); ax.set_yticklabels(rep["labels"])
    ax.set_xlabel("predicted"); ax.set_ylabel("actual")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=config.DATASET_DIR)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dev-fraction", type=float, default=0.5)
    ap.add_argument("--cache", default=os.path.join(config.REPORTS_FOLDER, "features_cache.csv"))
    ap.add_argument("--write", action="store_true", help="write tuned thresholds into thresholds.json")
    a = ap.parse_args(argv)

    config.ensure_dirs()
    cfg = config.load_thresholds()
    items = list_dataset(a.dataset)
    if not items:
        print("No images found in %s/{real,tampered,ai_generated}" % a.dataset)
        return 1
    counts = {c: sum(1 for _, k in items if k == c) for c in CLASSES}
    print("Dataset:", counts)
    t0 = time.time()
    rows, failed = extract_scores(items, cfg, a.cache)
    print("Feature extraction: %d ok, %d failed (%.0fs)" % (len(rows), len(failed), time.time() - t0))
    for p, why in failed[:10]:
        print("  FAILED", p, why)
    present = {CODE_OF[r["label"]] for r in rows}
    if len(present) < 2:
        print("Need at least two classes with images to calibrate.")
        return 1
    dev, test = split(rows, a.seed, a.dev_fraction)
    Sd, yd = to_arrays(dev)
    St, yt = to_arrays(test)
    tuned = tune(Sd, yd, cfg, present)
    tuned_cfg = {**cfg, "thresholds": tuned}
    default_rep = report(yt, predict(St, cfg), CLASSES)
    tuned_dev = report(yd, predict(Sd, tuned_cfg), CLASSES)
    tuned_test = report(yt, predict(St, tuned_cfg), CLASSES)

    out = {"dataset": counts, "failed_images": failed, "seed": a.seed, "dev_size": len(dev), "test_size": len(test),
           "tuned_thresholds": tuned, "default_thresholds": cfg["thresholds"],
           "dev_metrics_tuned": tuned_dev, "test_metrics_tuned": tuned_test, "test_metrics_default": default_rep,
           "single_feature_tuning_dev": single_feature_report(Sd, yd)}
    jpath = os.path.join(config.REPORTS_FOLDER, "validation_report.json")
    with open(jpath, "w") as fh:
        json.dump(out, fh, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    save_confusion_png(tuned_test, os.path.join(config.REPORTS_FOLDER, "confusion_matrix.png"))
    print("\nTuned thresholds (dev set only):", tuned)
    print("\n== DEV (%d images, used for tuning) ==" % len(dev)); print(text_report(tuned_dev))
    print("\n== HELD-OUT TEST (%d images) with tuned thresholds ==" % len(test)); print(text_report(tuned_test))
    print("confusion matrix [rows=actual, cols=pred %s]:\n%s" % (CLASSES, np.array(tuned_test["confusion_matrix"])))
    print("\n== HELD-OUT TEST with DEFAULT thresholds ==  accuracy: %s" % default_rep["accuracy"])
    print("\nThreshold tuning report per score (dev):")
    for k, v in out["single_feature_tuning_dev"].items():
        print("  %-10s %s" % (k, {a_: (round(b, 3) if isinstance(b, float) else b) for a_, b in v.items()}))
    acc = tuned_test["accuracy"]
    print("\nMeasured held-out accuracy: %s" % ("n/a" if acc is None else "%.1f%%" % (100 * acc)),
          "(a target of 80% is NOT assumed; this is the measured value)")
    print("Report saved to", jpath)
    if a.write:
        cfg["thresholds"] = tuned
        cfg["calibrated"] = True
        cfg["version"] = "calibrated-%s-seed%d" % (time.strftime("%Y%m%d"), a.seed)
        with open(config.THRESHOLD_FILE, "w") as fh:
            json.dump(cfg, fh, indent=2)
        print("Wrote tuned thresholds to thresholds.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
