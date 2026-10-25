"""Organise an extracted Kaggle dataset (e.g. CG-1050) into dataset/{real,tampered,ai_generated}.

    python scripts/prepare_dataset.py --src path/to/extracted_kaggle_folder [--dest dataset] [--dry-run]

Images are classified by the words in their folder/file path. Ground-truth MASK images
are skipped. ALWAYS check the printed counts against the dataset's documentation:
this is a path-name heuristic, not a guarantee.
"""
import argparse
import os
import shutil
from collections import Counter

EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
SKIP = ("mask", "groundtruth", "ground_truth", "ground truth", "gt_", "_gt", "binary")
REAL = ("original", "orig", "authentic", "pristine", "untampered", "real", "genuine", "au_")
TAMP = ("tamper", "forg", "manipulat", "spliced", "copy-move", "copymove", "copy_move", "edited", "tp_")
AI = ("ai_generated", "ai-generated", "aigenerated", "synthetic", "generated", "dalle", "midjourney", "diffusion", "gan")


def classify(path):
    p = path.lower().replace("\\", "/")
    if any(s in p for s in SKIP):
        return "skip"
    if any(s in p for s in AI):
        return "ai_generated"
    if any(s in p for s in TAMP):          # check tampered BEFORE real ("untampered" handled by REAL list below)
        if "untamper" not in p:
            return "tampered"
    if any(s in p for s in REAL):
        return "real"
    return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dest", default="dataset")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    counts, unknown = Counter(), []
    for root, _, files in os.walk(a.src):
        for f in files:
            if not f.lower().endswith(EXTS):
                continue
            full = os.path.join(root, f)
            cls = classify(os.path.relpath(full, a.src))
            counts[cls] += 1
            if cls == "unknown":
                unknown.append(full)
            elif cls != "skip" and not a.dry_run:
                out = os.path.join(a.dest, cls)
                os.makedirs(out, exist_ok=True)
                name = "%s__%s" % (os.path.basename(root), f)
                shutil.copy2(full, os.path.join(out, name))
    print("Counts:", dict(counts))
    if unknown:
        print("%d files could not be classified, e.g.:" % len(unknown))
        for u in unknown[:5]:
            print("  ", u)
        print("Move them into dataset/real or dataset/tampered manually.")
    print("Dry run, nothing copied." if a.dry_run else "Done. Verify counts against the dataset description.")


if __name__ == "__main__":
    main()
