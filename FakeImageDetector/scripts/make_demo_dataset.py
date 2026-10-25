"""Create a SMALL SYNTHETIC demo dataset to smoke-test the pipeline.

Real images: crops of scikit-image sample photos (JPEG q95).
Tampered images: the same crops with a patch copied to another location, re-saved as JPEG.
This is NOT a benchmark. Accuracy on it says nothing about real-world performance.
Usage: python scripts/make_demo_dataset.py --out dataset_demo
"""
import argparse
import os
import random

import numpy as np
from PIL import Image
from skimage import data


def sources():
    out = []
    for name in ("astronaut", "coffee", "chelsea", "rocket", "camera", "coins", "moon"):
        try:
            img = getattr(data, name)()
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)
            out.append((name, img[:, :, :3]))
        except Exception as exc:  # sample image may need a download
            print("skip", name, type(exc).__name__)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dataset_demo")
    ap.add_argument("--crops", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    for cls in ("real", "tampered", "ai_generated"):
        os.makedirs(os.path.join(a.out, cls), exist_ok=True)
    n = 0
    for name, img in sources():
        H, W = img.shape[:2]
        for k in range(a.crops):
            ch, cw = int(H * rng.uniform(0.7, 1.0)), int(W * rng.uniform(0.7, 1.0))
            y, x = rng.randint(0, H - ch), rng.randint(0, W - cw)
            crop = img[y:y + ch, x:x + cw].copy()
            Image.fromarray(crop).save(os.path.join(a.out, "real", "%s_%d.jpg" % (name, k)), quality=95)
            t = crop.copy()
            ph, pw = max(24, ch // 6), max(24, cw // 6)
            sy, sx = rng.randint(0, ch - ph), rng.randint(0, cw - pw)
            for _ in range(20):
                dy, dx = rng.randint(0, ch - ph), rng.randint(0, cw - pw)
                if abs(dy - sy) > ph or abs(dx - sx) > pw:
                    break
            t[dy:dy + ph, dx:dx + pw] = crop[sy:sy + ph, sx:sx + pw]
            Image.fromarray(t).save(os.path.join(a.out, "tampered", "%s_%d.jpg" % (name, k)), quality=90)
            n += 1
    print("created %d real + %d tampered images in %s (synthetic demo, not a benchmark)" % (n, n, a.out))


if __name__ == "__main__":
    main()
