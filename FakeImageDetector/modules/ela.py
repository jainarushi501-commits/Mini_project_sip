"""Error Level Analysis (ELA).

Re-saves the image as JPEG (quality 90) and measures the per-pixel difference.
Regions with a different compression history than the rest of the image tend to
show a different error level. Limitations: sensitive to prior compression,
resizing and format conversion; not proof of editing on its own.
"""
import io

import cv2
import numpy as np
from PIL import Image

from modules.utils import clip01

QUALITY = 90
BLOCK = 16


def analyze(pil_rgb, scales):
    buf = io.BytesIO()
    pil_rgb.save(buf, "JPEG", quality=QUALITY)
    buf.seek(0)
    rec = Image.open(buf).convert("RGB")
    a = np.asarray(pil_rgb, dtype=np.int16)
    b = np.asarray(rec, dtype=np.int16)
    ela = np.abs(a - b).astype(np.float32).mean(axis=2)

    h, w = ela.shape
    h2, w2 = (h // BLOCK) * BLOCK, (w // BLOCK) * BLOCK
    blocks = ela[:h2, :w2].reshape(h2 // BLOCK, BLOCK, w2 // BLOCK, BLOCK).mean(axis=(1, 3))
    cv = float(blocks.std() / (blocks.mean() + 1e-6))
    med = np.median(blocks)
    mad = np.median(np.abs(blocks - med)) * 1.4826 + 1e-6
    z = (blocks - med) / mad
    abnormal = z > 3.0
    abn_ratio = float(abnormal.mean())

    score = clip01(0.5 * clip01(cv / scales["ela_cv"]) + 0.5 * clip01(abn_ratio / scales["ela_abnormal_ratio"]))

    vis = np.clip(ela * (255.0 / max(float(ela.max()), 1.0)), 0, 255).astype(np.uint8)
    vis = cv2.applyColorMap(vis, cv2.COLORMAP_INFERNO)
    ys, xs = np.nonzero(abnormal)
    for y, x in zip(ys, xs):                       # highlight suspicious blocks
        cv2.rectangle(vis, (int(x) * BLOCK, int(y) * BLOCK), (int(x) * BLOCK + BLOCK, int(y) * BLOCK + BLOCK),
                      (255, 255, 0), 1)
    return {
        "status": "ok", "score": score,
        "metrics": {"quality": QUALITY, "ela_mean": float(ela.mean()), "ela_max": float(ela.max()),
                    "ela_std": float(ela.std()), "ela_variance": float(ela.var()),
                    "block_cv": cv, "abnormal_block_ratio": abn_ratio, "abnormal_blocks": int(abnormal.sum())},
        "images": {"ela": vis},
    }
