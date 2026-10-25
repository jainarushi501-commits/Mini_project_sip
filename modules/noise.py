"""Noise residual analysis.

Residual = gray - GaussianBlur(gray, sigma=1.5). The image is divided into
32x32 blocks and the spread of per-block noise levels is measured. Spliced
regions often carry different noise statistics. Limitations: natural content,
denoising, sharpening and resizing also change local noise.
"""
import cv2
import numpy as np

from modules.utils import clip01, entropy_from_counts

BLOCK = 32
SIGMA = 1.5


def analyze(gray_u8, scales):
    g = gray_u8.astype(np.float32)
    res = g - cv2.GaussianBlur(g, (0, 0), SIGMA)
    absres = np.abs(res)
    h, w = g.shape
    h2, w2 = (h // BLOCK) * BLOCK, (w // BLOCK) * BLOCK
    rb = res[:h2, :w2].reshape(h2 // BLOCK, BLOCK, w2 // BLOCK, BLOCK)
    gb = g[:h2, :w2].reshape(h2 // BLOCK, BLOCK, w2 // BLOCK, BLOCK)
    bstd = rb.std(axis=(1, 3))
    bmean_int = gb.mean(axis=(1, 3))
    valid = (bmean_int > 8) & (bmean_int < 247)        # skip clipped (black/white) blocks
    vs = bstd[valid] if valid.sum() >= 4 else bstd.ravel()

    cv = float(vs.std() / (vs.mean() + 1e-6))
    med = float(np.median(vs))
    mad = float(np.median(np.abs(vs - med)) * 1.4826 + 1e-6)
    abnormal = (np.abs(bstd - med) / mad > 3.0) & valid
    consistency = float(1.0 / (1.0 + cv))

    q = np.clip(np.round(res), -32, 32).astype(np.int32).ravel() + 32
    entropy = entropy_from_counts(np.bincount(q, minlength=65))
    noise_std = float(res.std())

    score = clip01(cv / scales["noise_cv"])
    noise_low = clip01(1.0 - noise_std / scales["noise_ref_std"])   # smoothness indicator (AI score)

    vis = np.clip(cv2.GaussianBlur(absres, (0, 0), 2) * (255.0 / max(float(np.percentile(absres, 99)), 1e-3)), 0, 255)
    vis = cv2.applyColorMap(vis.astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    ys, xs = np.nonzero(abnormal)
    for y, x in zip(ys, xs):
        cv2.rectangle(vis, (int(x) * BLOCK, int(y) * BLOCK), (int(x) * BLOCK + BLOCK, int(y) * BLOCK + BLOCK),
                      (0, 0, 255), 1)
    return {
        "status": "ok", "score": score, "noise_low": noise_low,
        "metrics": {"noise_mean": float(absres.mean()), "noise_variance": float(res.var()), "noise_std": noise_std,
                    "noise_entropy": entropy, "block_std_cv": cv, "local_consistency": consistency,
                    "abnormal_blocks": int(abnormal.sum()), "valid_blocks": int(valid.sum()), "sigma": SIGMA},
        "images": {"noise": vis},
    }
