"""Edge analysis with Canny, Sobel and Laplacian (run on the enhanced image)."""
import cv2
import numpy as np
from scipy import ndimage

from modules.utils import clip01

BLOCK = 32


def analyze(enhanced_gray, scales):
    g = enhanced_gray
    canny = cv2.Canny(cv2.GaussianBlur(g, (3, 3), 0), 50, 150)
    edges = canny > 0
    sx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    sobel_mag = np.hypot(sx, sy)
    lap_var = float(cv2.Laplacian(g, cv2.CV_32F).var())

    density = float(edges.mean())
    n_edge = int(edges.sum())
    if n_edge > 0:
        labels, n = ndimage.label(edges, structure=np.ones((3, 3)))
        sizes = np.bincount(labels.ravel())[1:]
        continuity = float(sizes[sizes >= 20].sum() / sizes.sum())
        irregular = float((sizes < 5).sum() / max(n, 1))
    else:
        n, continuity, irregular = 0, 0.0, 0.0
    broken = 1.0 - continuity

    h, w = g.shape
    h2, w2 = (h // BLOCK) * BLOCK, (w // BLOCK) * BLOCK
    bd = edges[:h2, :w2].reshape(h2 // BLOCK, BLOCK, w2 // BLOCK, BLOCK).mean(axis=(1, 3))
    cv = float(bd.std() / (bd.mean() + 1e-6)) if bd.mean() > 0 else 0.0

    score = clip01(0.5 * clip01(cv / scales["edge_cv"]) + 0.5 * clip01(broken / scales["edge_broken"]))
    return {
        "status": "ok", "score": score,
        "metrics": {"edge_density": density, "edge_continuity": continuity, "broken_edge_ratio": broken,
                    "irregular_structures": irregular, "edge_components": int(n), "block_density_cv": cv,
                    "sobel_mean": float(sobel_mag.mean()), "laplacian_variance": lap_var},
        "images": {"edge": cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)},
    }
