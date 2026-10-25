"""Image statistics: brightness, contrast, sharpness, entropy, texture, colour, moments."""
import cv2
import numpy as np
from scipy import stats as sstats
from scipy.signal import find_peaks

from modules.utils import clip01, entropy_from_counts, scale01


def uniform_lbp(gray_u8):
    """8-neighbour, radius-1 uniform LBP (labels 0..9) in pure NumPy."""
    g = gray_u8.astype(np.int16)
    c = g[1:-1, 1:-1]
    h, w = g.shape
    offs = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    bits = [(g[1 + dy:h - 1 + dy, 1 + dx:w - 1 + dx] >= c).astype(np.int8) for dy, dx in offs]
    trans = sum((bits[i] != bits[(i + 1) % 8]).astype(np.int8) for i in range(8))
    ones = sum(bits)
    return np.where(trans <= 2, ones, 9)


def analyze(bgr, gray_u8, scales):
    hist = np.bincount(gray_u8.ravel(), minlength=256)
    entropy = entropy_from_counts(hist)
    lap = float(cv2.Laplacian(gray_u8, cv2.CV_64F).var())
    gx = cv2.Sobel(gray_u8, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray_u8, cv2.CV_32F, 0, 1)
    grad = float(np.hypot(gx, gy).mean())

    lbp = uniform_lbp(gray_u8)
    lh = np.bincount(lbp.astype(np.int32).ravel(), minlength=10).astype(np.float64)
    lh /= lh.sum()
    uniformity = float((lh ** 2).sum())

    smooth = np.convolve(hist.astype(np.float64), np.ones(5) / 5, mode="same")
    peaks, _ = find_peaks(smooth, prominence=0.05 * smooth.max())
    flat = gray_u8.astype(np.float64).ravel()
    rgb_hist = {c: (np.bincount(bgr[:, :, i].ravel(), minlength=256).reshape(64, 4).sum(axis=1)).tolist()
                for i, c in ((0, "b"), (1, "g"), (2, "r"))}
    color_var = float(np.mean([bgr[:, :, i].astype(np.float64).var() for i in range(3)]))

    comp_u = scale01(uniformity, scales["stats_uniformity_low"], scales["stats_uniformity_high"])
    comp_e = clip01(1.0 - entropy / scales["stats_entropy_max"])
    comp_s = clip01(1.0 - lap / scales["stats_sharpness_ref"])
    score = float((comp_u + comp_e + comp_s) / 3.0)

    return {
        "status": "ok", "score": score,
        "metrics": {"brightness": float(flat.mean()), "contrast": float(flat.std()), "sharpness": lap,
                    "entropy": entropy, "texture_uniformity": uniformity, "gradient_magnitude": grad,
                    "color_variance": color_var, "histogram_peaks": int(len(peaks)),
                    "mean": float(flat.mean()), "variance": float(flat.var()),
                    "skewness": float(sstats.skew(flat)), "kurtosis": float(sstats.kurtosis(flat))},
        "histograms": rgb_hist,
        "images": {},
    }
