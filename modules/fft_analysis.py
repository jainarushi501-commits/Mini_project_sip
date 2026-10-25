"""FFT analysis (Hann-windowed 2-D FFT of the grey image).

Features: low/high frequency energy, their ratio, spectral mean/variance,
power spectrum, radial energy distribution and isolated high-frequency peaks
(periodic up-sampling artefacts). Limitation: sharp edges, JPEG blocking and
resizing also create spectral peaks.
"""
import cv2
import numpy as np

from modules.utils import scale01

NBINS = 32


def analyze(gray_u8, scales):
    g = gray_u8.astype(np.float32)
    h, w = g.shape
    win = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    F = np.fft.fftshift(np.fft.fft2((g - g.mean()) * win))
    amp = np.abs(F)
    power = amp ** 2
    logmag = np.log1p(amp)

    yy, xx = np.indices((h, w))
    cy, cx = h // 2, w // 2
    r = np.hypot((yy - cy) / (h / 2.0), (xx - cx) / (w / 2.0))   # 1.0 = Nyquist on the axes
    total = float(power.sum()) + 1e-12
    low = float(power[r < 0.10].sum()) / total
    high = float(power[r > 0.50].sum()) / total
    ratio = high / (low + 1e-12)

    inb = r <= 1.0
    bins = np.minimum((r * NBINS).astype(np.int32), NBINS - 1)
    radial = np.bincount(bins[inb], weights=power[inb], minlength=NBINS)
    radial = (radial / (radial.sum() + 1e-12)).tolist()

    # isolated peaks above the radial mean of the log-spectrum (cross-shaped axes excluded)
    cnt = np.bincount(bins[inb], minlength=NBINS)
    prof = np.bincount(bins[inb], weights=logmag[inb], minlength=NBINS) / np.maximum(cnt, 1)
    resid = logmag - prof[bins]
    mask = (r > 0.40) & inb & (np.abs(yy - cy) > 3) & (np.abs(xx - cx) > 3)
    vals = resid[mask]
    if vals.size > 100:
        peak_z = float((vals.max() - vals.mean()) / (vals.std() + 1e-9))
        n_peaks = int((vals > vals.mean() + 8 * vals.std()).sum())
    else:
        peak_z, n_peaks = 0.0, 0

    fft_score = scale01(peak_z, scales["fft_peak_z_low"], scales["fft_peak_z_high"])
    freq_score = scale01(-np.log10(ratio + 1e-12), scales["freq_log_low"], scales["freq_log_high"])

    v = logmag / max(float(logmag.max()), 1e-9)
    vis = cv2.applyColorMap((v * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
    return {
        "status": "ok", "score": float(fft_score), "freq": float(freq_score),
        "metrics": {"low_freq_energy": low, "high_freq_energy": high, "frequency_ratio": ratio,
                    "spectral_mean": float(logmag.mean()), "spectral_variance": float(logmag.var()),
                    "total_power": total, "peak_z": peak_z, "high_freq_peaks": n_peaks,
                    "radial_energy": radial},
        "images": {"fft": vis},
    }
