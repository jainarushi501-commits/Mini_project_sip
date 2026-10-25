"""Image loading, validation and preprocessing.

Important: the *enhanced* image (bilateral, CLAHE, gamma, sharpening...) is used
only for edge and SIFT analysis. ELA, noise, FFT and statistics use the raw
resized pixels, because enhancement would destroy the very evidence they measure.
The original upload is never modified.
"""
import cv2
import numpy as np
from PIL import Image, ImageOps

import config


class ImageError(Exception):
    """Raised for unsupported, corrupted or oversized images (safe to show users)."""


def load_image(path):
    """Decode an image safely. Returns (PIL RGB image, format string)."""
    try:
        with Image.open(path) as probe:
            probe.verify()                      # structural check
        with Image.open(path) as im:            # re-open: verify() invalidates the handle
            fmt = im.format
            w, h = im.size
            if w * h > config.MAX_PIXELS:
                raise ImageError("Image is too large (%d x %d). Maximum is %.0f megapixels."
                                 % (w, h, config.MAX_PIXELS / 1e6))
            if min(w, h) < config.MIN_SIDE:
                raise ImageError("Image is too small (%d x %d). Minimum side is %d px." % (w, h, config.MIN_SIDE))
            im = ImageOps.exif_transpose(im)
            if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
                rgba = im.convert("RGBA")
                bg = Image.new("RGB", rgba.size, (255, 255, 255))
                bg.paste(rgba, mask=rgba.split()[3])      # remove alpha channel
                im = bg
            else:
                im = im.convert("RGB")
            im.load()
            return im.copy(), fmt
    except ImageError:
        raise
    except Image.DecompressionBombError:
        raise ImageError("Image is too large to process safely.")
    except Exception:
        raise ImageError("The file is not a valid or readable image (corrupted or unsupported format).")


def resize_keep_aspect(pil_img, max_dim=config.ANALYSIS_MAX_DIM):
    """Downscale (never upscale) so the longest side is <= max_dim."""
    w, h = pil_img.size
    longest = max(w, h)
    if longest <= max_dim:
        return pil_img
    s = max_dim / float(longest)
    return pil_img.resize((max(1, int(round(w * s))), max(1, int(round(h * s)))), Image.LANCZOS)


def _auto_gamma(gray):
    mean = float(np.clip(gray.mean(), 1, 254)) / 255.0
    return float(np.clip(np.log(0.5) / np.log(mean), 0.6, 1.6))


def enhance(bgr):
    """Bilateral filter -> hist-eq/CLAHE -> gamma -> contrast -> normalise -> unsharp mask."""
    den = cv2.bilateralFilter(bgr, 5, 50, 50)                 # noise normalisation
    lab = cv2.cvtColor(den, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_eq = cv2.equalizeHist(l)
    l = cv2.addWeighted(l_eq, 0.4, l, 0.6, 0)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    out = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    g = _auto_gamma(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY))    # gamma correction
    lut = np.array([((i / 255.0) ** g) * 255 for i in range(256)], dtype=np.uint8)
    out = cv2.LUT(out, lut)
    out = cv2.convertScaleAbs(out, alpha=1.1, beta=0)         # contrast enhancement
    out = cv2.normalize(out, None, 0, 255, cv2.NORM_MINMAX)   # intensity normalisation
    blur = cv2.GaussianBlur(out, (0, 0), 1.2)
    return cv2.addWeighted(out, 1.5, blur, -0.5, 0)           # sharpening (unsharp mask)


def preprocess(pil_rgb):
    """Return a dict with every image representation the modules need."""
    rgb = np.ascontiguousarray(np.asarray(pil_rgb, dtype=np.uint8))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    enh = enhance(bgr)
    return {"pil": pil_rgb, "rgb": rgb, "bgr": bgr, "gray": gray,
            "enhanced_bgr": enh, "enhanced_gray": cv2.cvtColor(enh, cv2.COLOR_BGR2GRAY)}
