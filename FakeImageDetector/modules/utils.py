"""Shared helpers: IDs, JSON, file housekeeping, PDF report."""
import json
import os
import re
import shutil
import uuid

import cv2
import numpy as np

_HEX_ID = re.compile(r"^[0-9a-f]{32}$")


def new_id():
    return uuid.uuid4().hex


def is_valid_id(value):
    return bool(value) and bool(_HEX_ID.match(value))


def clip01(x):
    return float(min(1.0, max(0.0, x)))


def scale01(value, low, high):
    """Linearly map value from [low, high] to [0, 1] (clipped)."""
    if high <= low:
        return 0.0
    return clip01((value - low) / (high - low))


def entropy_from_counts(counts):
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log2(p)).sum())


def to_jsonable(obj):
    """Recursively convert numpy types so json.dump never fails."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    return obj


def save_image(path, img):
    """Write a numpy BGR/gray image to disk (PNG/JPG by extension)."""
    ok = cv2.imwrite(path, img)
    if not ok:
        raise IOError("could not write " + os.path.basename(path))


def write_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(to_jsonable(data), fh, indent=2)
    os.replace(tmp, path)


def read_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def prune_results(results_dir, keep):
    """Delete the oldest result folders beyond ``keep``."""
    try:
        entries = [os.path.join(results_dir, d) for d in os.listdir(results_dir)]
        entries = [d for d in entries if os.path.isdir(d) and is_valid_id(os.path.basename(d))]
        entries.sort(key=os.path.getmtime, reverse=True)
        for old in entries[keep:]:
            shutil.rmtree(old, ignore_errors=True)
    except OSError:
        pass


def build_pdf_report(result, result_dir, pdf_path):
    """Create a PDF forensic report with reportlab."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import Image as RLImage
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    from xml.sax.saxutils import escape

    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(pdf_path, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                            title="Image Forensic Report")
    story = [Paragraph("Fake Image Detector - Forensic Report", styles["Title"]), Spacer(1, 6)]
    d = result["decision"]
    info = result["image"]
    story.append(Paragraph("Prediction: <b>%s</b>" % escape(d["display"]), styles["Heading2"]))
    story.append(Paragraph(
        "Decision margin: %s%% (rule-based margin, <b>not</b> a calibrated probability). "
        "Processing time: %.2f s. Rule fired: %s." % (d["confidence"], result["processing_time_s"],
                                                       escape(d["rule_fired"])), styles["Normal"]))
    story.append(Paragraph("File: %s | %dx%d px | %s | analysed at %dx%d" % (
        escape(info["filename"]), info["orig_width"], info["orig_height"], escape(str(info["format"])),
        info["analysis_width"], info["analysis_height"]), styles["Normal"]))
    story.append(Paragraph("Analysis ID: %s | SHA-256: %s" % (result["id"], info["sha256"]), styles["Normal"]))
    story.append(Spacer(1, 10))

    rows = [["Score", "Value (0-1)", "Module status"]]
    for key, label in (("ela", "ELA"), ("noise", "Noise"), ("edge", "Edge"), ("copy_move", "Copy-move"),
                       ("fft", "FFT"), ("freq", "Frequency"), ("stats", "Image statistics")):
        rows.append([label, "%.3f" % result["scores"].get(key, 0.0), result["modules"].get(key, {}).get("status", "n/a")])
    rows.append(["Tamper score", "%.3f" % d["tamper_score"], ""])
    rows.append(["AI score", "%.3f" % d["ai_score"], ""])
    rows.append(["Real (no-anomaly) score", "%.3f" % d["real_score"], ""])
    t = Table(rows, colWidths=[6 * cm, 4 * cm, 4 * cm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0e7490")),
                           ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                           ("GRID", (0, 0), (-1, -1), 0.4, colors.grey)]))
    story += [t, Spacer(1, 10), Paragraph("Evidence", styles["Heading3"])]
    for line in d["evidence"]:
        story.append(Paragraph("&bull; " + escape(line), styles["Normal"]))
    if result.get("warnings"):
        story.append(Paragraph("Warnings", styles["Heading3"]))
        for w in result["warnings"]:
            story.append(Paragraph("&bull; " + escape(w), styles["Normal"]))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "<i>Limitations: this is a rule-based forensic screening tool. Scores are heuristics; a result is "
        "not proof that an image is authentic or fake. Thresholds calibrated: %s (%s).</i>" % (
            result["config"]["calibrated"], escape(str(result["config"]["version"]))), styles["Normal"]))

    for fname, title in (("original.jpg", "Original (analysed)"), ("ela.png", "ELA"), ("noise.png", "Noise map"),
                         ("edge.png", "Edge map"), ("fft.png", "FFT spectrum"), ("copy_move.png", "Copy-move matches")):
        p = os.path.join(result_dir, fname)
        if os.path.exists(p):
            story.append(Spacer(1, 8))
            story.append(Paragraph(title, styles["Heading3"]))
            h, w = cv2.imread(p).shape[:2]
            width = min(12 * cm, 12 * cm)
            story.append(RLImage(p, width=width, height=width * h / w))
    doc.build(story)
