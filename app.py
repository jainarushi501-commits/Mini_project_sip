"""Fake Image Detector - Flask application (rule-based digital image forensics)."""
import json
import logging
import os
import shutil
import time

from flask import Flask, abort, jsonify, render_template, request, send_file, send_from_directory, url_for

import config
from modules.pipeline import run_analysis
from modules.preprocessing import ImageError
from modules.utils import build_pdf_report, is_valid_id, new_id, prune_results, read_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH
config.ensure_dirs()


def _wants_json():
    return request.path.startswith("/api/")


def _load_result(rid):
    if not is_valid_id(rid):
        abort(404)
    path = os.path.join(config.RESULTS_FOLDER, rid, "result.json")
    if not os.path.exists(path):
        abort(404)
    return read_json(path)


# ------------------------------------------------------------------ pages
@app.route("/")
def index():
    return render_template("index.html", page="home")


@app.route("/detect")
def detect():
    return render_template("detect.html", page="detect", max_mb=config.MAX_UPLOAD_MB,
                           exts=", ".join(sorted(config.ALLOWED_EXTENSIONS)))


@app.route("/result/<rid>")
def result(rid):
    return render_template("result.html", page="detect", r=_load_result(rid))


@app.route("/about")
def about():
    return render_template("about.html", page="about")


@app.route("/contact")
def contact():
    return render_template("contact.html", page="contact")


@app.route("/history")
def history():
    items = []
    try:
        dirs = [d for d in os.listdir(config.RESULTS_FOLDER) if is_valid_id(d)]
        dirs.sort(key=lambda d: os.path.getmtime(os.path.join(config.RESULTS_FOLDER, d)), reverse=True)
        for d in dirs[:50]:
            try:
                items.append(read_json(os.path.join(config.RESULTS_FOLDER, d, "result.json")))
            except (OSError, ValueError):
                continue
    except OSError:
        pass
    return render_template("history.html", page="history", items=items)


@app.route("/health")
def health():
    return jsonify(status="ok")


# ------------------------------------------------------------------ files
@app.route("/results/<rid>/<name>")
def result_file(rid, name):
    if not is_valid_id(rid) or name not in {"original.jpg", "ela.png", "noise.png", "edge.png", "fft.png", "copy_move.png"}:
        abort(404)
    return send_from_directory(os.path.join(config.RESULTS_FOLDER, rid), name)


@app.route("/report/<rid>.pdf")
def report(rid):
    r = _load_result(rid)
    rdir = os.path.join(config.RESULTS_FOLDER, rid)
    pdf = os.path.join(config.REPORTS_FOLDER, rid + ".pdf")
    try:
        build_pdf_report(r, rdir, pdf)
    except Exception:  # noqa: BLE001
        log.exception("pdf build failed")
        abort(500)
    return send_file(pdf, as_attachment=True, download_name="forensic_report_%s.pdf" % rid[:8], mimetype="application/pdf")


# ------------------------------------------------------------------ API
@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    f = request.files.get("image")
    if f is None or not f.filename:
        return jsonify(error="No image was uploaded."), 400
    ext = f.filename.rsplit(".", 1)[-1].lower() if "." in f.filename else ""
    if ext not in config.ALLOWED_EXTENSIONS:
        return jsonify(error="Unsupported file type. Allowed: %s." % ", ".join(sorted(config.ALLOWED_EXTENSIONS))), 400
    rid = new_id()
    upload = os.path.join(config.UPLOAD_FOLDER, "%s.%s" % (rid, ext))       # never use the client filename as a path
    rdir = os.path.join(config.RESULTS_FOLDER, rid)
    try:
        f.save(upload)
        res = run_analysis(upload, f.filename, rdir, analysis_id=rid)
    except ImageError as exc:
        shutil.rmtree(rdir, ignore_errors=True)
        return jsonify(error=str(exc)), 400
    except Exception:  # noqa: BLE001
        log.exception("analysis failed")
        shutil.rmtree(rdir, ignore_errors=True)
        return jsonify(error="The analysis failed unexpectedly. Please try a different image."), 500
    finally:
        try:
            os.remove(upload)                                                 # uploads are not retained
        except OSError:
            pass
    prune_results(config.RESULTS_FOLDER, config.HISTORY_LIMIT)
    return jsonify(id=rid, url=url_for("result", rid=rid), prediction=res["decision"]["display"])


@app.route("/api/contact", methods=["POST"])
def api_contact():
    data = request.get_json(silent=True) or {}
    name, email, msg = (str(data.get(k, "")).strip() for k in ("name", "email", "message"))
    if not name or "@" not in email or len(msg) < 5:
        return jsonify(error="Please enter your name, a valid email and a message."), 400
    entry = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "name": name[:100], "email": email[:150], "message": msg[:2000]}
    with open(os.path.join(config.REPORTS_FOLDER, "contact_messages.jsonl"), "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")
    return jsonify(ok=True)


# ------------------------------------------------------------------ errors
@app.errorhandler(413)
def too_large(_):
    msg = "File is too large. Maximum size is %d MB." % config.MAX_UPLOAD_MB
    return (jsonify(error=msg), 413) if _wants_json() else (render_template("error.html", code=413, msg=msg), 413)


@app.errorhandler(404)
def not_found(_):
    return (jsonify(error="Not found."), 404) if _wants_json() else (render_template("error.html", code=404, msg="Page not found."), 404)


@app.errorhandler(500)
def server_error(_):
    return (jsonify(error="Server error."), 500) if _wants_json() else (render_template("error.html", code=500, msg="Something went wrong on our side."), 500)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
