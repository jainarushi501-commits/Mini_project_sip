# Fake Image Detector (Digital Image Forensics)

Rule-based web app that classifies an uploaded image as **LIKELY AUTHENTIC**, **TAMPERED IMAGE** or **AI GENERATED IMAGE**.
No machine learning, deep learning or pretrained models: only classical image forensics (ELA, noise residuals, Canny/Sobel/Laplacian, SIFT copy-move, FFT, image statistics) plus a configurable rule engine.

Stack: Python, Flask, OpenCV (headless), NumPy, SciPy, Pillow, Matplotlib, ReportLab, Bootstrap 5, Chart.js.

## Run locally

```bash
python3 -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py                       # http://localhost:5000
```

Tests: `pip install -r requirements-dev.txt && python -m pytest -q`

## Dataset (Kaggle CG-1050) and calibration

The dataset is **not included** (it needs your Kaggle login).

1. Search "CG-1050" on kaggle.com, download and unzip it.
2. Sort the images into `dataset/real` and `dataset/tampered`:
   ```bash
   python scripts/prepare_dataset.py --src path/to/unzipped --dry-run   # check the counts first
   python scripts/prepare_dataset.py --src path/to/unzipped
   ```
   The script guesses classes from folder names and skips mask images. **Verify the counts against the dataset page**; move any "unknown" files by hand.
3. Calibrate thresholds and measure accuracy:
   ```bash
   python -m modules.validation --dataset dataset --seed 42 --write
   ```
   Thresholds are tuned on a development half only. Accuracy, precision, recall, F1, confusion matrix and a per-score tuning report are printed for the **held-out half** and saved to `reports/validation_report.json` and `reports/confusion_matrix.png`. `--write` stores the tuned thresholds in `thresholds.json`.

Notes for your thesis:
- CG-1050 is (to my knowledge) a copy-move/tampering dataset with **no AI-generated class**, so the AI rule cannot be tuned from it. To evaluate AI detection, add genuinely AI-generated images (realistic resolution, not tiny thumbnails) to `dataset/ai_generated`.
- Report the accuracy the script prints, whatever it is. 80% is a goal, not something the code can guarantee.
- `python scripts/make_demo_dataset.py` builds a small synthetic set to smoke-test the pipeline. It is **not a benchmark**: its tampered images also differ in JPEG quality, which ELA detects trivially.

## Configuration

`thresholds.json` holds decision thresholds, rule weights and score scales. Edit by hand or let the calibration script write it. Until calibrated, results are labelled "uncalibrated defaults".

Decision order: copy-move score > threshold -> TAMPERED; else weighted(ELA, noise, edge) > threshold -> TAMPERED; else weighted(FFT, frequency, statistics) > threshold -> AI GENERATED; else LIKELY AUTHENTIC.
The "confidence" shown is the rule's margin over its threshold, **not** a probability.

## Deploy

- **Render (Docker):** push to GitHub, then Render -> New -> Blueprint -> select the repo (`render.yaml` included). Health check: `/health`.
- **Heroku-style / Railway:** `Procfile` and `runtime.txt` included.
- **Any server:** `docker build -t fake-image-detector . && docker run -p 8000:8000 fake-image-detector`

Hosted disks are usually ephemeral, so history resets on restart. Uploads are deleted right after analysis; the last 200 result folders are kept.
Bootstrap, icons and Chart.js load from a CDN, so visitors need internet access for styling and charts.

## Push to GitHub

```bash
git init && git add . && git commit -m "Fake Image Detector"
git branch -M main
git remote add origin https://github.com/<your-user>/<your-repo>.git
git push -u origin main
```

## Layout

```
app.py  config.py  thresholds.json  requirements.txt  Dockerfile  Procfile  render.yaml
modules/   preprocessing, ela, noise, edge, copy_move, fft_analysis, statistics,
           rule_engine, pipeline, validation, utils
templates/ index, detect, result, about, contact, history, error
static/    css/style.css, js/detect.js, js/result.js
scripts/   prepare_dataset.py, make_demo_dataset.py
tests/     test_forensics.py, test_app.py
```

## Limitations

Hand-crafted features are weak against modern AI generators, recompressed images (social media) and heavy post-processing.
"Likely authentic" only means no rule fired. Results are indicators, not proof.
