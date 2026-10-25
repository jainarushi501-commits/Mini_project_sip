import io
import numpy as np
import pytest
from PIL import Image

import config
from modules import preprocessing, rule_engine, validation
from modules.pipeline import compute_features

CFG = config.load_thresholds()


def _noise_img(seed=0, size=(256, 320)):
    rng = np.random.default_rng(seed)
    base = rng.normal(128, 40, size + (3,)).clip(0, 255).astype(np.uint8)
    return Image.fromarray(base)


def test_preprocess_shapes():
    pre = preprocessing.preprocess(_noise_img())
    assert pre["gray"].shape == (256, 320) and pre["enhanced_bgr"].shape == (256, 320, 3)


def test_alpha_removed_and_resize(tmp_path):
    p = tmp_path / "a.png"
    Image.new("RGBA", (2000, 1000), (255, 0, 0, 128)).save(p)
    img, fmt = preprocessing.load_image(str(p))
    assert img.mode == "RGB" and fmt == "PNG"
    small = preprocessing.resize_keep_aspect(img, 1000)
    assert small.size == (1000, 500)


def test_corrupt_and_tiny_rejected(tmp_path):
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not an image")
    with pytest.raises(preprocessing.ImageError):
        preprocessing.load_image(str(bad))
    tiny = tmp_path / "t.png"
    Image.new("RGB", (10, 10)).save(tiny)
    with pytest.raises(preprocessing.ImageError):
        preprocessing.load_image(str(tiny))


def test_all_scores_in_range():
    scores, metrics, modules, *_ = compute_features(_noise_img(), CFG)
    assert all(m["status"] == "ok" for m in modules.values())
    for k, v in scores.items():
        assert 0.0 <= v <= 1.0, k


def test_flat_image_does_not_crash():
    flat = Image.fromarray(np.full((200, 200, 3), 120, np.uint8))
    scores, _, modules, *_ = compute_features(flat, CFG)
    assert all(m["status"] == "ok" for m in modules.values())


def test_copy_move_found_on_duplicated_textured_region():
    rng = np.random.default_rng(1)
    img = (rng.random((400, 500, 3)) * 255).astype(np.uint8)
    import cv2
    img = cv2.GaussianBlur(img, (0, 0), 1.5)
    img[50:150, 50:150] = img[250:350, 300:400]          # exact duplicate
    scores, metrics, *_ = compute_features(Image.fromarray(img), CFG)
    assert metrics["copy_move"]["good_matches"] > 0


def test_rule_cascade_order():
    z = {k: 0.0 for k in rule_engine.SCORE_KEYS}
    base = {"copy_move": {}, "ela": {}, "noise": {}, "fft": {}}
    assert rule_engine.classify(z, base, {}, CFG)["label"] == "REAL"
    cm = dict(z, copy_move=0.9, fft=1.0, freq=1.0, stats=1.0)
    assert rule_engine.classify(cm, base, {}, CFG)["label"] == "TAMPERED"      # copy-move wins over AI rule
    ai = dict(z, fft=1.0, freq=1.0, stats=1.0)
    assert rule_engine.classify(ai, base, {}, CFG)["label"] == "AI"
    tp = dict(z, ela=1.0, noise=1.0, edge=1.0)
    out = rule_engine.classify(tp, base, {}, CFG)
    assert out["label"] == "TAMPERED" and out["rule_fired"] == "tamper_core"


def test_failed_module_is_reported():
    z = {k: 0.0 for k in rule_engine.SCORE_KEYS}
    out = rule_engine.classify(z, {}, {"ela": {"status": "failed"}}, CFG)
    assert any("did not complete" in e for e in out["evidence"])


def test_metrics_hand_checked():
    y = np.array([0, 0, 1, 1, 1])
    p = np.array([0, 1, 1, 1, 0])
    rep = validation.report(y, p, ["real", "tampered", "ai_generated"])
    assert rep["confusion_matrix"][0][:2] == [1, 1] and rep["confusion_matrix"][1][:2] == [1, 2]
    assert rep["accuracy"] == pytest.approx(3 / 5)
    assert rep["per_class"]["tampered"]["precision"] == pytest.approx(2 / 3)
    assert rep["per_class"]["tampered"]["recall"] == pytest.approx(2 / 3)
    assert rep["per_class"]["ai_generated"]["precision"] is None      # undefined, not 0 or 100%
