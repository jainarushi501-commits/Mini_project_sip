import io
import numpy as np
import pytest
from PIL import Image

import app as app_module


@pytest.fixture()
def client(tmp_path, monkeypatch):
    import config
    for name in ("UPLOAD_FOLDER", "RESULTS_FOLDER", "REPORTS_FOLDER"):
        d = tmp_path / name
        d.mkdir()
        monkeypatch.setattr(config, name, str(d))
    app_module.app.config["TESTING"] = True
    return app_module.app.test_client()


def _jpeg(size=(320, 240)):
    arr = (np.random.default_rng(0).random(size[::-1] + (3,)) * 255).astype(np.uint8)
    b = io.BytesIO()
    Image.fromarray(arr).save(b, "JPEG", quality=85)
    b.seek(0)
    return b


@pytest.mark.parametrize("path", ["/", "/detect", "/about", "/contact", "/history", "/health"])
def test_pages_ok(client, path):
    assert client.get(path).status_code == 200


def test_full_flow_and_pdf(client):
    r = client.post("/api/analyze", data={"image": (_jpeg(), "x.jpg")}, content_type="multipart/form-data")
    assert r.status_code == 200, r.get_json()
    rid = r.get_json()["id"]
    page = client.get("/result/" + rid)
    assert page.status_code == 200 and b"Prediction" in page.data
    assert client.get("/results/%s/ela.png" % rid).status_code == 200
    pdf = client.get("/report/%s.pdf" % rid)
    assert pdf.status_code == 200 and pdf.data[:4] == b"%PDF"
    assert client.get("/history").status_code == 200


def test_bad_inputs(client):
    assert client.post("/api/analyze", data={}, content_type="multipart/form-data").status_code == 400
    r = client.post("/api/analyze", data={"image": (io.BytesIO(b"hello"), "a.txt")}, content_type="multipart/form-data")
    assert r.status_code == 400
    r = client.post("/api/analyze", data={"image": (io.BytesIO(b"hello"), "a.jpg")}, content_type="multipart/form-data")
    assert r.status_code == 400 and "valid" in r.get_json()["error"]


def test_path_traversal_blocked(client):
    assert client.get("/results/../app.py").status_code == 404
    assert client.get("/result/not-a-valid-id").status_code == 404
    assert client.get("/results/" + "a" * 32 + "/../../app.py").status_code == 404
