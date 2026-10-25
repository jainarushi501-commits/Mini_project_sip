"""Copy-move detection: SIFT keypoints + FLANN + Lowe ratio test + RANSAC homography.

Descriptors are matched against themselves; the nearest neighbour is the point
itself, so neighbours 2 and 3 are used for the ratio test. Pairs closer than
5% of the image diagonal are ignored (neighbouring structure, not duplication).
Limitation: repeated natural patterns (windows, tiles, leaves) cause false hits.
"""
import cv2
import numpy as np

from modules.utils import clip01

RATIO = 0.6
MIN_INLIERS = 6


def analyze(enhanced_gray, scales, draw_bgr=None):
    if not hasattr(cv2, "SIFT_create"):
        return {"status": "failed", "score": 0.0, "error": "This OpenCV build has no SIFT support.",
                "metrics": {}, "images": {}}
    sift = cv2.SIFT_create()
    kps, des = sift.detectAndCompute(enhanced_gray, None)
    n_kp = len(kps)
    base = {"keypoints": n_kp, "good_matches": 0, "inliers": 0, "forgery_score": 0.0}
    canvas = draw_bgr.copy() if draw_bgr is not None else cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)
    if des is None or n_kp < 10:
        return {"status": "ok", "score": 0.0, "metrics": base, "images": {"copy_move": canvas},
                "note": "Too few keypoints for copy-move analysis."}

    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    knn = flann.knnMatch(des.astype(np.float32), des.astype(np.float32), k=3)
    h, w = enhanced_gray.shape
    min_dist = 0.05 * np.hypot(h, w)
    pairs = set()
    for m in knn:
        if len(m) < 3:
            continue
        if m[1].distance < RATIO * m[2].distance:            # Lowe ratio test (skip self-match m[0])
            i, j = m[1].queryIdx, m[1].trainIdx
            if i == j:
                continue
            p1, p2 = np.array(kps[i].pt), np.array(kps[j].pt)
            if np.linalg.norm(p1 - p2) >= min_dist:
                pairs.add((min(i, j), max(i, j)))
    good = len(pairs)
    base["good_matches"] = good
    inliers = 0
    if good >= 4:
        pl = list(pairs)
        # use both directions so the homography sees a symmetric correspondence set
        src = np.float32([kps[i].pt for i, j in pl] + [kps[j].pt for i, j in pl])
        dst = np.float32([kps[j].pt for i, j in pl] + [kps[i].pt for i, j in pl])
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if H is not None and mask is not None:
            mask = mask.ravel().astype(bool)
            inliers = int(mask.sum() // 2)
            for k in range(len(pl)):
                if mask[k]:
                    i, j = pl[k]
                    a, b = tuple(int(v) for v in kps[i].pt), tuple(int(v) for v in kps[j].pt)
                    cv2.line(canvas, a, b, (0, 255, 255), 1)
                    cv2.circle(canvas, a, 4, (0, 0, 255), 1)
                    cv2.circle(canvas, b, 4, (0, 255, 0), 1)
    score = clip01(inliers / scales["copy_move_inliers"]) if inliers >= MIN_INLIERS else 0.0
    base.update({"inliers": inliers, "forgery_score": score, "ratio": RATIO, "min_inliers": MIN_INLIERS})
    return {"status": "ok", "score": score, "metrics": base, "images": {"copy_move": canvas}}
