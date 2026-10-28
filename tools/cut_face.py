"""Cut the head out of a photo onto a transparent background.

    python tools/cut_face.py <photo> <out.png> [--cx 640 --cy 740]

GrabCut seeded with ellipses around the face; the neck fades out below the chin.
"""
import argparse

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("photo")
    ap.add_argument("out")
    ap.add_argument("--cx", type=int, default=650)   # face centre in the photo
    ap.add_argument("--cy", type=int, default=740)
    ap.add_argument("--size", type=int, default=520)
    a = ap.parse_args()

    img = cv2.imread(a.photo)
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = a.cx, a.cy
    ell = lambda rx, ry, oy=0: ((xx - cx) / rx) ** 2 + ((yy - cy - oy) / ry) ** 2 <= 1

    mask = np.full((h, w), cv2.GC_BGD, np.uint8)
    mask[ell(380, 500, -20)] = cv2.GC_PR_BGD
    mask[ell(300, 420, -30)] = cv2.GC_PR_FGD
    mask[ell(200, 280, 20)] = cv2.GC_FGD
    # collar / shoulders beside the chin are not the face
    mask[(yy > cy + 280) & (abs(xx - cx) > 150)] = cv2.GC_BGD
    bgd, fgd = np.zeros((1, 65)), np.zeros((1, 65))
    cv2.grabCut(img, mask, None, bgd, fgd, 8, cv2.GC_INIT_WITH_MASK)
    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))
    # keep the largest blob (the head)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg)
    if n > 1:
        fg = np.where(lab == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]), 255, 0).astype(np.uint8)

    # face only: fade the neck out just below the chin
    chin = cy + 330
    fade = np.clip((chin + 40 - yy) / 80.0, 0, 1)
    alpha = cv2.GaussianBlur(fg, (0, 0), 2.0).astype(np.float32) * fade

    ys, xs = np.where(alpha > 10)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    side = max(y1 - y0, x1 - x0) + 40
    mx, my = (x0 + x1) // 2, (y0 + y1) // 2
    rgba = np.dstack([img, alpha.astype(np.uint8)])
    canvas = np.zeros((side, side, 4), np.uint8)
    sx, sy = mx - side // 2, my - side // 2
    src = rgba[max(sy, 0):sy + side, max(sx, 0):sx + side]
    canvas[max(-sy, 0):max(-sy, 0) + src.shape[0], max(-sx, 0):max(-sx, 0) + src.shape[1]] = src
    out = cv2.resize(canvas, (a.size, a.size), interpolation=cv2.INTER_AREA)
    cv2.imwrite(a.out, out)


if __name__ == "__main__":
    main()
