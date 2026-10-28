"""Build dark.svg and light.svg (profile hero banner) and an avatar PNG.

    python tools/build_banner.py <photo> [--avatar out.png]

The photo is only read here: the SVGs carry the portrait as text characters,
never as an embedded image. Needs numpy, opencv-python-headless and Pillow.
"""
import argparse
from pathlib import Path
from xml.sax.saxutils import escape

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
W, H = 1180, 680
CARD_Y, CARD_H = 72, 580
NBSP = " "  # spaces that SVG renderers never collapse
MONO = "ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,'Liberation Mono',monospace"

# ---- profile (all from the resume / ApplyPilot profile; nothing invented) ----
HANDLE = "sandeep@midde"
NAME = "Sandeep Midde"
ROLES = [
    "Technology & Delivery Leader",
    "Principal AI Technical Product Manager",
    "GenAI Systems Builder",
]
# each value is one or more lines
ROWS = [
    ("EXPERIENCE", ["~20 years · regulated banking & finance"]),
    ("WORKED WITH", ["Wells Fargo · Oracle · Infosys",
                     "Clients: Charles Schwab · Dow Jones · UNO"]),
    ("FOCUS", ["Platform Modernization · Applied AI · Governance"]),
    ("AI PROJECTS", ["Krishna Uvaca · ObserverPrime · Elara · ApplyPilot"]),
    ("EDUCATION", ["Post Graduation - IIIT Bangalore",
                   "MBA - ICFAI",
                   "Graduation - BITS-Pilani, Rajasthan"]),
]
STACK = ["Python", "RAG", "LangChain", "ChromaDB", "Prompt Eng.", "AI Guardrails",
         "Kafka", "Azure", "Apigee", "PeopleSoft", "CI/CD", "Playwright"]
# github.com is left out: visitors are already on GitHub
LINKS = [("IN", "linkedin.com/in/sandeepmidde"), ("WEB", "sandeepmidde.com")]
TAG_TOP = "TECHNOLOGY · DELIVERY · APPLIED AI"
TAG_SUB = "Modernizing regulated platforms · shipping GenAI"
PROMPT = "~/sandeep $ ./deliver --modernize --ai"

THEMES = {
    "dark": dict(
        bg0="#030712", bg1="#07111F", bg2="#0B1220", panel="#0F172A", panel_op=".55",
        port_bg="#020617", port_op=".75", stroke="#1E293B", stroke_op=".9",
        text="#F8FAFC", muted="#94A3B8", dim="#64748B", label="#22D3EE",
        a1="#7C3AED", a2="#22D3EE", a3="#10B981", grid=".05", glow="1",
        ink=("#A78BFA", "#67E8F9", "#6EE7B7"), pill_fill="#0F172A", pill_op=".7",
        blob_op=".22", particle="#67E8F9", shadow=False, scan_op=".28",
    ),
    "light": dict(
        bg0="#FFFFFF", bg1="#F8FAFC", bg2="#EEF6FF", panel="#FFFFFF", panel_op=".62",
        port_bg="#F1F5F9", port_op=".9", stroke="#CBD5E1", stroke_op="1",
        text="#0F172A", muted="#475569", dim="#64748B", label="#0E7490",
        a1="#6D28D9", a2="#2563EB", a3="#0891B2", grid=".06", glow="0",
        ink=("#4338CA", "#1D4ED8", "#0E7490"), pill_fill="#FFFFFF", pill_op=".85",
        blob_op=".16", particle="#2563EB", shadow=True, scan_op=".18",
    ),
}

# ---- portrait ----------------------------------------------------------------
COLS, ROWS_N = 100, 58
PORT_X, PORT_Y, PORT_W, PORT_H = 54, 114, 384, 408
CELL_H = PORT_H / ROWS_N
FONT_ASCII = 6.9


def cutout(photo):
    """Head + shoulders crop and a foreground mask (GrabCut seeded by shape)."""
    img = cv2.imread(str(photo))
    img = img[20:860, 40:832]
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    cx = 400
    mask = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
    head = ((xx - cx) / 240) ** 2 + ((yy - 320) / 320) ** 2 <= 1
    shoulders = yy >= 600 + (abs(xx - cx) - 200).clip(0) * 0.6
    mask[head | shoulders] = cv2.GC_PR_FGD
    mask[((xx - cx) / 160) ** 2 + ((yy - 360) / 220) ** 2 <= 1] = cv2.GC_FGD
    mask[(yy > 650) & (abs(xx - cx) < 250)] = cv2.GC_FGD
    mask[(yy < 480) & ((xx < cx - 300) | (xx > cx + 290))] = cv2.GC_BGD
    mask[(yy < 30) & ((xx < cx - 150) | (xx > cx + 150))] = cv2.GC_BGD
    mask[(xx < 230) & (yy < 650 + (230 - xx) * 0.55) & (yy > 480)] = cv2.GC_BGD
    bgd, fgd = np.zeros((1, 65)), np.zeros((1, 65))
    cv2.grabCut(img, mask, None, bgd, fgd, 8, cv2.GC_INIT_WITH_MASK)
    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    return img, fg


def ascii_rows(img, fg, invert):
    ramp = " .:-=+*#%@"
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8)).apply(gray)
    blur = cv2.GaussianBlur(gray, (0, 0), 2.2)
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.hypot(gx, gy)
    edge = cv2.resize(edge, (COLS, ROWS_N), interpolation=cv2.INTER_AREA)
    edge = np.clip(edge / np.percentile(edge, 97), 0, 1)
    lum = cv2.resize(gray, (COLS, ROWS_N), interpolation=cv2.INTER_AREA) / 255.0
    m = cv2.resize(fg, (COLS, ROWS_N), interpolation=cv2.INTER_AREA) / 255.0
    tone = 1 - lum if invert else lum
    v = np.clip(0.82 * tone ** (0.9 if invert else 0.65) + 0.38 * edge, 0, 1)
    rows = []
    for r in range(ROWS_N):
        line = []
        for c in range(COLS):
            if m[r, c] < 0.3:
                line.append(" ")
            elif m[r, c] < 0.6:
                line.append(".")
            else:
                line.append(ramp[1 + int(v[r, c] * (len(ramp) - 2) + 0.5)])
        rows.append("".join(line))
    return rows


def avatar(img, fg, out):
    """Round avatar: the cut-out on a soft violet/cyan disc (upload it on GitHub)."""
    size = 640
    rgb = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    alpha = Image.fromarray(fg).filter(ImageFilter.GaussianBlur(1.5))
    person = rgb.convert("RGBA")
    person.putalpha(alpha)
    scale = size * 0.98 / person.width
    person = person.resize((int(person.width * scale), int(person.height * scale)), Image.LANCZOS)
    yy, xx = np.mgrid[0:size, 0:size] / size
    t = np.clip(0.55 * xx + 0.45 * yy, 0, 1)[..., None]
    c1, c2 = np.array([124, 58, 237]), np.array([34, 211, 238])
    grad = (c1 * (1 - t) + c2 * t).astype(np.uint8)
    base = Image.fromarray(grad, "RGB").convert("RGBA")
    ring = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(ring)
    d.ellipse((size * .1, size * .1, size * .9, size * .9), fill=(255, 255, 255, 38))
    base = Image.alpha_composite(base, ring)
    base.alpha_composite(person, ((size - person.width) // 2, size - person.height))
    circle = Image.new("L", (size, size), 0)
    ImageDraw.Draw(circle).ellipse((0, 0, size - 1, size - 1), fill=255)
    base.putalpha(circle)
    base.save(out)


# ---- svg helpers -------------------------------------------------------------
def reveal(delay, dur=0.5, rise=0):
    """Fade in after `delay` seconds; starts at 0s so nothing flashes first."""
    k = round(delay / (delay + dur), 4)
    s = (f'<animate attributeName="opacity" values="0;0;1" keyTimes="0;{k};1" '
         f'dur="{delay + dur:.2f}s" fill="freeze"/>')
    if rise:
        s += (f'<animateTransform attributeName="transform" type="translate" '
              f'values="0 {rise};0 {rise};0 0" keyTimes="0;{k};1" dur="{delay + dur:.2f}s" '
              f'fill="freeze"/>')
    return s


def typing(roles, x, y, size, t, gid):
    """Character-by-character role rotation with a cursor, SMIL only."""
    cw = size * 0.6
    type_s, hold, erase_s, gap = 0.055, 2.4, 0.022, 0.35
    spans = []
    for role in roles:
        n = len(role)
        spans.append(n * type_s + hold + n * erase_s + gap)
    total = sum(spans)
    out, start = [], 0.0
    for i, role in enumerate(roles):
        n = len(role)
        times, widths = [0.0], [0.0]
        for k in range(1, n + 1):
            times.append(start + k * type_s)
            widths.append(k * cw)
        times.append(start + n * type_s + hold)
        widths.append(n * cw)
        for k in range(n - 1, -1, -1):
            times.append(start + n * type_s + hold + (n - k) * erase_s)
            widths.append(k * cw)
        if times[-1] < total - 1e-6:
            times.append(total)
            widths.append(0.0)
        kt = ";".join(f"{tm / total:.5f}" for tm in times)
        wv = ";".join(f"{w_:.1f}" for w_ in widths)
        xv = ";".join(f"{x + w_ + 2:.1f}" for w_ in widths)
        shown = ";".join("1" if (start <= tm < start + spans[i] - gap) else "0" for tm in times)
        base_w = n * cw if i == 0 else 0
        out.append(
            f'<clipPath id="{gid}{i}"><rect x="{x - 2}" y="{y - size}" width="{base_w:.1f}" '
            f'height="{size * 1.4}"><animate attributeName="width" values="{wv}" keyTimes="{kt}" '
            f'dur="{total:.2f}s" calcMode="discrete" repeatCount="indefinite"/></rect></clipPath>'
            f'<text x="{x}" y="{y}" font-size="{size}" fill="url(#accent)" font-weight="600" '
            f'textLength="{n * cw:.1f}" lengthAdjust="spacingAndGlyphs" clip-path="url(#{gid}{i})">'
            f'{escape(role)}</text>'
            f'<rect x="{x + base_w + 2:.1f}" y="{y - size + 3}" width="2.5" height="{size}" '
            f'fill="{t["a2"]}" opacity="{1 if i == 0 else 0}">'
            f'<animate attributeName="x" values="{xv}" keyTimes="{kt}" dur="{total:.2f}s" '
            f'calcMode="discrete" repeatCount="indefinite"/>'
            f'<animate attributeName="opacity" values="{shown}" keyTimes="{kt}" dur="{total:.2f}s" '
            f'calcMode="discrete" repeatCount="indefinite"/></rect>'
        )
        start += spans[i]
    return f'<g>{"".join(out)}</g>'


def build_svg(theme, rows):
    t = THEMES[theme]
    o = []
    label = f"{NAME} - {ROLES[0]}. " + " ".join(
        f"{k.capitalize()}: {'; '.join(v)}." for k, v in ROWS)
    o.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}" role="img" aria-label="{escape(label)}">')
    o.append(f"<title>{escape(NAME)} - profile</title>")
    o.append("<defs>")
    o.append(f'<linearGradient id="bg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="{t["bg0"]}"/>'
             f'<stop offset=".55" stop-color="{t["bg1"]}"/><stop offset="1" stop-color="{t["bg2"]}"/></linearGradient>')
    o.append(f'<radialGradient id="blobA"><stop offset="0" stop-color="{t["a1"]}" stop-opacity="{t["blob_op"]}"/>'
             f'<stop offset="1" stop-color="{t["a1"]}" stop-opacity="0"/></radialGradient>')
    o.append(f'<radialGradient id="blobB"><stop offset="0" stop-color="{t["a2"]}" stop-opacity="{t["blob_op"]}"/>'
             f'<stop offset="1" stop-color="{t["a2"]}" stop-opacity="0"/></radialGradient>')
    o.append(f'<radialGradient id="blobC"><stop offset="0" stop-color="{t["a3"]}" stop-opacity="{t["blob_op"]}"/>'
             f'<stop offset="1" stop-color="{t["a3"]}" stop-opacity="0"/></radialGradient>')
    o.append(f'<pattern id="grid" width="32" height="32" patternUnits="userSpaceOnUse">'
             f'<path d="M32 0H0V32" fill="none" stroke="{t["muted"]}" stroke-opacity="{t["grid"]}"/></pattern>')
    o.append(f'<linearGradient id="accent" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="{t["a1"]}"/>'
             f'<stop offset=".5" stop-color="{t["a2"]}"/><stop offset="1" stop-color="{t["a3"]}"/></linearGradient>')
    o.append(f'<linearGradient id="name" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="{t["text"]}"/>'
             f'<stop offset=".7" stop-color="{t["text"]}"/><stop offset="1" stop-color="{t["a2"]}"/></linearGradient>')
    # moving shimmer for borders
    o.append(f'<linearGradient id="shimmer" x1="0" y1="0" x2="1" y2="0" gradientUnits="objectBoundingBox">'
             f'<stop offset="0" stop-color="{t["stroke"]}"/><stop offset=".45" stop-color="{t["stroke"]}"/>'
             f'<stop offset=".5" stop-color="{t["a2"]}"/><stop offset=".55" stop-color="{t["stroke"]}"/>'
             f'<stop offset="1" stop-color="{t["stroke"]}"/>'
             f'<animateTransform attributeName="gradientTransform" type="translate" values="-1 0;1 0" '
             f'dur="7s" repeatCount="indefinite"/></linearGradient>')
    i0, i1, i2 = t["ink"]
    o.append(f'<linearGradient id="ink" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="{i0}"/>'
             f'<stop offset=".5" stop-color="{i1}"/><stop offset="1" stop-color="{i2}"/>'
             f'<animateTransform attributeName="gradientTransform" type="rotate" values="0 .5 .5;360 .5 .5" '
             f'dur="24s" repeatCount="indefinite"/></linearGradient>')
    o.append(f'<linearGradient id="glass" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="{t["panel"]}" '
             f'stop-opacity="{t["panel_op"]}"/><stop offset="1" stop-color="{t["panel"]}" '
             f'stop-opacity="{float(t["panel_op"]) * .6:.2f}"/></linearGradient>')
    o.append(f'<linearGradient id="sheen" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#FFFFFF" '
             f'stop-opacity="0"/><stop offset=".5" stop-color="#FFFFFF" stop-opacity="{".05" if theme == "dark" else ".35"}"/>'
             f'<stop offset="1" stop-color="#FFFFFF" stop-opacity="0"/></linearGradient>')
    o.append(f'<clipPath id="frame"><rect x="8" y="8" width="{W - 16}" height="{H - 16}" rx="22"/></clipPath>')
    o.append(f'<clipPath id="port"><rect x="{PORT_X}" y="{PORT_Y}" width="{PORT_W}" height="{PORT_H}" rx="12"/></clipPath>')
    o.append('<filter id="noise" x="0" y="0" width="100%" height="100%"><feTurbulence type="fractalNoise" '
             'baseFrequency=".9" numOctaves="2" seed="7"/><feColorMatrix values="0 0 0 0 .5  0 0 0 0 .5  0 0 0 0 .5  0 0 0 .05 0"/></filter>')
    if t["shadow"]:
        o.append('<filter id="soft" x="-10%" y="-10%" width="120%" height="130%"><feDropShadow dx="0" dy="8" '
                 'stdDeviation="12" flood-color="#1E3A8A" flood-opacity=".10"/></filter>')
    else:
        o.append('<filter id="soft" x="-10%" y="-10%" width="120%" height="130%"><feGaussianBlur stdDeviation="0"/></filter>')
    o.append('<filter id="glow" x="-20%" y="-20%" width="140%" height="140%"><feGaussianBlur stdDeviation="1.6" result="b"/>'
             '<feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>')
    o.append("</defs>")

    # background
    o.append('<g clip-path="url(#frame)">')
    o.append(f'<rect width="{W}" height="{H}" fill="url(#bg)"/>')
    o.append(f'<ellipse cx="180" cy="120" rx="420" ry="300" fill="url(#blobA)">'
             f'<animate attributeName="cx" values="180;260;180" dur="18s" repeatCount="indefinite"/></ellipse>')
    o.append(f'<ellipse cx="1000" cy="520" rx="460" ry="320" fill="url(#blobB)">'
             f'<animate attributeName="cy" values="520;440;520" dur="22s" repeatCount="indefinite"/></ellipse>')
    o.append(f'<ellipse cx="760" cy="80" rx="300" ry="200" fill="url(#blobC)">'
             f'<animate attributeName="cx" values="760;680;760" dur="26s" repeatCount="indefinite"/></ellipse>')
    o.append(f'<rect width="{W}" height="{H}" fill="url(#grid)"/>')
    o.append(f'<rect width="{W}" height="{H}" filter="url(#noise)" opacity=".6"/>')
    rng = np.random.default_rng(11)
    for _ in range(16):
        px, py = rng.uniform(40, W - 40), rng.uniform(70, H - 40)
        r = rng.uniform(0.8, 1.7)
        dy = rng.uniform(-40, -14)
        dur = rng.uniform(9, 17)
        o.append(f'<circle cx="{px:.0f}" cy="{py:.0f}" r="{r:.1f}" fill="{t["particle"]}" opacity="0">'
                 f'<animateMotion path="M0 0 L{rng.uniform(-12, 12):.0f} {dy:.0f}" dur="{dur:.1f}s" repeatCount="indefinite"/>'
                 f'<animate attributeName="opacity" values="0;.55;0" dur="{dur:.1f}s" repeatCount="indefinite"/></circle>')
    o.append("</g>")
    o.append(f'<rect x="8.5" y="8.5" width="{W - 17}" height="{H - 17}" rx="22" fill="none" stroke="url(#shimmer)" stroke-width="1.2"/>')

    # title bar
    o.append(f'<g font-family="{MONO}">')
    o.append(f'<line x1="9" y1="52" x2="{W - 9}" y2="52" stroke="{t["stroke"]}" stroke-opacity="{t["stroke_op"]}"/>')
    for i, c in enumerate(["#FF5F57", "#FEBC2E", "#28C840"]):
        o.append(f'<circle cx="{36 + i * 20}" cy="30" r="6" fill="{c}"/>')
    o.append(f'<text x="{W / 2}" y="34" font-size="12.5" fill="{t["muted"]}" text-anchor="middle">'
             f'{HANDLE}: ~/profile — ./profile-scan --live</text>')
    o.append(f'<circle cx="{W - 104}" cy="30" r="4" fill="{t["a3"]}"><animate attributeName="opacity" '
             f'values="1;.3;1" dur="2.2s" repeatCount="indefinite"/></circle>')
    o.append(f'<text x="{W - 36}" y="34" font-size="11.5" fill="{t["a3"]}" text-anchor="end" '
             f'letter-spacing="1.5">ONLINE</text>')
    o.append("</g>")

    # left card
    cb = CARD_Y + CARD_H
    o.append(f'<g font-family="{MONO}">{reveal(0.05, .6)}')
    o.append(f'<rect x="28" y="{CARD_Y}" width="436" height="{CARD_H}" rx="16" fill="url(#glass)" stroke="url(#shimmer)" filter="url(#soft)"/>')
    o.append(f'<rect x="28" y="{CARD_Y}" width="436" height="{CARD_H}" rx="16" fill="url(#sheen)"/>')
    o.append(f'<text x="48" y="99" font-size="12" fill="{t["label"]}" letter-spacing="2.4" font-weight="700">VISUAL.MAP</text>')
    o.append(f'<text x="444" y="99" font-size="11" fill="{t["dim"]}" text-anchor="end">ascii · {COLS}×{ROWS_N}</text>')
    o.append(f'<rect x="{PORT_X}" y="{PORT_Y}" width="{PORT_W}" height="{PORT_H}" rx="12" fill="{t["port_bg"]}" '
             f'fill-opacity="{t["port_op"]}" stroke="{t["stroke"]}" stroke-opacity="{t["stroke_op"]}"/>')
    # corner brackets
    bx0, by0, bx1, by1 = PORT_X - 6, PORT_Y - 6, PORT_X + PORT_W + 6, PORT_Y + PORT_H + 6
    for (x, y, sx, sy) in [(bx0, by0, 1, 1), (bx1, by0, -1, 1), (bx0, by1, 1, -1), (bx1, by1, -1, -1)]:
        o.append(f'<path d="M{x} {y + 14 * sy}V{y}H{x + 14 * sx}" fill="none" stroke="{t["a2"]}" '
                 f'stroke-width="1.6" stroke-opacity=".8"/>')
    # ascii
    o.append(f'<g clip-path="url(#port)">')
    glow = ' filter="url(#glow)"' if t["glow"] == "1" else ""
    o.append(f'<g font-size="{FONT_ASCII}" fill="url(#ink)" style="white-space:pre"{glow}>')
    o.append('<animateTransform attributeName="transform" type="translate" values="0 0;0 -2;0 0" '
             'dur="7s" repeatCount="indefinite"/>')
    o.append('<animate attributeName="opacity" values="1;1;.86;1;1;1" keyTimes="0;.62;.63;.64;.9;1" '
             'dur="11s" repeatCount="indefinite"/>')
    for r, line in enumerate(rows):
        y = PORT_Y + 4 + (r + 0.8) * CELL_H
        o.append(f'<text x="{PORT_X}" y="{y:.1f}" textLength="{PORT_W}" lengthAdjust="spacingAndGlyphs">'
                 f'{escape(line).replace(" ", NBSP)}{reveal(0.25 + r * 0.02, .3)}</text>')
    o.append("</g>")
    o.append(f'<rect x="{PORT_X}" y="{PORT_Y}" width="{PORT_W}" height="2" fill="{t["a2"]}" opacity="{t["scan_op"]}">'
             f'<animate attributeName="y" values="{PORT_Y};{PORT_Y + PORT_H};{PORT_Y}" dur="6s" repeatCount="indefinite"/></rect>')
    o.append("</g>")
    cx = 246
    o.append(f'<text x="{cx}" y="{cb - 92}" font-size="12.5" fill="url(#accent)" letter-spacing="2.2" font-weight="700" '
             f'text-anchor="middle">{escape(TAG_TOP)}{reveal(1.4)}</text>')
    o.append(f'<text x="{cx}" y="{cb - 64}" font-size="14" fill="{t["muted"]}" text-anchor="middle" '
             f'textLength="{len(TAG_SUB) * 8.1:.1f}" lengthAdjust="spacingAndGlyphs">{escape(TAG_SUB)}{reveal(1.6)}</text>')
    pw = len(PROMPT) * 8.4
    px0 = cx - pw / 2 - 5
    o.append(f'<g>{reveal(1.8)}<text x="{px0:.1f}" y="{cb - 30}" font-size="14" fill="{t["a3"]}" textLength="{pw:.1f}" '
             f'lengthAdjust="spacingAndGlyphs">{escape(PROMPT)}</text>'
             f'<rect x="{px0 + pw + 4:.1f}" y="{cb - 42}" width="8" height="15" fill="{t["a2"]}">'
             f'<animate attributeName="opacity" values="1;0;1" dur="1.05s" calcMode="discrete" repeatCount="indefinite"/></rect></g>')
    o.append("</g>")

    # right card
    X0, LBL, VAL, XR = 504, 632, 650, 1132
    VF, LF, VC = 15.5, 12, 9.3  # value font, label font, value char width
    o.append(f'<g font-family="{MONO}">{reveal(0.15, .6)}')
    o.append(f'<rect x="484" y="{CARD_Y}" width="668" height="{CARD_H}" rx="16" fill="url(#glass)" stroke="url(#shimmer)" filter="url(#soft)"/>')
    o.append(f'<rect x="484" y="{CARD_Y}" width="668" height="{CARD_H}" rx="16" fill="url(#sheen)"/>')
    o.append(f'<text x="{X0}" y="99" font-size="12" fill="{t["label"]}" letter-spacing="2.4" font-weight="700">SYSTEM.INFO</text>')
    o.append(f'<text x="{XR}" y="99" font-size="11" fill="{t["dim"]}" text-anchor="end">uptime ~20y · build 2026</text>')
    o.append(f'<text x="{X0}" y="131" font-size="14">{reveal(0.5)}<tspan fill="{t["a3"]}">{HANDLE}:~$</tspan>'
             f'<tspan fill="{t["muted"]}"> ./profile.sh</tspan></text>')
    o.append(f'<text x="{X0}" y="182" font-size="40" font-weight="700" fill="url(#name)" letter-spacing=".5">'
             f'{escape(NAME)}{reveal(0.9, .6, rise=6)}</text>')
    roles_svg = typing(ROLES, X0 + 26, 220, 20, t, f"role{theme}")
    o.append(f'<g>{reveal(1.2)}<text x="{X0}" y="220" font-size="20" fill="{t["a2"]}" font-weight="700">›</text>'
             f'{roles_svg}</g>')
    o.append(f'<line x1="{X0}" y1="240" x2="{XR}" y2="240" stroke="{t["stroke"]}" stroke-opacity="{t["stroke_op"]}"/>')
    y = 272
    for i, (k, vals) in enumerate(ROWS):
        parts = [f'<text x="{LBL}" y="{y}" font-size="{LF}" fill="{t["label"]}" text-anchor="end" '
                 f'letter-spacing="1.2" font-weight="700">{k}</text>']
        for j, v in enumerate(vals):
            assert VAL + len(v) * VC <= XR, v
            parts.append(f'<text x="{VAL}" y="{y + j * 23}" font-size="{VF}" fill="{t["text"]}" '
                         f'textLength="{len(v) * VC:.1f}" lengthAdjust="spacingAndGlyphs">{escape(v)}</text>')
        o.append(f'<g>{reveal(1.6 + i * 0.15, .45)}{"".join(parts)}</g>')
        y += 23 * (len(vals) - 1) + 32
    # stack pills
    sy = y + 2
    o.append(f'<text x="{LBL}" y="{sy}" font-size="{LF}" fill="{t["label"]}" text-anchor="end" '
             f'letter-spacing="1.2" font-weight="700">{reveal(2.4)}STACK</text>')
    x, py = VAL, sy - 17
    pc = 7.8
    for i, s in enumerate(STACK):
        pw = len(s) * pc + 32
        if x + pw > XR:
            x, py = VAL, py + 32
        col = [t["a1"], t["a2"], t["a3"]][i % 3]
        o.append(f'<g>{reveal(2.45 + i * 0.05, .35)}<g>'
                 f'<animateTransform attributeName="transform" type="translate" values="0 0;0 -1.4;0 0" '
                 f'dur="{4 + (i % 4) * 0.7:.1f}s" begin="{i * 0.3:.1f}s" repeatCount="indefinite"/>'
                 f'<rect x="{x:.1f}" y="{py}" width="{pw:.0f}" height="25" rx="12.5" fill="{t["pill_fill"]}" '
                 f'fill-opacity="{t["pill_op"]}" stroke="{col}" stroke-opacity=".55"/>'
                 f'<circle cx="{x + 13:.1f}" cy="{py + 12.5}" r="3.2" fill="{col}"/>'
                 f'<text x="{x + 23:.1f}" y="{py + 17}" font-size="13" fill="{t["text"]}" textLength="{len(s) * pc:.1f}" '
                 f'lengthAdjust="spacingAndGlyphs">{escape(s)}</text></g></g>')
        x += pw + 8
    ly = cb - 26
    assert py + 25 < ly - 28, "stack pills overlap the links"
    o.append(f'<line x1="{X0}" y1="{ly - 26}" x2="{XR}" y2="{ly - 26}" stroke="{t["stroke"]}" stroke-opacity="{t["stroke_op"]}"/>')
    lc, gapw = 8.6, 36
    widths = [len(tg) * 7.4 + 14 + 8 + len(lk) * lc for tg, lk in LINKS]
    x = (X0 + XR) / 2 - (sum(widths) + gapw * (len(LINKS) - 1)) / 2
    for i, (tg, lk) in enumerate(LINKS):
        tw = len(tg) * 7.4 + 14
        o.append(f'<g>{reveal(3.0 + i * 0.12)}<rect x="{x:.1f}" y="{ly - 15}" width="{tw:.1f}" height="21" rx="5" '
                 f'fill="none" stroke="{t["a2"]}" stroke-opacity=".6"/>'
                 f'<text x="{x + tw / 2:.1f}" y="{ly}" font-size="11.5" fill="{t["a2"]}" text-anchor="middle" '
                 f'font-weight="700">{tg}</text>'
                 f'<text x="{x + tw + 8:.1f}" y="{ly}" font-size="14.5" fill="{t["muted"]}" '
                 f'textLength="{len(lk) * lc:.1f}" lengthAdjust="spacingAndGlyphs">{escape(lk)}</text></g>')
        x += widths[i] + gapw
    o.append("</g>")
    o.append("</svg>")
    return "\n".join(o)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("photo")
    ap.add_argument("--avatar")
    args = ap.parse_args()
    img, fg = cutout(args.photo)
    for theme in THEMES:
        rows = ascii_rows(img, fg, invert=(theme == "light"))
        (ROOT / f"{theme}.svg").write_text(build_svg(theme, rows), encoding="utf-8")
    if args.avatar:
        avatar(img, fg, args.avatar)


if __name__ == "__main__":
    main()
