#!/usr/bin/env python3
"""Pose line-drawing generator for workout posters.

Strike a pose in front of the webcam and get a clean black-outline drawing in
the style of hand-drawn exercise illustrations: white mannequin, grey floor
shadow, tank top / shorts. Capture one or two poses (start / end position) and
save them as an editable SVG plus a PNG.

    python pose_drawing.py                      # live webcam, interactive
    python pose_drawing.py --image a.jpg b.jpg  # draw from photos, no window

How it works: MediaPipe finds 33 body landmarks, every body part is rebuilt as
a smooth tapered shape, parts are layered by depth (near limbs cover far ones)
and the outline of each layer is the union of its shapes.
"""
from __future__ import annotations

import argparse
import datetime
import math
import os
import sys
import time
from dataclasses import dataclass, field

import cv2
import mediapipe as mp
import numpy as np
from shapely import affinity
from shapely.geometry import LineString, MultiPoint, Point, Polygon
from shapely.ops import unary_union

mp_pose = mp.solutions.pose

# MediaPipe pose landmark indices
NOSE, L_EYE, R_EYE, L_EAR, R_EAR, MOUTH_L, MOUTH_R = 0, 2, 5, 7, 8, 9, 10
L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 11, 12, 13, 14, 15, 16
L_PINKY, R_PINKY, L_INDEX, R_INDEX, L_THUMB, R_THUMB = 17, 18, 19, 20, 21, 22
L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE = 23, 24, 25, 26, 27, 28
L_HEEL, R_HEEL, L_TOE, R_TOE = 29, 30, 31, 32

INK, PAPER, SHADOW = "#000000", "#FFFFFF", "#B0B0B0"

# Body shape is expressed in "u": the shoulder-to-hip length of the person.
# t values run along the spine, 0 = shoulders, 1 = hips.
STYLES = {
    "male": dict(waist=0.86, hem=None, shorts_len=0.55, flare=1.22, limb=1.0, head=1.0, ponytail=False),
    "female": dict(waist=0.80, hem=0.56, shorts_len=0.36, flare=1.04, limb=0.93, head=0.96, ponytail=True),
}
STYLE_ORDER = list(STYLES)
LAYOUTS = ["auto", "row", "column"]

PART_NAMES = {
    NOSE: "head", L_SH: "L shoulder", R_SH: "R shoulder", L_EL: "L elbow", R_EL: "R elbow",
    L_WR: "L wrist", R_WR: "R wrist", L_HIP: "L hip", R_HIP: "R hip", L_KNEE: "L knee",
    R_KNEE: "R knee", L_ANKLE: "L ankle", R_ANKLE: "R ankle",
}


# --------------------------------------------------------------------------- geometry helpers

def unit(v, fallback=(0.0, 1.0)):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.array(fallback, dtype=float)


def perp(v):
    return np.array([-v[1], v[0]])


def aligned(v, ref):
    return v if np.dot(v, ref) >= 0 else -v


def polys(geom):
    if geom.is_empty:
        return []
    if geom.geom_type == "Polygon":
        return [geom]
    return [g for g in getattr(geom, "geoms", []) if g.geom_type == "Polygon"]


def lines(geom):
    if geom.is_empty:
        return []
    if geom.geom_type == "LineString":
        return [geom]
    return [g for g in getattr(geom, "geoms", []) if g.geom_type == "LineString"]


def closing(geom, k):
    return geom.buffer(k, 8).buffer(-k, 8)


def opening(geom, k):
    return geom.buffer(-k, 8).buffer(k, 8)


def limb(p0, p1, r0, r1, bulge=0.0, peak=0.4, cap0=True, cap1=True):
    """Tapered capsule from p0 to p1 with an optional muscle bulge."""
    d = p1 - p0
    length = np.linalg.norm(d)
    if length < 1e-6:
        return Point(*p0).buffer(max(r0, r1), 12)
    t = d / length
    n = perp(t)
    ts = np.linspace(0, 1, 24)
    k = math.log(0.5) / math.log(peak)
    radius = r0 + (r1 - r0) * ts + bulge * np.sin(np.pi * ts ** k)
    centers = p0 + np.outer(ts * length, t)
    left = centers + np.outer(radius, n)
    right = centers - np.outer(radius, n)
    phi = np.linspace(0, np.pi, 12)[1:-1]
    empty = np.empty((0, 2))
    cap_end = p1 + r1 * (np.outer(np.cos(phi), n) + np.outer(np.sin(phi), t)) if cap1 else empty
    cap_start = p0 - r0 * (np.outer(np.cos(phi), n) + np.outer(np.sin(phi), t)) if cap0 else empty
    return Polygon(np.vstack([left, cap_end, right[::-1], cap_start])).buffer(0)


def egg(center, up, a, b, n=64):
    """Head outline: an oval that is a little wider at the top than at the jaw."""
    side = perp(up)
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = a * np.cos(th) * (1 + 0.10 * np.sin(th))
    y = b * np.sin(th)
    return Polygon(center + np.outer(x, side) + np.outer(y, up))


def quad_bezier(p0, c, p2, n=24):
    s = np.linspace(0, 1, n)[:, None]
    return (1 - s) ** 2 * p0 + 2 * (1 - s) * s * c + s ** 2 * p2


def ellipse_poly(cx, cy, rx, ry, n=96):
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return Polygon(np.column_stack([cx + rx * np.cos(th), cy + ry * np.sin(th)]))


# --------------------------------------------------------------------------- pose -> figure

@dataclass
class Item:
    kind: str            # "shape" (white fill + outline), "line" (outline only), "shadow"
    geom: object


@dataclass
class Figure:
    items: list = field(default_factory=list)
    bounds: tuple = (0, 0, 1, 1)
    unit: float = 1.0


def estimate_unit(xy):
    """Torso-length scale of the person, robust to a foreshortened torso."""
    def dist(a, b):
        return float(np.linalg.norm(xy[a] - xy[b]))
    est = [float(np.linalg.norm((xy[L_SH] + xy[R_SH]) / 2 - (xy[L_HIP] + xy[R_HIP]) / 2))]
    for a, b, ratio in [(L_SH, L_EL, .58), (R_SH, R_EL, .58), (L_EL, L_WR, .50), (R_EL, R_WR, .50),
                        (L_HIP, L_KNEE, .83), (R_HIP, R_KNEE, .83), (L_KNEE, L_ANKLE, .83), (R_KNEE, R_ANKLE, .83)]:
        est.append(dist(a, b) / ratio)
    return max(float(np.percentile(est, 70)), 20.0)


def build_figure(P, style="male", flip_depth=False):
    """Turn a (33, 4) array of [x_px, y_px, z_px, visibility] into drawable layers."""
    st = STYLES[style]
    xy = P[:, :2].astype(float)
    z = P[:, 2].astype(float) * (-1.0 if flip_depth else 1.0)
    vis = P[:, 3]
    u = estimate_unit(xy)
    ls = st["limb"]
    zm = lambda ids: float(np.mean(z[list(ids)]))

    S = (xy[L_SH] + xy[R_SH]) / 2
    H = (xy[L_HIP] + xy[R_HIP]) / 2
    spine = H - S
    d = unit(spine)
    up = -d
    e = perp(d)
    sep_s = float(np.linalg.norm(xy[L_SH] - xy[R_SH]))
    sep_h = float(np.linalg.norm(xy[L_HIP] - xy[R_HIP]))
    es = aligned(unit(xy[L_SH] - xy[R_SH]), e) if sep_s > .3 * u else e
    eh = aligned(unit(xy[L_HIP] - xy[R_HIP]), e) if sep_h > .2 * u else e
    front = float(np.clip((sep_s / u - .35) / .35, 0, 1))   # 1 = facing the camera
    ch = max(sep_s / 2 - .04 * u, .20 * u)                   # chest half width
    hh = max(sep_h / 2 + .04 * u, .18 * u)                   # hip half width
    wmid = .9 * (ch + hh) / 2

    def edge(t, side):
        ev = unit(es * (1 - t) + eh * t)
        return S + spine * t + side * ev * np.interp(t, [0, .55, 1.0], [ch, wmid, hh])

    layers = []   # (sort key, [(kind, geom), ...]); drawn far to near (largest key first)

    # --- torso (tank top / crop top)
    left = [edge(t, +1) for t in (0, .3, .55, .8, 1.04)]
    right = [edge(t, -1) for t in (0, .3, .55, .8, 1.04)]
    ring = [left[0], S + es * .14 * u + up * .055 * u, S - es * .14 * u + up * .055 * u,
            right[0], *right[1:], *left[:0:-1]]
    torso = opening(Polygon(ring).buffer(0), .05 * u)
    torso_items = [("shape", torso)]
    if front > .3:
        neck = quad_bezier(S - es * .115 * u + up * .045 * u, S + d * .30 * u, S + es * .115 * u + up * .045 * u)
        torso_items.append(("line", LineString(neck)))
        for side in (+1, -1):
            p0 = S + side * es * ch * .62 + up * .02 * u
            p2 = S + spine * .34 + side * unit(es * .66 + eh * .34) * ch * .98
            torso_items.append(("line", LineString(quad_bezier(p0, (p0 + p2) / 2 - side * es * .05 * u, p2))))
    if st["hem"]:
        c = S + spine * st["hem"]
        torso_items.append(("line", LineString([c + e * 4 * u, c - e * 4 * u])))
    tz = zm([L_SH, R_SH, L_HIP, R_HIP])
    layers.append((tz, torso_items, torso))

    # --- legs: shoe + thigh + shin, one outline per leg
    for hip, knee, ankle, heel, toe in [(L_HIP, L_KNEE, L_ANKLE, L_HEEL, L_TOE), (R_HIP, R_KNEE, R_ANKLE, R_HEEL, R_TOE)]:
        thigh = limb(xy[hip], xy[knee], .16 * u * ls, .10 * u * ls, bulge=.014 * u, peak=.35)
        shin = limb(xy[knee], xy[ankle], .095 * u * ls, .058 * u * ls, bulge=.022 * u, peak=.3)
        shoe = MultiPoint([xy[ankle], xy[heel], xy[toe]]).convex_hull.buffer(.045 * u, 8)
        geom = closing(unary_union([thigh, shin, shoe]), .03 * u)
        layers.append((1e5 + zm([hip, knee, ankle]), [("shape", geom)], geom))

    # --- shorts: pelvis + one tube per thigh, cut off flat at the waist
    tw = st["waist"]
    Wm = S + spine * tw
    big = 8 * u
    below = Polygon([Wm + e * big, Wm - e * big, Wm - e * big + d * big, Wm + e * big + d * big])
    pelvis = Polygon([edge(tw, +1), edge(tw, -1), edge(1.06, -1), edge(1.06, +1)])
    tubes, hem_centers = [], []
    for hip, knee in [(L_HIP, L_KNEE), (R_HIP, R_KNEE)]:
        frac = st["shorts_len"]
        p1 = xy[hip] + (xy[knee] - xy[hip]) * frac
        r0 = .15 * u * ls * st["flare"]
        r1 = (.15 - .058 * frac) * u * ls * st["flare"] * 1.04
        tubes.append(limb(xy[hip], p1, r0, r1, cap1=False))
        hem_centers.append(p1)
    shorts = closing(unary_union([pelvis, *tubes]), .03 * u).intersection(below).buffer(0)
    shorts_items = [("shape", shorts)]
    if sep_h > .25 * u:
        seam = LineString([S + spine * 1.06, (hem_centers[0] + hem_centers[1]) / 2]).intersection(shorts)
        shorts_items += [("line", g) for g in lines(seam)]
    layers.append((tz - 1e-3, shorts_items, shorts))

    # --- arms: upper arm + forearm + fist, one outline per arm
    for sh, el, wr, pinky, index, thumb in [(L_SH, L_EL, L_WR, L_PINKY, L_INDEX, L_THUMB),
                                            (R_SH, R_EL, R_WR, R_PINKY, R_INDEX, R_THUMB)]:
        upper = limb(xy[sh], xy[el], .10 * u * ls, .072 * u * ls, bulge=.014 * u, peak=.4)
        fore = limb(xy[el], xy[wr], .07 * u * ls, .05 * u * ls, bulge=.014 * u, peak=.3)
        hand = MultiPoint([xy[wr], xy[index], xy[pinky], xy[thumb]]).convex_hull.buffer(.035 * u, 8)
        geom = closing(unary_union([upper, fore, hand]), .025 * u)
        layers.append((zm([sh, el, wr]) - .02 * u, [("shape", geom)], geom))

    # --- head: oval, neck, optional ponytail
    ears = [i for i in (L_EAR, R_EAR) if vis[i] > .4]
    eye_mid = (xy[L_EYE] + xy[R_EYE]) / 2
    mouth_mid = (xy[MOUTH_L] + xy[MOUTH_R]) / 2
    face_ok = min(vis[L_EYE], vis[R_EYE], vis[MOUTH_L], vis[MOUTH_R]) > .4
    C = xy[ears].mean(axis=0) if ears else xy[NOSE] + up * .06 * u
    hup = unit(eye_mid - mouth_mid) if face_ok and np.linalg.norm(eye_mid - mouth_mid) > 1e-6 else unit(C - S, fallback=up)
    ear_sep = float(np.linalg.norm(xy[L_EAR] - xy[R_EAR])) if len(ears) == 2 else 0.0
    a = np.interp(ear_sep / u, [0, .35], [.19, .15]) * u * st["head"]
    b = .215 * u * st["head"]
    head = egg(C + hup * .01 * u, hup, a, b)
    neck = limb(S + up * .02 * u, C - hup * b * .7, .08 * u, .068 * u)
    hz = min(zm([NOSE]), tz) - .12 * u
    layers.append((tz + 1, [("shape", neck)], neck))
    layers.append((hz, [("shape", head)], head))
    if st["ponytail"]:
        fwd = xy[NOSE] - C
        fwd = fwd - hup * np.dot(fwd, hup)
        if ear_sep < .2 * u and np.linalg.norm(fwd) > .05 * u:
            back = -unit(fwd)
            root = C + hup * b * .5 + back * a * .8
            tip = root + unit(back * .5 + np.array([0.0, 1.0])) * .34 * u
            tail = limb(root, tip, .05 * u, .012 * u, bulge=.01 * u)
            layers.append((hz + 1, [("shape", tail)], tail))

    # draw far -> near; ties keep insertion order
    layers.sort(key=lambda l: -l[0])
    items = []
    for _, layer_items, geom in layers:
        for kind, g in layer_items:
            if kind == "line":
                g = g.intersection(geom.buffer(.5)) if geom is not None else g
                items += [Item("line", seg) for seg in lines(g)]
            elif not g.is_empty:
                items.append(Item("shape", g))

    body = unary_union([it.geom for it in items if it.kind == "shape"])
    x0, y0, x1, y1 = body.bounds
    fig_w, fig_h = x1 - x0, y1 - y0
    rx = max(.55 * fig_w, .30 * fig_h)
    ry = max(min(.4 * rx, .25 * fig_h), .08 * u)
    shadow = ellipse_poly((x0 + x1) / 2, y1 - .45 * ry, rx, ry)
    items.insert(0, Item("shadow", shadow))
    return Figure(items=items, bounds=shadow.union(body).bounds, unit=u)


# --------------------------------------------------------------------------- layout, scene

@dataclass
class Scene:
    groups: list          # one list of ops per pose; op = (kind, [ndarray rings])
    width: int
    height: int
    line_width: float


def wobble(ring, amp, u, rng, closed=True):
    """Gentle hand-drawn waviness along the normal; closed rings stay closed."""
    if amp <= 0 or len(ring) < 4:
        return ring
    pts = ring
    seg = np.linalg.norm(np.diff(np.vstack([pts, pts[:1]]) if closed else pts, axis=0), axis=1)
    per = seg.sum()
    if per < 1:
        return ring
    s = np.concatenate([[0], np.cumsum(seg)])[:len(pts)] / per
    tangent = (np.roll(pts, -1, axis=0) - np.roll(pts, 1, axis=0)) if closed else np.gradient(pts, axis=0)
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-9)
    disp = np.zeros(len(pts))
    for wavelength, weight in [(.42 * u, .65), (.19 * u, .35)]:
        k = max(1, int(round(per / wavelength)))
        disp += weight * np.sin(2 * np.pi * k * s + rng.uniform(0, 2 * np.pi))
    return pts + normal * (amp * disp)[:, None]


def geom_rings(geom):
    out = []
    for p in polys(geom):
        out.append([np.array(p.exterior.coords)[:-1]] + [np.array(r.coords)[:-1] for r in p.interiors])
    return out


def make_scene(poses, style="male", flip_depth=False, mirror=False, layout="auto",
               size=(2000, 1400), wobble_amount=1.0, line_scale=1.0):
    W, H = size
    figs = [build_figure(P, style, flip_depth) for P in poses if P is not None]
    sizes = [(f.bounds[2] - f.bounds[0], f.bounds[3] - f.bounds[1]) for f in figs]
    margin, gap = .07 * W, .06 * W
    avail_w, avail_h = W - 2 * margin, H - 2 * margin
    s_row = min((avail_w - gap * (len(figs) - 1)) / sum(w for w, _ in sizes), avail_h / max(h for _, h in sizes))
    s_col = min(avail_w / max(w for w, _ in sizes), (avail_h - gap * (len(figs) - 1)) / sum(h for _, h in sizes))
    mode = layout if layout != "auto" else ("row" if s_row >= s_col else "column")
    s = s_row if mode == "row" else s_col

    # top-left corner of each figure on the canvas
    origins = []
    if mode == "row":
        x = (W - (sum(w for w, _ in sizes) * s + gap * (len(figs) - 1))) / 2
        bottom = H / 2 + max(h for _, h in sizes) * s / 2
        for w, h in sizes:
            origins.append((x, bottom - h * s))
            x += w * s + gap
    else:
        y = (H - (sum(h for _, h in sizes) * s + gap * (len(figs) - 1))) / 2
        for w, h in sizes:
            origins.append(((W - w * s) / 2, y))
            y += h * s + gap

    u_canvas = float(np.mean([f.unit for f in figs])) * s
    lw = float(np.clip(.02 * u_canvas, 4, 12)) * (W / 2000) * line_scale
    groups = []
    for gi, (fig, (ox, oy)) in enumerate(zip(figs, origins)):
        x0, y0, x1, y1 = fig.bounds
        m = [-s, 0, 0, s, ox + x1 * s, oy - y0 * s] if mirror else [s, 0, 0, s, ox - x0 * s, oy - y0 * s]
        rng = np.random.default_rng(1000 + gi)
        ops = []
        for item in fig.items:
            g = affinity.affine_transform(item.geom, m)
            if item.kind == "line":
                for ln in lines(g):
                    pts = wobble(np.array(ln.coords), .0045 * u_canvas * wobble_amount, u_canvas, rng, closed=False)
                    ops.append(("line", [pts]))
            else:
                for rings in geom_rings(g):
                    if item.kind == "shape":
                        rings = [wobble(r, .0045 * u_canvas * wobble_amount, u_canvas, rng) for r in rings]
                    ops.append((item.kind, rings))
        groups.append(ops)
    return Scene(groups, W, H, lw)


# --------------------------------------------------------------------------- output: PNG, SVG

def rasterize(scene, scale=1.0, ss=3):
    """Anti-aliased render with OpenCV (supersampled), no external libraries needed."""
    k = scale * ss
    img = np.full((int(scene.height * k), int(scene.width * k), 3), 255, np.uint8)
    fx = lambda r: np.round(r * k * 16).astype(np.int32)
    thick = max(1, int(round(scene.line_width * k)))
    for ops in scene.groups:
        for kind, rings in ops:
            if kind == "shadow":
                cv2.fillPoly(img, [fx(r) for r in rings], (176, 176, 176), cv2.LINE_AA, 4)
            elif kind == "shape":
                cv2.fillPoly(img, [fx(r) for r in rings], (255, 255, 255), cv2.LINE_AA, 4)
                for r in rings:
                    cv2.polylines(img, [fx(r)], True, (0, 0, 0), thick, cv2.LINE_AA, 4)
            else:
                cv2.polylines(img, [fx(rings[0])], False, (0, 0, 0), thick, cv2.LINE_AA, 4)
    return cv2.resize(img, (int(scene.width * scale), int(scene.height * scale)), interpolation=cv2.INTER_AREA)


def _path(ring, closed):
    simple = np.array(LineString(np.vstack([ring, ring[:1]]) if closed else ring).simplify(.4).coords)
    if closed:
        simple = simple[:-1]
    d = "M" + " L".join(f"{x:.1f} {y:.1f}" for x, y in simple)
    return d + (" Z" if closed else "")


def to_svg(scene):
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{scene.width}" height="{scene.height}" '
           f'viewBox="0 0 {scene.width} {scene.height}">',
           f'<rect id="background" width="{scene.width}" height="{scene.height}" fill="{PAPER}"/>']
    stroke = (f'stroke="{INK}" stroke-width="{scene.line_width:.2f}" '
              'stroke-linejoin="round" stroke-linecap="round"')
    for gi, ops in enumerate(scene.groups, 1):
        out.append(f'<g id="pose-{gi}">')
        for kind, rings in ops:
            d = " ".join(_path(r, kind != "line") for r in rings)
            if kind == "shadow":
                out.append(f'<path d="{d}" fill="{SHADOW}"/>')
            elif kind == "shape":
                out.append(f'<path d="{d}" fill="{PAPER}" fill-rule="evenodd" {stroke}/>')
            else:
                out.append(f'<path d="{d}" fill="none" {stroke}/>')
        out.append("</g>")
    out.append("</svg>")
    return "\n".join(out)


def save_scene(scene, out_dir, name=None):
    os.makedirs(out_dir, exist_ok=True)
    name = name or datetime.datetime.now().strftime("pose_%Y%m%d_%H%M%S")
    base = os.path.join(out_dir, name)
    with open(base + ".svg", "w") as f:
        f.write(to_svg(scene))
    cv2.imwrite(base + ".png", rasterize(scene))
    return base + ".svg", base + ".png"


# --------------------------------------------------------------------------- pose detection

def landmarks_to_array(result, w, h):
    if not result.pose_landmarks:
        return None
    return np.array([[l.x * w, l.y * h, l.z * w, l.visibility] for l in result.pose_landmarks.landmark])


def detect_image(path):
    img = cv2.imread(path)
    if img is None:
        sys.exit(f"Cannot read image: {path}")
    with mp_pose.Pose(static_image_mode=True, model_complexity=2, min_detection_confidence=0.3) as pose:
        P = landmarks_to_array(pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), img.shape[1], img.shape[0])
    if P is None:
        sys.exit(f"No person found in {path}")
    return P


def missing_parts(P, w, h):
    out = []
    for i, name in PART_NAMES.items():
        x, y = P[i, 0], P[i, 1]
        if P[i, 3] < .5 or not (0 <= x < w and 0 <= y < h):
            out.append(name)
    return out


# --------------------------------------------------------------------------- interactive window

WINDOW = "Pose drawing"


def put_lines(img, text_lines, x, y, scale=.6, color=(255, 255, 255), pad=6):
    for i, text in enumerate(text_lines):
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        yy = y + i * (th + 2 * pad)
        cv2.rectangle(img, (x - pad, yy - th - pad), (x + tw + pad, yy + pad), (0, 0, 0), -1)
        cv2.putText(img, text, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def capture_pose(cap, pose, frames=12):
    """Median of several consecutive detections, to cancel out jitter."""
    samples = []
    for _ in range(frames * 3):
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        P = landmarks_to_array(pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), w, h)
        if P is not None:
            samples.append(P)
        if len(samples) >= frames:
            break
    return np.median(np.stack(samples), axis=0) if len(samples) >= 4 else None


def live_view(frame, P, state):
    h, w = frame.shape[:2]
    view = cv2.flip(frame, 1)
    if P is not None:
        pts = [(int(w - x), int(y)) for x, y, _, _ in P]
        for a, b in mp_pose.POSE_CONNECTIONS:
            cv2.line(view, pts[a], pts[b], (255, 255, 255), 2, cv2.LINE_AA)
        for p, v in zip(pts, P[:, 3]):
            cv2.circle(view, p, 4, (0, 200, 0) if v > .5 else (0, 0, 255), -1, cv2.LINE_AA)
    names = "AB"
    for i in range(2):
        if state["poses"][i] is not None:
            put_lines(view, [f"Pose {names[i]} saved"], 14, 30 + i * 34, .6, (0, 255, 0))
    if state["countdown"] is not None:
        n = max(1, math.ceil(state["countdown"]))
        cv2.putText(view, str(n), (w // 2 - 60, h // 2 + 60), cv2.FONT_HERSHEY_SIMPLEX, 6, (0, 0, 0), 14, cv2.LINE_AA)
        cv2.putText(view, str(n), (w // 2 - 60, h // 2 + 60), cv2.FONT_HERSHEY_SIMPLEX, 6, (255, 255, 255), 6, cv2.LINE_AA)
    target = names[state["target"]]
    if P is None:
        status = ("No body detected - step back so your whole body is in view", (0, 0, 255))
    else:
        miss = missing_parts(P, w, h)
        status = (("Out of view: " + ", ".join(miss), (0, 0, 255)) if miss
                  else ("Whole body visible - ready", (0, 255, 0)))
    help_lines = [f"SPACE: capture pose {target} after {state['delay']:g}s timer",
                  "ENTER: finish   R: reset   Q: quit"]
    put_lines(view, [status[0]], 14, h - 88, .6, status[1])
    put_lines(view, help_lines, 14, h - 50, .55)
    if state["message"]:
        put_lines(view, [state["message"]], 14, h - 128, .6, (0, 255, 255))
    return view


def review_view(scene_img, state):
    h, w = scene_img.shape[:2]
    bar = np.full((84, w, 3), 30, np.uint8)
    put_lines(bar, [f"S: style ({state['style']})   F: flip depth   X: mirror   L: layout ({state['layout']})   1/2: retake pose",
                    "ENTER: save SVG + PNG   R: start over   Q: quit"], 12, 28, .55)
    if state["message"]:
        put_lines(bar, [state["message"]], 12, 76, .55, (0, 255, 255))
    return np.vstack([scene_img, bar])


def run_interactive(args):
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        sys.exit("Cannot open the camera. On macOS, allow camera access for your terminal in "
                 "System Settings > Privacy & Security > Camera, then run again.")
    pose = mp_pose.Pose(model_complexity=1, smooth_landmarks=True,
                        min_detection_confidence=.5, min_tracking_confidence=.5)
    state = dict(poses=[None, None], target=0, countdown=None, delay=args.delay, message="",
                 style=args.style, layout=args.layout, flip=False, mirror=args.mirror)
    mode, t_end, cache_key, cache_img, scene = "live", 0.0, None, None, None
    message_until = 0.0

    def say(text, seconds=4):
        nonlocal message_until
        state["message"], message_until = text, time.time() + seconds

    say("Stand back so your whole body is visible")
    while True:
        if state["message"] and time.time() > message_until:
            state["message"] = ""
        if mode == "review":
            key_now = (state["style"], state["flip"], state["mirror"], state["layout"],
                       tuple(id(p) for p in state["poses"]))
            if key_now != cache_key:
                scene = make_scene([p for p in state["poses"] if p is not None], state["style"], state["flip"],
                                   state["mirror"], state["layout"], wobble_amount=args.wobble,
                                   line_scale=args.line_scale)
                cache_img, cache_key = rasterize(scene, .5, 2), key_now
            cv2.imshow(WINDOW, review_view(cache_img, state))
            key = cv2.waitKey(30) & 0xFF
        else:
            ok, frame = cap.read()
            if not ok:
                continue
            h, w = frame.shape[:2]
            P = landmarks_to_array(pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), w, h)
            state["countdown"] = max(0.0, t_end - time.time()) if mode == "countdown" else None
            cv2.imshow(WINDOW, live_view(frame, P, state))
            key = cv2.waitKey(1) & 0xFF
            if mode == "countdown" and time.time() >= t_end:
                got = capture_pose(cap, pose)
                mode = "live"
                if got is None:
                    say("Could not see you - try again")
                else:
                    state["poses"][state["target"]] = got
                    state["target"] = 1 if state["target"] == 0 and state["poses"][1] is None else state["target"]
                    if all(p is not None for p in state["poses"]):
                        mode, cache_key = "review", None
                    else:
                        say("Pose A saved. Press SPACE for pose B, or ENTER to finish with one pose")

        if key in (ord("q"), 27):
            break
        if mode == "live":
            if key == ord(" "):
                mode, t_end = "countdown", time.time() + args.delay
            elif key == 13 and state["poses"][0] is not None:
                mode, cache_key = "review", None
            elif key == ord("r"):
                state["poses"], state["target"] = [None, None], 0
                say("Reset")
        elif mode == "countdown" and key == ord(" "):
            mode = "live"
            say("Cancelled")
        elif mode == "review":
            if key == ord("s"):
                state["style"] = STYLE_ORDER[(STYLE_ORDER.index(state["style"]) + 1) % len(STYLE_ORDER)]
            elif key == ord("f"):
                state["flip"] = not state["flip"]
            elif key == ord("x"):
                state["mirror"] = not state["mirror"]
            elif key == ord("l"):
                state["layout"] = LAYOUTS[(LAYOUTS.index(state["layout"]) + 1) % len(LAYOUTS)]
            elif key in (ord("1"), ord("2")):
                i = key - ord("1")
                if i == 1 and state["poses"][1] is None:
                    pass
                state["target"], mode = i, "live"
                state["poses"][i] = None
                say(f"Retaking pose {'AB'[i]} - press SPACE")
            elif key == ord("r"):
                state["poses"], state["target"], mode = [None, None], 0, "live"
            elif key == 13:
                final = make_scene([p for p in state["poses"] if p is not None], state["style"], state["flip"],
                                   state["mirror"], state["layout"], wobble_amount=args.wobble,
                                   line_scale=args.line_scale)
                svg, png = save_scene(final, args.out, args.name)
                say(f"Saved {svg} and {png}", 8)
                print(f"Saved {svg}\nSaved {png}")
    cap.release()
    cv2.destroyAllWindows()


# --------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description="Turn webcam poses into workout-poster line drawings.")
    ap.add_argument("--image", nargs="+", metavar="FILE", help="use 1-2 photos instead of the webcam (no window)")
    ap.add_argument("--camera", type=int, default=0, help="webcam index (default 0)")
    ap.add_argument("--delay", type=float, default=5, help="countdown seconds before each capture (default 5)")
    ap.add_argument("--style", choices=STYLE_ORDER, default="male")
    ap.add_argument("--layout", choices=LAYOUTS, default="auto", help="two poses side by side or stacked")
    ap.add_argument("--mirror", action="store_true", help="flip the drawing left-right")
    ap.add_argument("--wobble", type=float, default=1.0, help="hand-drawn waviness, 0 = perfectly smooth")
    ap.add_argument("--line-scale", type=float, default=1.0, help="line thickness multiplier")
    ap.add_argument("--size", default="2000x1400", help="output size in pixels, WIDTHxHEIGHT")
    ap.add_argument("--out", default="output", help="output folder")
    ap.add_argument("--name", help="output file name without extension")
    args = ap.parse_args()

    if args.image:
        if len(args.image) > 2:
            sys.exit("Give at most two images (start and end pose).")
        W, H = (int(v) for v in args.size.lower().split("x"))
        scene = make_scene([detect_image(p) for p in args.image], args.style, False, args.mirror, args.layout,
                           (W, H), args.wobble, args.line_scale)
        for path in save_scene(scene, args.out, args.name):
            print("Saved", path)
    else:
        run_interactive(args)


if __name__ == "__main__":
    main()
