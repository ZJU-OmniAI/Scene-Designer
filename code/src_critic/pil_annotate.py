"""Annotate a rendered top-down PNG in the scenesmith style.

Given:
  * a rendered top-down PNG (Blender ortho or persp top-down),
  * the world-space camera meta returned by blender_render.py,
  * the room geometry + furniture poses (our `Scene` model),
  * an optional list of `highlight_ids` for the changed object(s),

this draws on top of the image:
  * a 5x5 grid of red coordinate dots labelled `(x, y)`,
  * blue wall-surface labels (north_wall, south_wall, east_wall, west_wall),
  * green door labels and orange window labels,
  * a black/red arrow on each furniture rectangle showing its FRONT direction,
  * a blue object_id label for each furniture, *placed outside* its footprint
    with a thin leader line — so the photoreal mesh below is never hidden,
  * a thick red border + FRONT/BACK markers around the changed object(s).

The label layout uses a tiny greedy collision-avoidance pass that tries 8
positions around each object and falls back to "centre" only if every
exterior slot is occupied.
"""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .scene_io import Furniture, Scene


_RED = (210, 48, 48)
_BLUE_LABEL = (77, 153, 255)
_GREEN_LABEL = (34, 139, 34)
_ORANGE_LABEL = (255, 165, 0)
_WHITE = (255, 255, 255)
_BLACK = (10, 10, 10)
_LEADER_GREY = (60, 60, 60)
_LABEL_PADDING = 4


def _try_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    ]
    for c in candidates:
        if Path(c).exists():
            try:
                return ImageFont.truetype(c, size)
            except Exception:
                continue
    return ImageFont.load_default()


class _Projector:
    """Project world (x, y, z=0) to pixel (px, py) for either an orthographic
    top-down or a perspective top-down camera centred at (cx, cy, distance)
    looking straight down along -z."""

    def __init__(self, image_size, *, ortho_view_box=None, persp=None):
        self.W, self.H = image_size
        self._ortho_view_box = ortho_view_box
        self._persp = persp
        if ortho_view_box is not None:
            x0, y0, x1, y1 = ortho_view_box
            self.sx = self.W / (x1 - x0)
            self.sy = self.H / (y1 - y0)
            self._mode = "ortho"
        else:
            assert persp is not None
            self.cx, self.cy = persp["center"]
            self.distance = persp["distance"]
            self.fov = persp["fov_radians"]
            self.focal_px = (self.W / 2.0) / math.tan(self.fov / 2.0)
            self._mode = "persp"

    def __call__(self, wx, wy, wz: float = 0.0):
        if self._mode == "ortho":
            x0, y0, x1, y1 = self._ortho_view_box
            u = (wx - x0) * self.sx
            v = (y1 - wy) * self.sy
            return u, v
        local_x = wx - self.cx
        local_y = wy - self.cy
        local_z_forward = self.distance - wz
        if local_z_forward <= 1e-6:
            return -1e6, -1e6
        u = local_x / local_z_forward
        v = local_y / local_z_forward
        px = self.W / 2.0 + u * self.focal_px
        py = self.H / 2.0 - v * self.focal_px
        return px, py

    def world_dx_to_px_at_floor(self, dx: float) -> float:
        if self._mode == "ortho":
            return dx * self.sx
        return dx * self.focal_px / self.distance


def _measure(draw, font, text):
    bb = draw.textbbox((0, 0), text, font=font)
    return bb[2] - bb[0], bb[3] - bb[1]


def _label_rect(cx: float, cy: float, tw: int, th: int, padding: int = _LABEL_PADDING):
    return (
        int(cx - tw / 2 - padding),
        int(cy - th / 2 - padding),
        int(cx + tw / 2 + padding),
        int(cy + th / 2 + padding),
    )


def _rects_overlap(a, b, slack: int = 0) -> bool:
    return not (
        a[2] + slack < b[0]
        or b[2] + slack < a[0]
        or a[3] + slack < b[1]
        or b[3] + slack < a[1]
    )


def _inside_image(rect, image_size, margin: int = 2):
    W, H = image_size
    return (
        rect[0] >= margin and rect[1] >= margin
        and rect[2] <= W - margin and rect[3] <= H - margin
    )


def _aabb_of_polygon(pts):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (min(xs), min(ys), max(xs), max(ys))


def _draw_label_box(draw, font, text, rect, *, bg, fg=_WHITE, outline=None):
    """Render the box at the given rect (already padded)."""
    draw.rectangle(list(rect), fill=bg, outline=outline,
                   width=2 if outline else 0)
    bb = draw.textbbox((0, 0), text, font=font)
    tw = bb[2] - bb[0]
    th = bb[3] - bb[1]
    cx = (rect[0] + rect[2]) / 2.0
    cy = (rect[1] + rect[3]) / 2.0
    draw.text((cx - tw / 2, cy - th / 2 - bb[1]), text, fill=fg, font=font)


def _draw_arrow(draw, x0, y0, x1, y1, color, width=4, head=12):
    draw.line([(x0, y0), (x1, y1)], fill=color, width=width)
    angle = math.atan2(y1 - y0, x1 - x0)
    a = angle + math.pi - math.radians(28)
    b = angle + math.pi + math.radians(28)
    p1 = (x1 + head * math.cos(a), y1 + head * math.sin(a))
    p2 = (x1 + head * math.cos(b), y1 + head * math.sin(b))
    draw.polygon([(x1, y1), p1, p2], fill=color)


def _candidate_positions(anchor_rect, label_size, image_size):
    """Yield (cx, cy) candidates outside `anchor_rect` for a label of the
    given (text_width, text_height). Tries 8 positions around the rect, then
    a couple of farther variants."""
    ax0, ay0, ax1, ay1 = anchor_rect
    acx = (ax0 + ax1) / 2.0
    acy = (ay0 + ay1) / 2.0
    tw, th = label_size
    pad = _LABEL_PADDING
    half_w = tw / 2.0 + pad
    half_h = th / 2.0 + pad
    gap = 6  # px gap between bbox and label
    base_offsets = [
        # (dx_from_anchor_centre, dy_from_anchor_centre)
        (0, -(ay1 - ay0) / 2.0 - half_h - gap),                  # north
        (0,  (ay1 - ay0) / 2.0 + half_h + gap),                  # south
        ( (ax1 - ax0) / 2.0 + half_w + gap, 0),                  # east
        (-((ax1 - ax0) / 2.0 + half_w + gap), 0),                # west
        ( (ax1 - ax0) / 2.0 + half_w + gap, -(ay1 - ay0) / 2.0 - half_h - gap),  # NE
        (-((ax1 - ax0) / 2.0 + half_w + gap), -(ay1 - ay0) / 2.0 - half_h - gap),  # NW
        ( (ax1 - ax0) / 2.0 + half_w + gap,  (ay1 - ay0) / 2.0 + half_h + gap),    # SE
        (-((ax1 - ax0) / 2.0 + half_w + gap),  (ay1 - ay0) / 2.0 + half_h + gap),  # SW
    ]
    for dx, dy in base_offsets:
        yield acx + dx, acy + dy
    # Extra "double-distance" candidates if everything closer is blocked
    for dx, dy in base_offsets:
        yield acx + dx * 1.6, acy + dy * 1.6
    # Final fallback: anchor centre
    yield acx, acy


def _layout_label(
    *,
    anchor_rect,
    label_size,
    image_size,
    blockers,
    extra_padding: int = 4,
):
    """Pick the first candidate position whose label rect (a) is inside the
    image and (b) does NOT overlap any rect in `blockers`. Returns the chosen
    (cx, cy) and the resulting label rect."""
    tw, th = label_size
    for cx, cy in _candidate_positions(anchor_rect, label_size, image_size):
        rect = _label_rect(cx, cy, tw, th)
        if not _inside_image(rect, image_size):
            continue
        if any(_rects_overlap(rect, b, slack=extra_padding) for b in blockers):
            continue
        return (cx, cy), rect
    # Forced fallback: the very last candidate (centre).
    cx, cy = (anchor_rect[0] + anchor_rect[2]) / 2.0, (anchor_rect[1] + anchor_rect[3]) / 2.0
    return (cx, cy), _label_rect(cx, cy, tw, th)


def _draw_grid_dots(draw, project, room, *, n_x=5, n_y=5):
    """Pass A — draw the small red coordinate dots only.

    The text labels for these dots are drawn LATER (in `_draw_grid_text`)
    after every other annotation has been placed so we can skip a label
    when it would collide with a furniture footprint or another label.
    """
    xs = [room.x_min + i * room.width / (n_x - 1) for i in range(n_x)]
    ys = [room.y_min + j * room.length / (n_y - 1) for j in range(n_y)]
    r = max(4, int(project.W * 0.005))
    coords = []
    for x in xs:
        for y in ys:
            px, py = project(x, y)
            draw.ellipse([px - r, py - r, px + r, py + r], fill=_RED)
            coords.append((x, y, px, py, r))
    return coords


def _draw_grid_text(draw, font, project, coords, blockers, furniture_rects):
    """Pass B — draw `(x, y)` text under each dot, but only if the dot is
    NOT inside (or right against) a furniture footprint AND the text rect
    doesn't collide with any previously-placed label."""
    placed = 0
    skipped = 0
    for (x, y, px, py, r) in coords:
        # Skip dots that fall inside any furniture footprint (with a tiny
        # padding so dots right at the bbox edge are also dropped — those
        # tend to overlap the object polygon outline).
        inside_furniture = any(
            (rect[0] - 2 <= px <= rect[2] + 2 and rect[1] - 2 <= py <= rect[3] + 2)
            for rect in furniture_rects
        )
        if inside_furniture:
            skipped += 1
            continue
        sx = f"{x:.2f}".rstrip("0").rstrip(".") or "0"
        sy = f"{y:.2f}".rstrip("0").rstrip(".") or "0"
        txt = f"({sx},{sy})"
        tw, th = _measure(draw, font, txt)
        # Try below the dot first, then above, then to the right, then left.
        for off_x, off_y in (
            (0, r + 2),       # below
            (0, -(r + 2 + th)),  # above
            (r + 4, -th / 2),    # right
            (-(r + 4 + tw), -th / 2),  # left
        ):
            tx = int(px + off_x - (tw / 2 if abs(off_x) < 1 else 0))
            ty = int(py + off_y)
            tx = max(2, min(tx, project.W - tw - 2))
            ty = max(2, min(ty, project.H - th - 2))
            rect = (tx, ty, tx + tw, ty + th)
            if any(_rects_overlap(rect, b, slack=1) for b in blockers):
                continue
            draw.text((tx, ty), txt, fill=_RED, font=font)
            blockers.append(rect)
            placed += 1
            break
        else:
            skipped += 1
    return placed, skipped


def _place_with_offsets(
    *, anchor_px, label_size, image_size, blockers,
    offsets_px, slack: int = 4,
):
    """Try anchor_px + each (dx, dy) in `offsets_px`; return the first that
    fits inside the image and does not collide with `blockers`.
    Returns (rect, (cx, cy)) or (None, None) if every candidate fails."""
    tw, th = label_size
    ax, ay = anchor_px
    for dx, dy in offsets_px:
        cx, cy = ax + dx, ay + dy
        rect = _label_rect(cx, cy, tw, th)
        if not _inside_image(rect, image_size):
            continue
        if any(_rects_overlap(rect, b, slack=slack) for b in blockers):
            continue
        return rect, (cx, cy)
    return None, None


def _opening_anchor_and_axis(room, *, direction: str, pos: float):
    """Return ((wx, wy), 'h' or 'v') for an opening on the named wall.
    The axis is the direction the label can slide along the wall."""
    if direction in ("north", "south"):
        wx = room.x_min + pos
        wy = room.y_max if direction == "north" else room.y_min
        return (wx, wy), "h"
    wy = room.y_min + pos
    wx = room.x_max if direction == "east" else room.x_min
    return (wx, wy), "v"


def _draw_openings(draw, font, project, room, blockers, sx_px, sy_px):
    """Place door/window labels FIRST (highest priority). For each, the
    label tries its natural anchor, then small slides along the wall axis,
    then small offsets perpendicular into the room."""
    placed = []  # (label, world_xy, axis, bg)
    for d in room.doors:
        direction = {"A": "north", "B": "south", "C": "east", "D": "west"}.get(
            d.get("boundary_label", "A"), "north")
        pos = float(d.get("position_exact", 0.0))
        anchor_w, axis = _opening_anchor_and_axis(room, direction=direction, pos=pos)
        placed.append((d.get("id", "door"), anchor_w, axis, direction, _GREEN_LABEL))
    for w in room.windows:
        direction = w.get("wall_direction") or {
            "A": "north", "B": "south", "C": "east", "D": "west"
        }.get(w.get("boundary_label", "A"), "north")
        pos = float(w.get("position_along_wall", 0.0))
        anchor_w, axis = _opening_anchor_and_axis(room, direction=direction, pos=pos)
        placed.append((w.get("id", "window"), anchor_w, axis, direction, _ORANGE_LABEL))

    for label, (wx, wy), axis, direction, bg in placed:
        px, py = project(wx, wy)
        tw, th = _measure(draw, font, label)
        # candidates: at the door, then ±along-wall (slide), then ±into-room
        slides = [0.0, -0.40, 0.40, -0.80, 0.80, -1.20, 1.20]
        into = [0.0, -0.40, -0.80, -1.20]
        offsets = []
        for s in slides:
            for io in into:
                if axis == "h":
                    dx = s * sx_px
                    dy = (io if direction == "north" else -io) * sy_px
                else:
                    dy = s * sy_px
                    dx = (io if direction == "east" else -io) * sx_px
                offsets.append((dx, dy))
        rect, _ = _place_with_offsets(
            anchor_px=(px, py),
            label_size=(tw, th),
            image_size=(project.W, project.H),
            blockers=blockers,
            offsets_px=offsets,
            slack=2,
        )
        if rect is None:
            # forced — draw at the natural anchor anyway, doors are critical
            rect = _label_rect(px, py, tw, th)
        _draw_label_box(draw, font, label, rect, bg=bg)
        blockers.append(rect)


def _draw_walls(draw, font, project, room, blockers, sx_px, sy_px):
    """Place wall-surface labels with the ability to slide along the wall.
    Walls are LOW priority — if no candidate fits, we drop the label entirely."""
    cx_w = (room.x_min + room.x_max) / 2.0
    cy_w = (room.y_min + room.y_max) / 2.0
    inset = 0.25
    items = [
        ("north_wall", (cx_w, room.y_max - inset), "h"),
        ("south_wall", (cx_w, room.y_min + inset), "h"),
        ("east_wall",  (room.x_max - inset, cy_w), "v"),
        ("west_wall",  (room.x_min + inset, cy_w), "v"),
    ]
    for label, (wx, wy), axis in items:
        px, py = project(wx, wy)
        tw, th = _measure(draw, font, label)
        # slide along wall axis, then small inward shifts
        slides = [0.0, -0.40, 0.40, -0.80, 0.80, -1.20, 1.20, -1.60, 1.60]
        into = [0.0, 0.30, 0.60, 0.90]  # away from the wall edge
        offsets = []
        for s in slides:
            for io in into:
                if axis == "h":
                    dx = s * sx_px
                    dy = io * sy_px * (-1 if "north" in label else 1)
                else:
                    dy = s * sy_px
                    dx = io * sx_px * (-1 if "east" in label else 1)
                offsets.append((dx, dy))
        rect, _ = _place_with_offsets(
            anchor_px=(px, py),
            label_size=(tw, th),
            image_size=(project.W, project.H),
            blockers=blockers,
            offsets_px=offsets,
            slack=2,
        )
        if rect is None:
            # drop the wall label rather than overlap something else
            continue
        _draw_label_box(draw, font, label, rect, bg=_BLUE_LABEL)
        blockers.append(rect)


def _draw_furniture_box_and_arrow(
    draw, project, f: Furniture,
    *,
    highlight: bool,
    is_rotation_highlight: bool = False,
    lateral_override: float | None = None,
):
    """Draw the (optional) red border + the orientation arrow.

    Highlight policy:
      * `highlight=False` → black arrow at default length, no border.
      * `highlight=True, is_rotation_highlight=True` → red border, RED arrow
        at the SAME default length (NOT extended like before), so the arrow
        flip is visible without an attention-grabbing oversized arrow.
      * `highlight=True, is_rotation_highlight=False` → red border, BLACK
        arrow (same as non-highlighted): the perturbation isn't a yaw
        change, so the arrow direction is irrelevant — drawing a red
        arrow would falsely imply orientation matters here.
    """
    corners_world = f.world_corners()
    px_corners = [project(*c) for c in corners_world]
    if highlight:
        draw.polygon(px_corners, outline=_RED, fill=None, width=6)

    cx_w, cy_w = f.tx, f.ty
    cx_px, cy_px = project(cx_w, cy_w)

    footprint_rect = _aabb_of_polygon(px_corners)

    # Skip arrow drawing for decorative floor coverings (rugs, carpets,
    # mats, doormats, runners). They lie flat on the floor with no
    # functional front; drawing an arrow on top of a rug just adds
    # clutter to the photoreal render. Other non-rotatable pieces
    # (round tables, planters, lamps) DO still get arrows — their yaw
    # is at least a stable per-piece reference even if not functional.
    from .perturb import _is_decorative_floor as _is_floor
    if _is_floor(f):
        return footprint_rect, (cx_px, cy_px), (cx_px, cy_px), (cx_px, cy_px)

    fx, fy = f.front_vector()

    # Arrow geometry: tail and tip near piece edges (5 cm inside each
    # edge). Length ≈ depth − 0.10 m, mostly within the footprint with
    # minor visible body coverage on average pieces. Separation
    # between adjacent pieces' arrows is handled by lateral offset
    # below (perpendicular shift), NOT by shortening.
    back_offset = max(0.10, f.local_depth / 2.0 - 0.05)
    front_offset = max(0.10, f.local_depth / 2.0 - 0.05)
    # Lateral offset (perpendicular to front_vec) computed in a
    # PRE-PASS at the caller level — only pieces whose arrows would
    # actually conflict with a parallel-and-close neighbour get a
    # shift. Most pieces stay on the centre axis (override=0 or None).
    lateral_world = lateral_override if lateral_override is not None else 0.0
    line_w, head_w = 4, 12

    # Perpendicular (left-of-front) vector for the lateral offset.
    perp_x, perp_y = -fy, fx
    tip_x_w = cx_w + fx * front_offset + perp_x * lateral_world
    tip_y_w = cy_w + fy * front_offset + perp_y * lateral_world
    tip_x_px, tip_y_px = project(tip_x_w, tip_y_w)
    back_x_w = cx_w - fx * back_offset + perp_x * lateral_world
    back_y_w = cy_w - fy * back_offset + perp_y * lateral_world
    back_x_px, back_y_px = project(back_x_w, back_y_w)

    halo = (255, 255, 255)
    _draw_arrow(draw, back_x_px, back_y_px, tip_x_px, tip_y_px,
                color=halo, width=line_w + 4, head=head_w + 6)
    arrow_color = _RED if (highlight and is_rotation_highlight) else _BLACK
    _draw_arrow(draw, back_x_px, back_y_px, tip_x_px, tip_y_px,
                color=arrow_color, width=line_w, head=head_w)

    return footprint_rect, (tip_x_px, tip_y_px), (back_x_px, back_y_px), (cx_px, cy_px)


def annotate_topdown(
    png_path: str | Path,
    *,
    scene: Scene,
    camera_meta: dict,
    highlight_id: str | None = None,
    highlight_ids: list[str] | None = None,
    rotation_highlight_ids: list[str] | None = None,
    out_path: str | Path | None = None,
) -> Path:
    png_path = Path(png_path)
    out_path = Path(out_path or png_path)

    img = Image.open(png_path).convert("RGB")
    if camera_meta.get("mode") == "ortho_top" or "view_box_world" in camera_meta:
        project = _Projector(img.size, ortho_view_box=tuple(camera_meta["view_box_world"]))
    else:
        project = _Projector(img.size, persp=camera_meta)
    draw = ImageDraw.Draw(img)
    font_grid = _try_font(max(11, img.size[0] // 90))
    font_label = _try_font(max(13, img.size[0] // 70))
    font_obj = _try_font(max(13, img.size[0] // 75))
    font_front = _try_font(max(15, img.size[0] // 50))

    # Each rect we add to `blockers` becomes a no-go zone for later labels,
    # so the order matters: walls / openings / grid go first (anchored
    # positions), furniture footprints next, and finally the dynamically-laid
    # object labels.
    blockers: list[tuple[int, int, int, int]] = []

    # Pixel-per-metre helpers for offsetting in world units.
    sx_px = project.world_dx_to_px_at_floor(1.0)
    sy_px = sx_px  # square FOV

    # Pass 0 — small red coordinate dots. Their text labels are added LAST.
    grid_coords = _draw_grid_dots(draw, project, scene.room)

    # Pass 1 — DOORS / WINDOWS first (highest priority). They communicate
    # accessibility constraints and absolutely must remain visible.
    _draw_openings(draw, font_label, project, scene.room, blockers, sx_px, sy_px)

    hi_set = set(highlight_ids or [])
    if highlight_id:
        hi_set.add(highlight_id)
    rot_set = set(rotation_highlight_ids or []) & hi_set
    others = [f for f in scene.furniture if f.object_id not in hi_set]
    highlighted = [f for f in scene.furniture if f.object_id in hi_set]

    # Pre-pass: compute lateral shifts to keep parallel-and-close arrows
    # from visually merging. Two arrows conflict when (1) their front
    # vectors are parallel or anti-parallel (|cos| > 0.9), (2) their
    # centres are within 0.5 m perpendicular to either's front, AND
    # (3) their along-axis extents overlap. On conflict, both arrows
    # are pushed perpendicular by ±min(0.30 m, 35 % of piece width),
    # with the sign decided by lexicographic order of `object_id` so
    # the two conflicting pieces always end up on opposite sides.
    from .perturb import _is_decorative_floor as _is_floor
    lateral_shifts: dict[str, float] = {}
    arrow_pieces = [f for f in scene.furniture if not _is_floor(f)]
    for f in arrow_pieces:
        lateral_shifts[f.object_id] = 0.0
    for a in arrow_pieces:
        fxA, fyA = a.front_vector()
        perp_x, perp_y = -fyA, fxA
        for b in arrow_pieces:
            if b.object_id == a.object_id:
                continue
            fxB, fyB = b.front_vector()
            dot = fxA * fxB + fyA * fyB
            if abs(dot) < 0.9:
                continue
            # Perpendicular distance from A's arrow line to B's centre
            dx, dy = b.tx - a.tx, b.ty - a.ty
            dist_perp = abs(dx * perp_x + dy * perp_y)
            if dist_perp > 0.5:
                continue
            # Along-direction projection — both arrows must overlap on
            # their shared axis to actually merge visually.
            along = abs(dx * fxA + dy * fyA)
            if along > (a.local_depth / 2.0 + b.local_depth / 2.0 + 0.1):
                continue
            # Compute the piece-frame sign so that A and B end up on
            # OPPOSITE sides of the shared world axis.
            #   - Parallel pair (dot > 0): both pieces share the same
            #     world perp direction → use DIFFERENT signs in piece
            #     frame (lex-smaller +, lex-larger −).
            #   - Anti-parallel pair (dot < 0): pieces have FLIPPED
            #     world perp directions → use the SAME sign in piece
            #     frame (so world shifts come out opposite).
            if a.object_id < b.object_id:
                sign = 1.0
            else:
                sign = -1.0 if dot > 0 else 1.0
            lateral_shifts[a.object_id] = sign * min(0.30, a.local_width * 0.35)
            break

    # Pass 2 — draw furniture polygons + arrows (no labels yet); collect meta.
    # `is_rot_hi` flag is carried along so Pass 3 (FRONT/BACK markers) only
    # fires on rotation-highlighted pieces.
    furn_meta: list[tuple[Furniture, bool, bool, tuple, tuple, tuple, tuple]] = []
    for f in others:
        meta = _draw_furniture_box_and_arrow(
            draw, project, f, highlight=False, is_rotation_highlight=False,
            lateral_override=lateral_shifts.get(f.object_id),
        )
        furn_meta.append((f, False, False, *meta))
    for f in highlighted:
        is_rot = f.object_id in rot_set
        meta = _draw_furniture_box_and_arrow(
            draw, project, f, highlight=True, is_rotation_highlight=is_rot,
            lateral_override=lateral_shifts.get(f.object_id),
        )
        furn_meta.append((f, True, is_rot, *meta))
    for f, hi, is_rot, footprint, tip, back, centre in furn_meta:
        blockers.append(footprint)
        # Also block a small zone around the arrow tip so the object_id
        # label picker (`_layout_label`) doesn't sit a label directly on
        # top of the arrow head. Tip is the most-distinctive part of the
        # arrow and getting it covered defeats the orientation cue. Use
        # a 24 px square around each tip (~ arrow_head_w x 2).
        tip_x, tip_y = tip
        blockers.append((tip_x - 24, tip_y - 24, tip_x + 24, tip_y + 24))

    # FRONT/BACK markers removed per user feedback — the boxes were
    # large enough to fully cover the arrow on wall-hugging pieces
    # like bookshelves. Rotation-highlighted pieces still get a red
    # border and a red (instead of black) arrow; that's the only
    # orientation cue.

    # Pass 4 — wall-surface labels (LOW priority; dropped if no slot).
    _draw_walls(draw, font_label, project, scene.room, blockers, sx_px, sy_px)

    # Pass 3: layout + draw object_id labels OUTSIDE each footprint, with a
    # short grey leader line back to the object centre.
    for f, hi, is_rot, footprint, tip, back, centre in furn_meta:
        text = f.object_id
        font = font_obj
        tw, th = _measure(draw, font, text)
        (lcx, lcy), rect = _layout_label(
            anchor_rect=footprint,
            label_size=(tw, th),
            image_size=img.size,
            blockers=blockers,
        )
        # Leader line: from the nearest point of the footprint AABB to the
        # closest edge midpoint of the label rect.
        cx_px, cy_px = centre
        # clamp endpoint to label rect edge
        anchor_x = min(max(cx_px, rect[0]), rect[2])
        anchor_y = min(max(cy_px, rect[1]), rect[3])
        draw.line(
            [(cx_px, cy_px), (anchor_x, anchor_y)],
            fill=_LEADER_GREY, width=1,
        )
        outline = _RED if hi else None
        _draw_label_box(draw, font, text, rect, bg=_BLUE_LABEL, outline=outline)
        blockers.append(rect)

    # Final pass — coordinate text. Skip dots that fall inside (or at the
    # edge of) a furniture footprint, and any whose text rect would collide
    # with an already-placed label.
    # element index in furn_meta tuple (f, hi, is_rot, footprint, tip, back, centre)
    furniture_rects = [m[3] for m in furn_meta]
    _draw_grid_text(draw, font_grid, project, grid_coords, blockers, furniture_rects)

    img.save(out_path)
    return out_path
