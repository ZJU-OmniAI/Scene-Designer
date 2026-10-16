"""Render a top-down view of a Scene in the scenesmith annotation style.

The rendering mirrors the conventions used by `scenesmith` Blender renders:

* Wood-tan floor inside a dark wall outline.
* A 5x5 grid of red coordinate dots labelled "(x,y)" so the judge has
  reference points to anchor positions.
* Wall surface labels ("north_wall", "south_wall", "east_wall", "west_wall")
  in blue.
* Object labels in BLUE boxes (matching scenesmith), door labels in GREEN,
  window labels in ORANGE.
* Each furniture rectangle has a black arrow showing its FRONT direction
  (yaw=0 points to +y / north).
* The "changed" object carries a thick RED border and red arrow.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow, Polygon, Rectangle

from .scene_io import Furniture, Room, Scene


# Colors picked to mirror scenesmith annotations.py
_FLOOR_COLOR = "#cfa56b"      # warm tan, like the existing per_step renders
_WALL_COLOR = "#3a3a3a"
_GRID_DOT = "#d23030"          # red
_FURN_FACE = "#bcd6e8"
_FURN_EDGE = "#1f4e79"
_HIGHLIGHT_EDGE = "#d62728"
_OBJ_LABEL_BG = "#4d99ff"      # scenesmith blue (77,153,255)
_DOOR_LABEL_BG = "#228b22"     # forest green (34,139,34)
_WINDOW_LABEL_BG = "#ffa500"   # orange (255,165,0)
_WALL_LABEL_BG = "#4d99ff"
_ARROW_COLOR = "#0a0a0a"
_HIGHLIGHT_ARROW = "#d62728"


def _wall_segment(room: Room, direction: str):
    if direction == "north":
        return ((room.x_min, room.y_max), (room.x_max, room.y_max))
    if direction == "south":
        return ((room.x_min, room.y_min), (room.x_max, room.y_min))
    if direction == "east":
        return ((room.x_max, room.y_min), (room.x_max, room.y_max))
    if direction == "west":
        return ((room.x_min, room.y_min), (room.x_min, room.y_max))
    raise ValueError(direction)


def _wall_dir_from_label(label: str) -> str:
    return {"A": "north", "B": "south", "C": "east", "D": "west"}.get(label, "north")


def _label(ax, text: str, x: float, y: float, *, bg: str, fg: str = "white",
           fontsize: int = 7, weight: str = "bold", ha: str = "center", va: str = "center",
           pad: float = 1.6, edgecolor: str | None = None) -> None:
    ax.text(
        x, y, text,
        ha=ha, va=va,
        fontsize=fontsize, color=fg, weight=weight,
        bbox=dict(facecolor=bg, edgecolor=edgecolor or bg, pad=pad, linewidth=0.6),
        zorder=5,
    )


def _format_coord(v: float) -> str:
    s = f"{v:.2f}".rstrip("0").rstrip(".")
    return s if s and s != "-0" else "0"


def _draw_floor_and_walls(ax, room: Room) -> None:
    floor = Rectangle(
        (room.x_min, room.y_min),
        room.width,
        room.length,
        facecolor=_FLOOR_COLOR,
        edgecolor=_WALL_COLOR,
        linewidth=4.0,
        zorder=0,
    )
    ax.add_patch(floor)
    inset = 0.18
    _label(ax, "north_wall", (room.x_min + room.x_max) / 2.0, room.y_max - inset, bg=_WALL_LABEL_BG)
    _label(ax, "south_wall", (room.x_min + room.x_max) / 2.0, room.y_min + inset, bg=_WALL_LABEL_BG)
    _label(ax, "east_wall", room.x_max - inset * 1.2, (room.y_min + room.y_max) / 2.0, bg=_WALL_LABEL_BG)
    _label(ax, "west_wall", room.x_min + inset * 1.2, (room.y_min + room.y_max) / 2.0, bg=_WALL_LABEL_BG)


def _draw_grid_marks(ax, room: Room, *, n_x: int = 5, n_y: int = 5) -> None:
    xs = [room.x_min + i * room.width / (n_x - 1) for i in range(n_x)]
    ys = [room.y_min + j * room.length / (n_y - 1) for j in range(n_y)]
    for x in xs:
        for y in ys:
            ax.plot([x], [y], marker="o", markersize=4.0, color=_GRID_DOT, zorder=4)
            ax.text(
                x, y - 0.18,
                f"({_format_coord(x)},{_format_coord(y)})",
                fontsize=5.5, color=_GRID_DOT,
                ha="center", va="top", zorder=4,
            )


def _draw_door(ax, room: Room, door: dict) -> None:
    label = door.get("id", "door")
    direction = _wall_dir_from_label(door.get("boundary_label", "A"))
    pos = float(door.get("position_exact", room.width / 2.0))
    width = float(door.get("width", 0.9))
    (x0, y0), (x1, y1) = _wall_segment(room, direction)
    if direction in ("north", "south"):
        cx = x0 + pos
        cy = y0
        ax.plot(
            [cx - width / 2.0, cx + width / 2.0],
            [cy, cy],
            color="#f5e1c0", linewidth=8.0, solid_capstyle="butt", zorder=2,
        )
        _label(ax, label, cx, cy, bg=_DOOR_LABEL_BG, fontsize=7)
    else:
        cy = y0 + pos
        cx = x0
        ax.plot(
            [cx, cx],
            [cy - width / 2.0, cy + width / 2.0],
            color="#f5e1c0", linewidth=8.0, solid_capstyle="butt", zorder=2,
        )
        _label(ax, label, cx, cy, bg=_DOOR_LABEL_BG, fontsize=7)


def _draw_window(ax, room: Room, win: dict) -> None:
    label = win.get("id", "window")
    direction = win.get("wall_direction") or _wall_dir_from_label(win.get("boundary_label", "A"))
    pos = float(win.get("position_along_wall", room.width / 2.0))
    width = float(win.get("width", 1.2))
    (x0, y0), (x1, y1) = _wall_segment(room, direction)
    if direction in ("north", "south"):
        cx = x0 + pos
        cy = y0
        ax.plot(
            [cx - width / 2.0, cx + width / 2.0],
            [cy, cy],
            color="#9ad8ff", linewidth=8.0, solid_capstyle="butt", zorder=2,
        )
        _label(ax, label, cx, cy, bg=_WINDOW_LABEL_BG, fontsize=7)
    else:
        cy = y0 + pos
        cx = x0
        ax.plot(
            [cx, cx],
            [cy - width / 2.0, cy + width / 2.0],
            color="#9ad8ff", linewidth=8.0, solid_capstyle="butt", zorder=2,
        )
        _label(ax, label, cx, cy, bg=_WINDOW_LABEL_BG, fontsize=7)


def _draw_furniture(ax, f: Furniture, *, highlight: bool) -> None:
    corners = f.world_corners()
    edge = _HIGHLIGHT_EDGE if highlight else _FURN_EDGE
    width = 2.8 if highlight else 1.2
    poly = Polygon(
        corners, closed=True,
        facecolor=_FURN_FACE,
        edgecolor=edge,
        linewidth=width,
        alpha=0.85,
        zorder=3,
    )
    ax.add_patch(poly)

    # Arrow extends from the centre OUTWARD past the front face of the bbox so
    # the head is always visible even when the object_id label covers the centre.
    fx, fy = f.front_vector()
    half_depth = f.local_depth / 2.0
    arrow_len = half_depth + 0.32
    arrow_color = _HIGHLIGHT_ARROW if highlight else _ARROW_COLOR
    # White halo so the arrow is readable on the tan floor.
    halo = FancyArrow(
        f.tx, f.ty,
        fx * arrow_len, fy * arrow_len,
        width=0.10, head_width=0.34, head_length=0.30,
        length_includes_head=True,
        color="white", zorder=4.5, alpha=0.9,
    )
    ax.add_patch(halo)
    ax.add_patch(FancyArrow(
        f.tx, f.ty,
        fx * arrow_len, fy * arrow_len,
        width=0.06, head_width=0.26, head_length=0.24,
        length_includes_head=True,
        color=arrow_color, zorder=5,
    ))

    # Place the label slightly off-centre, opposite the front direction, so the
    # arrow head is clear of the label.
    label_offset = max(half_depth * 0.4, 0.12)
    label_x = f.tx - fx * label_offset
    label_y = f.ty - fy * label_offset
    _label(ax, f.object_id, label_x, label_y, bg=_OBJ_LABEL_BG, fontsize=7)


def render_scene(
    scene: Scene,
    out_path: str | Path,
    *,
    title: str = "",
    highlight_id: str | None = None,
    show_legend: bool = True,
    n_grid: tuple[int, int] = (5, 5),
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    room = scene.room
    pad = 0.6
    fig, ax = plt.subplots(figsize=(7.5, 7.0), dpi=150)
    ax.set_aspect("equal")
    ax.set_xlim(room.x_min - pad, room.x_max + pad)
    ax.set_ylim(room.y_min - pad, room.y_max + pad)
    ax.set_facecolor("#f7f4ee")

    _draw_floor_and_walls(ax, room)
    _draw_grid_marks(ax, room, n_x=n_grid[0], n_y=n_grid[1])

    for d in room.doors:
        _draw_door(ax, room, d)
    for w in room.windows:
        _draw_window(ax, room, w)

    others = [f for f in scene.furniture if f.object_id != highlight_id]
    highlighted = [f for f in scene.furniture if f.object_id == highlight_id]
    for f in others:
        _draw_furniture(ax, f, highlight=False)
    for f in highlighted:
        _draw_furniture(ax, f, highlight=True)

    if title:
        ax.set_title(title, fontsize=11)

    if show_legend:
        legend_handles = [
            mpatches.Patch(facecolor=_FURN_FACE, edgecolor=_FURN_EDGE, label="Furniture footprint"),
            mpatches.Patch(facecolor="white", edgecolor=_HIGHLIGHT_EDGE, label="Changed object (red border)"),
            mpatches.Patch(facecolor=_OBJ_LABEL_BG, edgecolor=_OBJ_LABEL_BG, label="Furniture / wall label (blue)"),
            mpatches.Patch(facecolor=_DOOR_LABEL_BG, edgecolor=_DOOR_LABEL_BG, label="Door label (green)"),
            mpatches.Patch(facecolor=_WINDOW_LABEL_BG, edgecolor=_WINDOW_LABEL_BG, label="Window label (orange)"),
            mpatches.Patch(facecolor=_GRID_DOT, edgecolor=_GRID_DOT, label="Coordinate dot (red, x,y in meters)"),
            mpatches.FancyArrow(0, 0, 0.001, 0.001, color="black", label="Arrow = front of object"),
        ]
        ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=7, framealpha=0.95)

    ax.set_xlabel("x (m, +east →)")
    ax.set_ylabel("y (m, +north ↑)")
    ax.grid(False)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path
