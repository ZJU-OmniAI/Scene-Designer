"""Load a generated scene from the OpenSceneAgent raw dataset format."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Furniture:
    object_id: str
    name: str
    description: str
    tx: float
    ty: float
    yaw_deg: float
    bbox_min: tuple[float, float, float]
    bbox_max: tuple[float, float, float]
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def local_width(self) -> float:
        return self.bbox_max[0] - self.bbox_min[0]

    @property
    def local_depth(self) -> float:
        return self.bbox_max[1] - self.bbox_min[1]

    def world_corners(self) -> list[tuple[float, float]]:
        hw = self.local_width / 2.0
        hd = self.local_depth / 2.0
        local = [(-hw, -hd), (hw, -hd), (hw, hd), (-hw, hd)]
        c = math.cos(math.radians(self.yaw_deg))
        s = math.sin(math.radians(self.yaw_deg))
        return [(self.tx + lx * c - ly * s, self.ty + lx * s + ly * c) for lx, ly in local]

    def world_aabb(self) -> tuple[float, float, float, float]:
        xs, ys = zip(*self.world_corners())
        return min(xs), min(ys), max(xs), max(ys)

    def front_vector(self) -> tuple[float, float]:
        """+y in local frame is the 'front' (yaw=0 faces +y / north)."""
        c = math.cos(math.radians(self.yaw_deg))
        s = math.sin(math.radians(self.yaw_deg))
        return (-s, c)

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "name": self.name,
            "description": self.description,
            "translation": [self.tx, self.ty, self.extra.get("z", 0.0)],
            "yaw_deg": self.yaw_deg,
            "bbox_min": list(self.bbox_min),
            "bbox_max": list(self.bbox_max),
        }


@dataclass
class Room:
    room_id: str
    width: float   # along x
    length: float  # along y
    cx: float
    cy: float
    doors: list[dict[str, Any]] = field(default_factory=list)
    windows: list[dict[str, Any]] = field(default_factory=list)

    @property
    def x_min(self) -> float:
        return self.cx - self.width / 2.0

    @property
    def x_max(self) -> float:
        return self.cx + self.width / 2.0

    @property
    def y_min(self) -> float:
        return self.cy - self.length / 2.0

    @property
    def y_max(self) -> float:
        return self.cy + self.length / 2.0


@dataclass
class Scene:
    scene_dir: Path
    room: Room
    furniture: list[Furniture]
    house_prompt: str = ""

    def by_id(self, object_id: str) -> Furniture | None:
        return next((f for f in self.furniture if f.object_id == object_id), None)

    def clone(self) -> "Scene":
        return Scene(
            scene_dir=self.scene_dir,
            room=Room(
                room_id=self.room.room_id,
                width=self.room.width,
                length=self.room.length,
                cx=self.room.cx,
                cy=self.room.cy,
                doors=[dict(d) for d in self.room.doors],
                windows=[dict(w) for w in self.room.windows],
            ),
            furniture=[
                Furniture(
                    object_id=f.object_id,
                    name=f.name,
                    description=f.description,
                    tx=f.tx,
                    ty=f.ty,
                    yaw_deg=f.yaw_deg,
                    bbox_min=tuple(f.bbox_min),
                    bbox_max=tuple(f.bbox_max),
                    extra=dict(f.extra),
                )
                for f in self.furniture
            ],
            house_prompt=self.house_prompt,
        )


def _quat_to_yaw_deg(qwxyz: list[float]) -> float:
    w, x, y, z = qwxyz
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return math.degrees(yaw)


def load_scene(raw_scene_dir: str | Path) -> Scene:
    raw_scene_dir = Path(raw_scene_dir)
    fp_path = raw_scene_dir / "floor_plan.json"
    # GRPO branch dirs don't have floor_plan.json — fall back to house_layout.json in ancestors
    if not fp_path.exists():
        _found = False
        for _parent in raw_scene_dir.parents:
            for _name in ("floor_plan.json", "house_layout.json"):
                _candidate = _parent / _name
                if _candidate.exists():
                    fp_path = _candidate
                    _found = True
                    break
            if _found:
                break
    fr_files = [
        p for p in sorted(raw_scene_dir.glob("furniture_*.json"))
        if ".deepseek_raw" not in p.name
    ]
    if not fr_files:
        raise FileNotFoundError(f"No furniture_*.json in {raw_scene_dir}")
    fp = json.loads(fp_path.read_text())
    fr = json.loads(fr_files[0].read_text())
    if not isinstance(fr, dict):
        raise ValueError(
            f"Unexpected furniture file format in {fr_files[0]} (type={type(fr).__name__})"
        )

    room_data = fp["rooms"][0]
    # scenesmith convention: `length` is the x-axis extent (east-west),
    # `width` is the y-axis extent (north-south, a.k.a. depth).
    x_extent = float(room_data.get("length", room_data.get("width", 5.0)))
    y_extent = float(room_data.get("width", room_data.get("depth", 5.0)))
    pos = room_data.get("position", [0.0, 0.0])
    room = Room(
        room_id=room_data["id"],
        width=x_extent,
        length=y_extent,
        cx=float(pos[0]),
        cy=float(pos[1]),
        doors=fp.get("doors", []),
        windows=fp.get("windows", []),
    )

    furniture: list[Furniture] = []
    for f in fr.get("placed_furniture", []):
        tx, ty, tz = f["translation"]
        yaw = _quat_to_yaw_deg(f["rotation_wxyz"])
        furniture.append(
            Furniture(
                object_id=f["object_id"],
                name=f.get("name", f["object_id"]),
                description=f.get("description", ""),
                tx=float(tx),
                ty=float(ty),
                yaw_deg=float(yaw),
                bbox_min=tuple(f["bbox_min"]),
                bbox_max=tuple(f["bbox_max"]),
                extra={"z": float(tz)},
            )
        )

    return Scene(
        scene_dir=raw_scene_dir,
        room=room,
        furniture=furniture,
        house_prompt=fp.get("house_prompt", ""),
    )
