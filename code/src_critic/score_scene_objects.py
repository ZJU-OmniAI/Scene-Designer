"""Score every furniture piece in a scene on two attributes via GPT-5.2:

  * importance ∈ [0, 1]: how essential the piece is to the room's primary
    function (sofa in a living room → ~1.0; decorative side table → ~0.2).
  * directionality ∈ [0, 1]: how much the piece's yaw / orientation
    visually matters (TV / sofa / bed / dining_chair → ~1.0; round table,
    floor lamp, ottoman → ~0.0).

Single per-scene LLM call (image + the full furniture list). Result is
cached to `<scene_dir>/_obj_properties.json` so all downstream data
generators read it locally.

Usage:
    python -m src_critic.score_scene_objects                   # all scenes
    python -m src_critic.score_scene_objects --force            # re-score
    python -m src_critic.score_scene_objects --scene SCENE_NAME # one scene
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .judge import _img_data_url, make_client
from .scene_io import load_scene
from .topdown import render_scene


SCORE_MODEL = "gpt-5.2"


_SYSTEM = """You are scoring furniture pieces in a single room for two attributes:

importance ∈ [0, 1] — how essential is this piece to the room's primary function?
  * 1.0 — irreplaceable anchor of the room's primary activity (sofa in a
    living room; bed in a bedroom; dining_table in a dining room; vanity
    or kitchen_island in their respective rooms).
  * 0.7 — major supporting piece (armchair, desk, wardrobe, large media
    console, second seat in a small living room).
  * 0.4 — minor supporting piece (side_table, nightstand, console, small
    bookshelf).
  * 0.2 — auxiliary / functional accent (floor_lamp, small bench).
  * 0.05 — pure decor (plant, decorative sculpture, throw pillow).

directionality ∈ [0, 1] — how much does this piece's orientation matter
visually and functionally? Specifically: if you rotated this piece 180°
or 45°, would the resulting scene look obviously wrong?
  * 1.0 — strong front-back asymmetry; rotating obviously breaks the
    scene (TV, sofa, bed, dining_chair, desk, wardrobe with doors,
    armchair with a clear back).
  * 0.6 — mild directional preference (console, media_unit, bench).
  * 0.3 — slight asymmetry, mostly visual (rectangular coffee_table,
    rectangular ottoman).
  * 0.05 — rotationally near-symmetric (round side_table, round coffee
    table, floor_lamp, plant, square ottoman).

Output a single JSON object mapping each `object_id` to an object with
keys `importance`, `directionality`, and a short `rationale` string
(≤ 12 words). No extra prose, no commentary, just the JSON."""


def _user_text(scene, room_name: str) -> str:
    parts = ["Furniture inventory (id, semantic name, footprint width × depth in m):\n"]
    for f in scene.furniture:
        parts.append(
            f"  - {f.object_id}  ({f.name!r}, "
            f"{f.local_width:.2f}×{f.local_depth:.2f} m)"
        )
    parts.append(
        f"\nThe room is a {room_name} (size "
        f"{scene.room.width:.1f}×{scene.room.length:.1f} m). "
        "Score every object above on `importance` and `directionality`, "
        "then output the JSON object as instructed."
    )
    return "\n".join(parts)


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_json(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        # strip ```json ... ``` fences
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    m = _JSON_BLOCK_RE.search(raw)
    if not m:
        raise ValueError(f"no JSON object in response: {raw[:160]!r}")
    return json.loads(m.group(0))


def score_scene(scene_dir: Path, *, client=None, force: bool = False,
                model: str = SCORE_MODEL,
                max_attempts: int = 4) -> dict:
    """Return {object_id: {importance, directionality, rationale}}.
    Uses cached file unless `force=True`."""
    cache = scene_dir / "_obj_properties.json"
    if cache.exists() and not force:
        return json.loads(cache.read_text())

    scene = load_scene(scene_dir)
    if not scene.furniture:
        cache.write_text("{}")
        return {}

    # Render an annotated top-down (no highlight, no legend overlay) for
    # the LLM. Save to /tmp so the scene_dir stays clean.
    img_path = Path(f"/tmp/score_top_{scene_dir.name}.png")
    render_scene(scene, img_path,
                 title=f"{scene_dir.name} — {scene.room.room_id}",
                 highlight_id=None, show_legend=False)

    client = client or make_client()
    backoff = [3, 6, 12, 24]
    for attempt in range(1, max_attempts + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": [
                        {"type": "image_url",
                         "image_url": {"url": _img_data_url(
                             img_path, max_size=768, jpeg_quality=85)}},
                        {"type": "text",
                         "text": _user_text(scene, scene.room.room_id)},
                    ]},
                ],
                temperature=0.1,
                max_tokens=4000,
            )
            raw = resp.choices[0].message.content or ""
            parsed = _parse_json(raw)
            # Validate: every furniture must appear; values clamped to [0,1].
            out: dict = {}
            for f in scene.furniture:
                rec = parsed.get(f.object_id)
                if not isinstance(rec, dict):
                    raise ValueError(
                        f"missing {f.object_id} in LLM output for "
                        f"{scene_dir.name}")
                imp = float(rec.get("importance", 0.0))
                dirn = float(rec.get("directionality", 0.0))
                out[f.object_id] = {
                    "importance": max(0.0, min(1.0, imp)),
                    "directionality": max(0.0, min(1.0, dirn)),
                    "rationale": str(rec.get("rationale", ""))[:200],
                }
            cache.write_text(json.dumps(out, ensure_ascii=False, indent=2))
            return out
        except Exception as e:
            print(f"[score_err] {scene_dir.name} attempt {attempt}: {e!r}",
                  file=sys.stderr)
            if attempt < max_attempts:
                time.sleep(backoff[min(attempt - 1, len(backoff) - 1)])
    print(f"[score_giveup] {scene_dir.name} after {max_attempts} attempts",
          file=sys.stderr)
    return {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path,
                    default=Path("datasets/raw"))
    ap.add_argument("--scene", type=str, default=None,
                    help="Score only this single scene name; default all.")
    ap.add_argument("--force", action="store_true",
                    help="Re-score even if cache exists.")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    if args.scene:
        scene_dirs = [args.raw_dir / args.scene]
    else:
        scene_dirs = sorted(p for p in args.raw_dir.iterdir()
                            if p.is_dir() and not p.name.startswith("_"))
        # Apply blocklist
        bl = args.raw_dir / "_blocklist.txt"
        if bl.exists():
            blocked = {
                ln.strip() for ln in bl.read_text().splitlines()
                if ln.strip() and not ln.strip().startswith("#")
            }
            scene_dirs = [p for p in scene_dirs if p.name not in blocked]
    print(f"scoring {len(scene_dirs)} scene(s)")

    client = make_client()

    def worker(sd):
        try:
            res = score_scene(sd, client=client, force=args.force)
            return sd.name, len(res), None
        except Exception as e:
            traceback.print_exc()
            return sd.name, 0, repr(e)

    t0 = time.time()
    done = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(worker, sd) for sd in scene_dirs]
        for fut in as_completed(futs):
            name, n_obj, err = fut.result()
            done += 1
            if err:
                failed += 1
                print(f"[{done}/{len(scene_dirs)}] {name}: ERROR {err}")
            else:
                print(f"[{done}/{len(scene_dirs)}] {name}: {n_obj} objects "
                      f"({time.time()-t0:.0f}s)")
    print(f"\ndone: {done - failed} ok, {failed} failed, "
          f"elapsed {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
