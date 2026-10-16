"""Ask gemini-2.5-flash (via an OpenAI-compatible endpoint) to judge two scenes."""

from __future__ import annotations

import base64
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI


DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_API_KEY = None
# Judging (multimodal severity-ranking) goes to GPT-5.2 — its spatial
# reasoning over the top-down photos has been more reliable on this
# task than gemini-2.5-flash. Polish (text-only paraphrase that must
# strictly preserve the conclusion) keeps gemini-2.5-flash since it has
# proven obedient there.
DEFAULT_JUDGE_MODEL = os.environ.get("CRITIC_JUDGE_MODEL", "gpt-5.2")
DEFAULT_POLISH_MODEL = os.environ.get("CRITIC_POLISH_MODEL", "gpt-5.2")
# Backwards-compat alias kept so any older import sites don't break.
DEFAULT_MODEL = DEFAULT_JUDGE_MODEL


SYSTEM_PROMPT = """You are an expert interior-design critic.
Two annotated top-down floor plans of the same room are shown. Exactly
ONE furniture object differs (in position, orientation, or both) between
them — sometimes the difference is a hard error (collision, out-of-room,
blocking a door), sometimes it is a softer layout choice (storage moved
off its wall, sofa pulled away from the coffee table, awkward yaw).
Your job is to decide which scene is the better layout and explain why."""


def _user_prompt(
    changed_object_id: str,
    changed_object_name: str,
    *,
    secondary_object_id: str | None = None,
    is_removal: bool = False,
) -> str:
    swap_clause = ""
    if is_removal:
        swap_clause = (
            f"\nNote: the change is a REMOVAL — `{changed_object_id}` is PRESENT "
            "in one scene and ENTIRELY ABSENT (removed) from the other. The"
            " remaining furniture, walls, doors and windows are identical."
        )
    elif secondary_object_id:
        swap_clause = (
            f"\nNote: the change is a SWAP — both `{changed_object_id}` and "
            f"`{secondary_object_id}` exchanged positions and orientations between "
            "the two scenes."
        )
    return f"""You will see two top-down renderings of a single room: SCENE A and SCENE B.

Annotation legend (identical in both images, mirroring the scenesmith convention):
- Tan rectangle = room floor, dark outline = wall.
- Wall-surface labels in BLUE boxes inside each wall: `north_wall`, `south_wall`,
  `east_wall`, `west_wall`. +y is north (up), +x is east (right); distances in metres.
- Red dots arranged on a 5x5 floor grid, each annotated with its world-frame
  `(x, y)` coordinate so you can read absolute positions.
- Door openings = light-tan strip across a wall with a GREEN `door_*` label.
- Window openings = light-blue strip across a wall with an ORANGE `window_*` label.
- Furniture = light-blue filled rectangles outlined in dark blue. The BLUE label
  inside each rectangle is the object_id.
- An ARROW is drawn from each furniture's geometric centre toward its FRONT.
  The arrow tail is the object centre; the arrow HEAD shows where the FRONT
  points. For the changed object the arrow is red and there are explicit
  `FRONT` (red box) and `BACK` (grey box) text markers at the two ends.

  CONCEPT — what FRONT and BACK mean:
    * FRONT is the side of the furniture that the user / room is meant to
      face — i.e. the side where the functional / interactive surface is:
      open shelves, cabinet doors, drawers, seating cushions, keyboard
      surface, screen, foot of a bed, etc. It is the side that should face
      INTO the room (away from any wall the piece sits against).
    * BACK is the opposite side — the part that is typically hidden,
      turned away, or pushed against a wall.
  Use this rule even for furniture types not listed below:
    - a sofa / armchair / chair: FRONT = the cushion / leg side (where the
      sitter's knees go); BACK = the backrest;
    - a bed: FRONT = the foot; BACK = the headboard;
    - a media console / bookshelf / dresser / wardrobe: FRONT = the side
      with doors / drawers / open shelves; BACK = the flat back panel.
  Implications for layout judgement:
    * If a piece is against a wall, its BACK should touch the wall and the
      arrow (FRONT) must point INTO the room.
    * Two pieces forming a focal group (sofa ↔ TV, armchair ↔ coffee table)
      should have arrows pointing TOWARD each other (or toward the same
      target).
    * An arrow pointing at a wall, or at the back of another piece, is a
      strong sign of a bad orientation.
- A THICK RED border (and red arrow) marks the differing object.

The differing object is `{changed_object_id}` (a "{changed_object_name}"). All
other furniture, the room geometry, and the openings are identical.{swap_clause}

Decide which scene is the better layout and explain your reasoning.

Evaluate using these criteria, in order of importance:
1. **Physical validity** — does the object stay inside the room and not overlap
   any other furniture? Any out-of-room or colliding placement is strictly worse.
2. **Door / circulation clearance** — is the door opening clear, and are the
   primary walking paths unblocked?
3. **Functional orientation** — does the object's front arrow point somewhere
   that makes the object usable? Sofas/armchairs face the coffee table or TV;
   storage (bookshelf, media console, dresser, vanity) sits with its back to a
   wall; beds usually have the head against a wall; etc.
4. **Layout coherence** — does the placement keep functional groups together
   (sofa near coffee table, dining chair at the table, nightstand by the bed,
   storage along a wall)? Is the yaw aligned with the room's main axes?

Even when both scenes are physically valid, pick the one with a more coherent /
useful arrangement.

VERY IMPORTANT — anchor your reasoning to the LABELS in the image, not to
what you would *expect* the silhouette to be. A photoreal mesh that looks
like a bookshelf might still be labelled `coffee_table_0` if the placement
swapped — trust the BLUE label on top of each rectangle, not the wood
grain. Before judging, internally read each labelled object's actual
location in BOTH scenes and write that down first.

Respond in this EXACT format. Keep the reasoning concise (≤ 220 words):

<reasoning>
Step 1 — Inventory. Briefly state where the differing object named above
actually sits in SCENE A and in SCENE B (read the BLUE label, not the
silhouette). If a swap is involved, do the same for the secondary object.

Step 2 — Compare. Walk through the four criteria above (physical
validity, door/circulation, functional orientation, layout coherence)
referencing the inventory you just made.

Step 3 — Decide. Name the winning scene.
</reasoning>
<answer>A</answer>   or   <answer>B</answer>

The VERY LAST line of your response MUST be exactly `<answer>A</answer>`
or `<answer>B</answer>` — nothing after it."""


@dataclass
class JudgeResult:
    raw: str
    reasoning: str
    answer: str  # "A" or "B" or "UNKNOWN"


def _img_data_url(
    path: str | Path,
    *,
    max_size: int | None = None,
    jpeg_quality: int | None = None,
) -> str:
    """Encode an image as a data URL.

    Pass-through (PNG, original size) by default. When `max_size` and/or
    `jpeg_quality` are set, the image is loaded with PIL, downsized so its
    longer side is at most `max_size`, converted to RGB if necessary, and
    re-encoded as JPEG. This dramatically reduces the request payload sent
    to upstream proxies — important for 4-image ranking calls on flaky
    networks (we observed ~42% APIConnectionError rate at 4×1024² PNG vs
    <5% with 4×768² JPEG q85)."""
    if max_size is None and jpeg_quality is None:
        raw = Path(path).read_bytes()
        return "data:image/png;base64," + base64.b64encode(raw).decode("ascii")
    from io import BytesIO
    from PIL import Image
    img = Image.open(path)
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGB")
    if max_size:
        img.thumbnail((max_size, max_size), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=jpeg_quality or 85, optimize=True)
    raw = buf.getvalue()
    return "data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii")


def _parse(raw: str) -> JudgeResult:
    reasoning = ""
    answer = "UNKNOWN"
    m = re.search(r"<reasoning>(.*?)</reasoning>", raw, flags=re.DOTALL | re.IGNORECASE)
    if m:
        reasoning = m.group(1).strip()
    else:
        # No explicit reasoning tag — take everything before the <answer> tag
        # (or the whole reply if neither tag is present) as the reasoning.
        ans_pos = re.search(r"<answer>", raw, flags=re.IGNORECASE)
        head = raw[: ans_pos.start()] if ans_pos else raw
        reasoning = head.strip()
    m = re.search(r"<answer>\s*([AB])\s*</answer>", raw, flags=re.IGNORECASE)
    if m:
        answer = m.group(1).upper()
    else:
        # Fall back to common phrasings that pin down the winner.
        for pattern in (
            r"\bSCENE\s*([AB])\s+is\s+better",
            r"\bSCENE\s*([AB])\s+is\s+the\s+better",
            r"\bbetter\s+scene\s+is\s+([AB])\b",
            r"\bwinner[:\s]+SCENE\s+([AB])\b",
            r"^\s*([AB])\s*$",
        ):
            m = re.search(pattern, raw, flags=re.IGNORECASE | re.MULTILINE)
            if m:
                answer = m.group(1).upper()
                break
    return JudgeResult(raw=raw, reasoning=reasoning, answer=answer)


def make_client(
    base_url: str | None = None,
    api_key: str | None = None,
) -> OpenAI:
    return OpenAI(
        base_url=base_url or os.environ.get("CRITIC_BASE_URL") or os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL),
        api_key=api_key or os.environ.get("CRITIC_API_KEY") or os.environ.get("OPENAI_API_KEY"),
    )


# ---------------------------------------------------------------------------
# Ranking mode — judge an N-way ordering rather than a pair.
# ---------------------------------------------------------------------------

@dataclass
class RankingJudgeResult:
    raw: str
    reasoning: str
    ranking: list[str]   # e.g. ["A", "C", "B"] for 3-way; empty if unparseable


_RANKING_SYSTEM = """You are an expert interior-design critic.
You will see N (between 2 and 4) top-down photoreal renderings of the same room,
labelled SCENE A, SCENE B, ... — they share the same room geometry, doors and
windows; they differ only in furniture placement / orientation / presence.
One of them is the unperturbed original, and the others have a small,
increasing number of perturbations applied (move / rotate / swap / remove
a furniture piece). Your job is to RANK them from BEST to WORST layout."""


_TIER_RULE_BLOCK = """\

Severity tier rule (apply STRICTLY when ranking):
- Tier 1 (worst, physical violation): `move_overlap`, `rotate_into_collision`,
  `move_into_wall`, `rotate_out_of_room`, `move_blocking_door`.
- Tier 2 (severe but recoverable): `face_wall`, `rotate_180`, `swap_two`,
  `remove_object`.
- Tier 3 (soft / awkward): `move_to_corner`, `move_off_wall`,
  `move_far_from_focal`, `rotate_45_offset`.
- A scene with one tier-1 issue is worse than one with only tier-2 issues,
  which is worse than one with only tier-3 issues.
- In CUMULATIVE chains, scene at chain step k+1 has every issue scene k has
  PLUS one more, so it is strictly worse.
- In SIBLING groups, the original (no perturbation) is best, then the
  perturbed scenes are ordered by their single perturbation's tier.
- For tier-flexible kinds (`swap_two`, `move_to_corner` may sit at tier 2
  or 3; `rotate_45_offset` may be tier 0..3 depending on visible severity):
  if two perturbed scenes carry tier-flexible perts, pick the order that
  matches the visually more-disruptive one as worse."""


def _ranking_user_prompt(n: int, per_scene_perts: list[str] | None = None,
                         include_tier_rule: bool = False,
                         neutral_diff: str | None = None) -> str:
    labels = ["A", "B", "C", "D"][:n]
    label_chain_best = " > ".join(labels)
    perts_block = ""
    if neutral_diff:
        # Neutral diff mode: tell the judge WHAT differs between the
        # scenes but DO NOT reveal which scene contains the perturbation
        # nor characterise either side as good/bad. The judge must form
        # its own verdict from the visuals.
        perts_block = (
            f"\n\nThe {n} scenes differ in exactly ONE change between them: "
            f"{neutral_diff}\n"
            "You are NOT told which scene is the unperturbed original. "
            "Judge purely from the visual layout — do NOT assume the "
            "scene mentioned in the change description is the worse one."
        )
    elif per_scene_perts:
        lines = []
        for lab, descr in zip(labels, per_scene_perts):
            d = descr.strip() if descr else "no change (original layout)"
            lines.append(f"  SCENE {lab}: {d}")
        perts_block = (
            "\n\nGround-truth changes applied per scene (use this to focus your "
            "severity comparison; describe the visible problems in your reasoning, "
            "but do NOT mention these change-set bullets verbatim):\n"
            + "\n".join(lines)
        )
    if include_tier_rule:
        perts_block += _TIER_RULE_BLOCK
    return f"""You will see {n} top-down renderings of the same room: SCENE {", ".join(labels)}.{perts_block}

Annotation legend (identical in every image):
- Tan rectangle = floor; dark outline = wall.
- 5x5 floor grid of red dots labelled `(x, y)` (metres).
- Wall-surface labels in BLUE: `north_wall`, `south_wall`, `east_wall`, `west_wall`.
- Doors in GREEN labels, windows in ORANGE labels.
- Furniture: light-blue rectangles; the BLUE `object_id` label sits next to each.
- A black ARROW from each furniture's geometric centre toward its FRONT.
- A THICK RED border outlines every furniture piece that has been perturbed
  in that scene (i.e. moved / rotated / swapped relative to the unperturbed
  original); the perturbed piece's arrow is also drawn in RED instead of black.
  The original (unperturbed) scene has no red borders. Use these markers as
  hints for WHERE the change is — but your reasoning must describe what is
  wrong using natural spatial language (against the wall, blocks the door,
  faces the back of the sofa, etc.), NOT by referring to the red border or
  red arrow themselves.

Arrow concept (this matters):
- The arrow tail is the object centre; the HEAD points to the FRONT.
- FRONT = the side of the piece that the user / room is meant to face — the
  side with the functional / interactive surface (cabinet doors, drawer fronts,
  open shelves, seating cushions, screen, foot of a bed, etc.).
- BACK = the opposite side: backrest, head of a bed, back panel — the side
  that should be turned toward a wall or hidden.
- Apply this concept generally, even for furniture types not explicitly
  named: FRONT is whichever side a user interacts with.

Implications when judging:
- A piece against a wall should have its BACK against the wall (arrow points
  into the room).
- Two pieces in a focal group (sofa ↔ TV / coffee_table) should have arrows
  pointing TOWARD each other.
- An arrow pointing at a wall, or at the back of another piece, is a sign of
  bad orientation.

Rank the {n} scenes from BEST (most physically valid + functional layout) to
WORST. Use these criteria, in order:
1. Physical validity (no wall / object intersections; nothing out of room).
2. Door / circulation clearance.
3. Functional orientation (apply the FRONT / BACK concept above).
4. Layout coherence (focal groups together, walkable space).
5. Completeness (a missing piece is worse than a present one, all else equal).

Respond in this EXACT format. Keep the reasoning concise (≤ 240 words):

<reasoning>
Step 1 — Inventory. Briefly note what is different between the scenes.
Step 2 — Compare. Walk through the criteria above to compare each scene.
Step 3 — Decide. Name the ranking from best to worst.
</reasoning>
<answer>{label_chain_best}</answer>     # for example, replace with your order

The very last line MUST be exactly `<answer>X > Y > Z [> W]</answer>` using
the labels of all {n} scenes, separated by ` > ` from best to worst."""


def _parse_ranking(raw: str, n: int) -> RankingJudgeResult:
    reasoning = ""
    ranking: list[str] = []
    m = re.search(r"<reasoning>(.*?)</reasoning>", raw, flags=re.DOTALL | re.IGNORECASE)
    if m:
        reasoning = m.group(1).strip()
    else:
        ans_pos = re.search(r"<answer>", raw, flags=re.IGNORECASE)
        head = raw[: ans_pos.start()] if ans_pos else raw
        reasoning = head.strip()
    m = re.search(
        r"<answer>\s*([A-D](?:\s*[>≻>=<,]\s*[A-D]){1,3})\s*</answer>",
        raw, flags=re.IGNORECASE,
    )
    if m:
        ranking = [x.strip().upper() for x in re.split(r"[>≻>=<,]", m.group(1)) if x.strip()]
    else:
        # Loose fallback: any "A > B > C" sequence in the text.
        m2 = re.search(
            r"\b([A-D])\s*[>≻]\s*([A-D])\s*[>≻]\s*([A-D])(?:\s*[>≻]\s*([A-D]))?",
            raw,
        )
        if m2:
            ranking = [g.upper() for g in m2.groups() if g]
    expected = set(["A", "B", "C", "D"][:n])
    if ranking and (set(ranking) != expected or len(ranking) != n):
        ranking = []
    return RankingJudgeResult(raw=raw, reasoning=reasoning, ranking=ranking)


def judge_ranking(
    images: list[str | Path],
    *,
    client: OpenAI | None = None,
    model: str = DEFAULT_JUDGE_MODEL,
    temperature: float = 0.2,
    # Generous max_tokens budget — both gpt-5.2 and gemini-2.5-flash spend
    # several hundred tokens on hidden reasoning before emitting visible
    # text, and 4-way comparisons need a long visible response too.
    max_tokens: int = 5500,
    image_max_size: int = 768,
    image_jpeg_quality: int = 85,
    per_scene_perts: list[str] | None = None,
    include_tier_rule: bool = False,
    neutral_diff: str | None = None,
) -> RankingJudgeResult:
    """Ask the judge model (default gpt-5.2) to rank N (2-4) top-down
    scenes from best to worst. Caller is responsible for keeping image
    order = label order (image_list[0] is SCENE A, image_list[1] is
    SCENE B, etc.).

    Images are downsized + JPEG-encoded before transmission (default
    768² q85). At full 1024² PNG the 4-image payload pushes ~4 MB and
    triggered ~42% APIConnectionError on the previous gateway; with the
    default downsize each scene's slot is < 200 KB and connection
    failures drop sharply."""
    n = len(images)
    if not 2 <= n <= 4:
        raise ValueError(f"judge_ranking: n must be 2..4, got {n}")
    client = client or make_client()
    user_text = _ranking_user_prompt(
        n, per_scene_perts=per_scene_perts,
        include_tier_rule=include_tier_rule,
        neutral_diff=neutral_diff,
    )
    labels = ["A", "B", "C", "D"][:n]
    content = []
    for label, img in zip(labels, images):
        content.append({
            "type": "text",
            "text": f"Below is the image labelled SCENE {label}. Every reference to "
                    f"\"SCENE {label}\" in your reasoning must describe THIS image."
        })
        content.append({
            "type": "image_url",
            "image_url": {"url": _img_data_url(
                img, max_size=image_max_size, jpeg_quality=image_jpeg_quality
            )},
        })
    content.append({"type": "text", "text": user_text})
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _RANKING_SYSTEM},
            {"role": "user", "content": content},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return _parse_ranking(resp.choices[0].message.content or "", n)


def polish_ranking_reasoning(
    raw: str,
    *,
    ranking: list[str],
    perturbation_summary: str,
    client: OpenAI | None = None,
    model: str = DEFAULT_POLISH_MODEL,
    temperature: float = 0.4,
    max_tokens: int = 2400,
    record_id: str | None = None,
    max_attempts: int = 5,
) -> str:
    """Polish chain-of-thought reasoning. Retries up to `max_attempts` on
    transient errors; on final failure returns the input `raw` unchanged
    so the SFT record can still be written (just with un-polished text).
    Each attempt that fails logs a `[polish_error]` line to stderr; a
    fully exhausted attempt budget logs `[polish_giveup]`."""
    if not raw or not raw.strip():
        return raw
    client = client or make_client()
    order = " > ".join(ranking)
    sys = _POLISH_SYSTEM
    usr = (
        "Rewrite the ranking analysis below into ONE concise paragraph "
        "(80-160 words) that a senior interior designer would write. Keep "
        f"the conclusion `{order}` (best → worst). Refer to scenes simply "
        "as Scene A, Scene B, etc.\n\n"
        "STYLE — STATIC, NOT DYNAMIC:\n"
        "Describe what each scene LOOKS LIKE and why one ranks above the "
        "next. Do NOT use verbs that imply a transformation between "
        "scenes — never say 'moved', 'shifted', 'relocated', 'rotated', "
        "'swapped', 'removed', 'pushed', 'placed', 'has been turned', "
        "'is missing because it was taken out', etc. Instead state the "
        "STATIC layout features: 'in Scene A the island sits in front of "
        "the south door, blocking entry', 'Scene B has no nightstand "
        "beside the bed', 'in Scene C the sofa faces the back wall'. The "
        "SFT trainer never sees the original layout — it only sees the N "
        "scenes side by side, so phrasing them as static observations is "
        "the only honest framing.\n\n"
        "Equally important: do NOT use cross-scene COMPARATIVE phrases — "
        "never write 'compared to Scene X', 'relative to Scene Y', "
        "'unlike Scene Z', 'differs from Scene W', 'in contrast to', "
        "or any wording that frames one scene as a delta from another. "
        "Each scene's quality is a property OF that scene; the reader "
        "compares them by reading the description of each. Also avoid "
        "'by + Verbing' constructions ('by placing X', 'by rotating Y'), "
        "and avoid mentioning numeric degrees / yaw values.\n\n"
        "Do NOT mention numeric coordinates, yaws in degrees, FRONT/BACK "
        "boxes, red borders, blue labels, coordinate dots, or any other "
        "rendering overlays.\n\n"
        f"The change set producing these scenes (FOR YOUR REFERENCE ONLY — "
        "do not name the kinds nor describe the change as an action):\n"
        f"{perturbation_summary}\n\n"
        "Original analysis:\n\"\"\"\n"
        + raw
        + "\n\"\"\"\n\nOutput the rewritten paragraph and nothing else."
    )
    import sys as _sys
    import time as _time
    # Same exponential backoff as judge — proxy outages tend to come in
    # bursts; tight 2-second retries land in the same broken window.
    backoff = [2, 4, 8, 16, 32]
    tag = f"[polish_error] {record_id or '?'}"
    for attempt in range(1, max_attempts + 1):
        try:
            resp = client.chat.completions.create(
                model=model, temperature=temperature, max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": sys},
                    {"role": "user", "content": usr},
                ],
            )
            polished = (resp.choices[0].message.content or "").strip()
            polished = re.sub(r"</?(reasoning|answer)>", "", polished, flags=re.IGNORECASE).strip()
            polished = polished.strip("`\"'\n ")
            if polished:
                return polished
            print(f"[polish_empty] {record_id or '?'} attempt {attempt}: "
                  f"got empty rewrite", file=_sys.stderr)
        except Exception as e:
            print(f"{tag} attempt {attempt}: {e!r}", file=_sys.stderr)
        if attempt < max_attempts:
            _time.sleep(backoff[min(attempt - 1, len(backoff) - 1)])
    print(f"[polish_giveup] {record_id or '?'} after {max_attempts} attempts; "
          "falling back to un-polished text", file=_sys.stderr)
    return raw


_POLISH_SYSTEM = """You polish chain-of-thought reasoning written by another
model. Your job is to rewrite the analysis as a concise, natural-sounding
paragraph that a senior interior designer would write — never change the
final verdict. Output ONLY the polished paragraph, no headings, no Step
labels, no <reasoning> or <answer> tags."""


def _polish_user_prompt(
    raw: str,
    *,
    winner: str,
    changed_object_id: str,
    changed_object_name: str,
    secondary_object_id: str | None = None,
) -> str:
    swap_note = (
        f"; the change is a swap with `{secondary_object_id}`"
        if secondary_object_id else ""
    )
    return f"""Rewrite the analysis below into a single concise paragraph
(70-130 words, prose only). Keep the conclusion (SCENE {winner} is the
better layout). Use natural spatial language a designer would use ("against
the south wall", "in front of the sofa", "facing the TV", "blocks the door
swing", etc.).

Strictly do NOT mention any of: numeric coordinates, yaw / rotation in
degrees, "red arrow", "red border", "FRONT label", "BACK label", "blue
label", "coordinate dot", "grid marker", or any other rendering overlay
that exists only in the analysis image. The reader of the rewrite will see
ONLY a clean photo-real top-down with no annotations.

Reference the changed object as `{changed_object_id}` (a {changed_object_name}){swap_note}.

Original analysis:
\"\"\"
{raw}
\"\"\"

Output the rewritten paragraph and nothing else."""


def polish_reasoning(
    raw: str,
    *,
    winner: str,
    changed_object_id: str,
    changed_object_name: str,
    secondary_object_id: str | None = None,
    client: OpenAI | None = None,
    model: str = DEFAULT_POLISH_MODEL,
    temperature: float = 0.4,
    # gemini-2.5-flash budgets ~400-500 tokens for internal reasoning that
    # also count against this cap, so we set it well above the visible
    # paragraph length we want to elicit.
    max_tokens: int = 2400,
) -> str:
    """Ask Gemini to rewrite a stiff COT into a concise natural paragraph.

    Returns the polished text, or the original `raw` text if the API call
    fails. The verdict (winner) is fixed by the caller — Gemini cannot
    change it because we splice the `<answer>` tag separately.
    """
    if not raw or not raw.strip():
        return raw
    client = client or make_client()
    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": _POLISH_SYSTEM},
                {"role": "user", "content": _polish_user_prompt(
                    raw,
                    winner=winner,
                    changed_object_id=changed_object_id,
                    changed_object_name=changed_object_name,
                    secondary_object_id=secondary_object_id,
                )},
            ],
        )
        polished = (resp.choices[0].message.content or "").strip()
        # Strip stray quotes / answer tags if Gemini added them.
        polished = re.sub(r"</?(reasoning|answer)>", "", polished, flags=re.IGNORECASE).strip()
        polished = polished.strip("`\"'\n ")
        return polished or raw
    except Exception:
        return raw


def judge_pair(
    image_a: str | Path,
    image_b: str | Path,
    *,
    changed_object_id: str,
    changed_object_name: str,
    secondary_object_id: str | None = None,
    is_removal: bool = False,
    client: OpenAI | None = None,
    model: str = DEFAULT_JUDGE_MODEL,
    temperature: float = 0.2,
) -> JudgeResult:
    client = client or make_client()
    user_text = _user_prompt(
        changed_object_id, changed_object_name,
        secondary_object_id=secondary_object_id,
        is_removal=is_removal,
    )
    # Order is critical: the FIRST image is always SCENE A, the SECOND is
    # always SCENE B. We prefix each image with explicit text so the model
    # cannot confuse them, and we repeat the assertion in the user prompt.
    msg = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text",
                 "text": "Below is the FIRST image. This image IS SCENE A — every reference to "
                         "\"SCENE A\" in your reasoning must describe THIS image, not the next one."},
                {"type": "image_url", "image_url": {"url": _img_data_url(image_a)}},
                {"type": "text",
                 "text": "Below is the SECOND image. This image IS SCENE B — every reference to "
                         "\"SCENE B\" in your reasoning must describe THIS image, not the previous one."},
                {"type": "image_url", "image_url": {"url": _img_data_url(image_b)}},
                {"type": "text", "text": user_text},
            ],
        },
    ]
    resp = client.chat.completions.create(
        model=model,
        messages=msg,
        temperature=temperature,
        # gemini-2.5-flash burns 500-1500 invisible reasoning tokens that
        # also count against this cap. Give the visible Step1/2/3 COT plenty
        # of headroom on top so the `<answer>` tag is always emitted.
        max_tokens=5000,
    )
    raw = resp.choices[0].message.content or ""
    return _parse(raw)
