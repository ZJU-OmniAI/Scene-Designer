"""Generate image-grounded reasoning for n4 records.

Unlike `generate_n4_reasoning.py` (which writes a rule-based draft then
asks gemini-2.5-flash to polish text without seeing any image), this
script gives gpt-5.2 the four scene renderings plus the ground-truth
ranking and asks it to write a natural justification that references
visible spatial features. The result is less stiff because the model
can describe what it actually sees rather than paraphrasing a templated
draft.

Pipeline per record:
  1. Resolve the 4 image paths in label order A/B/C/D from
     `record['images']` (after permutation the dict keys are the
     final labels).
  2. Build a perturbation-per-scene summary that lists each scene's
     applied perturbations (kind + object_id), so the model knows
     where to look.
  3. Send (4 images, perturbation summary, GT ranking) to gpt-5.2 with
     a prompt that fixes the ranking and asks for an 80-160 word
     justification. The conclusion is NOT up for debate; the model
     only writes the reasoning text.
  4. Wrap output in `<reasoning>...</reasoning>\n<answer>RANKING</answer>`
     and replace messages[1].

Run in parallel; network-bound. Idempotent on the output file's `id`s
when --resume is set.
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
from threading import Lock

from .judge import _img_data_url, make_client


_SYSTEM_PROMPT = """You are a senior interior-design critic writing a short
ranking analysis for a training dataset. You will see N (between 2 and 4)
top-down photoreal renderings of the SAME room — they share geometry,
doors and windows; they differ only in furniture placement / orientation /
presence. The ground-truth ranking from best to worst is provided to you
as a given; do NOT debate or change it. Your job is to write a concise,
natural paragraph that justifies that ranking by describing what is
visually wrong (or right) in each scene."""


_STYLE_RULES = """STYLE — STATIC, NOT DYNAMIC:
Describe what each scene LOOKS LIKE and why one ranks above the next. Do
NOT use verbs that imply a transformation between scenes — never say
'moved', 'shifted', 'relocated', 'rotated', 'swapped', 'removed',
'pushed', 'placed', 'has been turned', 'is missing because it was taken
out'. Instead state the STATIC layout features visible in the image:
'in Scene A the office chair sits in the doorway, blocking entry',
'Scene B has no desk on the back wall', 'in Scene C the bookcase floats
in the middle of the room', etc.

Do NOT use cross-scene COMPARATIVE phrases — never write 'compared to
Scene X', 'relative to Scene Y', 'unlike Scene Z', 'differs from
Scene W', 'in contrast to', or any wording that frames one scene as a
delta from another. Each scene's quality is a property OF that scene.

Do NOT mention numeric coordinates, yaw / rotation in degrees, FRONT or
BACK labels, red borders, blue labels, coordinate dots, grid markers, or
any rendering overlay. The reader sees only clean top-down photos.

Do NOT name perturbation kinds (move_to_corner, rotate_180, etc.) — use
natural spatial language ("against the south wall", "in front of the
sofa", "blocking the door", "facing the wall")."""


_USER_TEMPLATE = """You will see {n} top-down renderings of the same room, in
label order SCENE A, SCENE B{rest_labels}.

Annotation legend (identical in every image):
- Tan rectangle = floor; dark outline = wall.
- 5x5 floor grid of red dots labelled `(x, y)` (metres).
- Wall-surface labels in BLUE: `north_wall`, `south_wall`, `east_wall`, `west_wall`.
- Doors in GREEN labels, windows in ORANGE labels.
- Furniture: light-blue rectangles; the BLUE `object_id` label sits next to each.
- A black ARROW from each furniture's geometric centre toward its FRONT.
- A THICK RED border outlines every furniture piece that has been perturbed
  in that scene (i.e. moved / rotated / swapped relative to the unperturbed
  original); the perturbed piece's arrow is also drawn in RED. Use these
  markers as hints for WHERE the change is — but your reasoning must
  describe what is wrong using natural spatial language, NOT by referring
  to the red border or red arrow themselves.

Arrow concept:
- The arrow tail is the object centre; the HEAD points to the FRONT.
- FRONT = the interactive side (cushions, screen, drawer fronts, foot of a bed).
- BACK = the opposite side (backrest, headboard, back panel).
- A piece against a wall should have its BACK to the wall (arrow into room).
- A focal group (sofa ↔ TV) should have arrows pointing toward each other.
- An arrow at a wall or at the back of another piece is a sign of bad orientation.

Ground-truth changes applied per scene (use this to focus on WHERE the
problems are — describe the visible result in your reasoning, but do
NOT mention these change-set bullets verbatim and do NOT name the kinds):
{perts_block}

The ground-truth ranking from BEST to WORST is FIXED as:
    {ranking_chain}

This conclusion is GIVEN. You must not change it. Write the reasoning
that justifies WHY this ranking is correct.

{style_rules}

Output format — exactly one paragraph of 80-160 words, then the answer
tag on its own line:

<reasoning>
[your single paragraph here]
</reasoning>
<answer>{answer_str}</answer>

The very last line MUST be exactly `<answer>{answer_str}</answer>`."""


_REASONING_RE = re.compile(r"<reasoning>(.*?)</reasoning>",
                           flags=re.DOTALL | re.IGNORECASE)
_ANSWER_RE = re.compile(r"<answer>\s*([A-D](?:\s*[>≻>=<,]\s*[A-D]){1,3})\s*</answer>",
                        flags=re.IGNORECASE)


def _obj_pretty(obj_id: str) -> str:
    parts = obj_id.split("_")
    if parts and parts[-1].isdigit():
        parts = parts[:-1]
    return " ".join(parts) if parts else obj_id


def _perts_block(perts_by_label: dict[str, list[dict]], labels: list[str]) -> str:
    out = []
    for lab in labels:
        ps = perts_by_label.get(lab) or []
        if not ps:
            out.append(f"  SCENE {lab}: original (no perturbation)")
            continue
        bits = []
        for p in ps:
            kind = p.get("kind", "?")
            obj = p.get("object_id", "?")
            bits.append(f"{kind} on `{obj}` (a {_obj_pretty(obj)})")
        out.append(f"  SCENE {lab}: " + "; ".join(bits))
    return "\n".join(out)


def _resolve_image_paths(rec: dict, base_dir: Path) -> list[str] | None:
    """Return the 4 image paths in label order A/B/C/D.

    Prefer `record['messages'][0].content` (image entries in the actual
    SCENE A..D order the trainer will see). Fall back to `record['images']`
    which is a dict {label: path}.
    """
    n = int(rec.get("n", 4))
    labels = list("ABCD"[:n])
    # Path resolution helper: dataset records use repo-relative paths.
    def _abs(p: str) -> str:
        pp = Path(p)
        if pp.is_absolute():
            return str(pp)
        return str((base_dir / pp).resolve())
    msgs = rec.get("messages") or []
    if msgs:
        content = msgs[0].get("content") or []
        img_entries = [c for c in content if c.get("type") == "image"]
        if len(img_entries) == n:
            paths = []
            for c in img_entries:
                p = c.get("path") or c.get("image")
                if not p:
                    paths = []
                    break
                paths.append(_abs(p))
            if paths:
                return paths
    images = rec.get("images") or {}
    if isinstance(images, dict) and all(lab in images for lab in labels):
        return [_abs(images[lab]) for lab in labels]
    return None


def _call_with_retry(client, *, system: str, user_text: str,
                     image_urls: list[str], labels: list[str],
                     temperature: float, max_tokens: int, model: str,
                     record_id: str, max_attempts: int) -> str:
    """Send the request, retrying on transient errors."""
    backoff = [2, 4, 8, 16, 32]
    content = []
    for label, url in zip(labels, image_urls):
        content.append({
            "type": "text",
            "text": f"Below is the image labelled SCENE {label}. Every reference to "
                    f"\"SCENE {label}\" in your reasoning must describe THIS image.",
        })
        content.append({"type": "image_url", "image_url": {"url": url}})
    content.append({"type": "text", "text": user_text})
    last_err = None
    for attempt in range(1, max_attempts + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": content},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            txt = resp.choices[0].message.content or ""
            if txt.strip():
                return txt
            print(f"[empty] {record_id} attempt {attempt}", file=sys.stderr)
        except Exception as e:
            last_err = e
            print(f"[err] {record_id} attempt {attempt}: {e!r}", file=sys.stderr)
        if attempt < max_attempts:
            time.sleep(backoff[min(attempt - 1, len(backoff) - 1)])
    print(f"[giveup] {record_id} after {max_attempts} attempts (last={last_err!r})",
          file=sys.stderr)
    return ""


def _extract_reasoning(raw: str, ranking: list[str]) -> str | None:
    """Pull the reasoning paragraph out of `<reasoning>...</reasoning>`.
    If absent, fall back to text before `<answer>`."""
    if not raw:
        return None
    m = _REASONING_RE.search(raw)
    if m:
        text = m.group(1).strip()
    else:
        ans_pos = raw.lower().find("<answer>")
        text = (raw[:ans_pos] if ans_pos >= 0 else raw).strip()
    text = re.sub(r"</?(reasoning|answer)>", "", text, flags=re.IGNORECASE).strip()
    text = text.strip("`\"'\n ")
    if not text:
        return None
    return text


def _build_assistant_msg(reasoning: str, ranking: list[str]) -> str:
    answer = "".join(ranking)
    return f"<reasoning>\n{reasoning}\n</reasoning>\n<answer>{answer}</answer>"


def process_record(rec: dict, *, client, base_dir: Path,
                   image_max_size: int, image_jpeg_quality: int,
                   model: str, temperature: float, max_tokens: int,
                   max_attempts: int) -> dict | None:
    record_id = rec.get("id", "?")
    n = int(rec.get("n", 4))
    labels = list("ABCD"[:n])
    gt = rec.get("ground_truth") or {}
    ranking = gt.get("ranking_best_to_worst") or []
    perts_by_label = gt.get("perturbations_by_label") or {}
    if len(ranking) != n:
        print(f"[skip] {record_id}: bad ranking {ranking}", file=sys.stderr)
        return None
    paths = _resolve_image_paths(rec, base_dir)
    if not paths or len(paths) != n:
        print(f"[skip] {record_id}: couldn't resolve {n} image paths", file=sys.stderr)
        return None
    image_urls = []
    for p in paths:
        try:
            image_urls.append(_img_data_url(
                p, max_size=image_max_size, jpeg_quality=image_jpeg_quality,
            ))
        except Exception as e:
            print(f"[skip] {record_id}: image load fail {p}: {e!r}", file=sys.stderr)
            return None
    rest_labels = ", ".join(labels[2:])
    rest_clause = f", SCENE {rest_labels}" if rest_labels else ""
    perts_block = _perts_block(perts_by_label, labels)
    ranking_chain = " > ".join(ranking)
    answer_str = "".join(ranking)
    user_text = _USER_TEMPLATE.format(
        n=n,
        rest_labels=rest_clause,
        perts_block=perts_block,
        ranking_chain=ranking_chain,
        style_rules=_STYLE_RULES,
        answer_str=answer_str,
    )
    raw = _call_with_retry(
        client,
        system=_SYSTEM_PROMPT,
        user_text=user_text,
        image_urls=image_urls,
        labels=labels,
        temperature=temperature,
        max_tokens=max_tokens,
        model=model,
        record_id=record_id,
        max_attempts=max_attempts,
    )
    if not raw:
        return None
    reasoning = _extract_reasoning(raw, ranking)
    if not reasoning:
        print(f"[skip] {record_id}: no reasoning extractable from raw", file=sys.stderr)
        return None
    asst_msg = _build_assistant_msg(reasoning, ranking)
    out = dict(rec)
    msgs = list(rec.get("messages", []))
    if len(msgs) >= 2 and msgs[1].get("role") == "assistant":
        msgs[1] = {"role": "assistant", "content": asst_msg}
    elif len(msgs) >= 1:
        msgs.append({"role": "assistant", "content": asst_msg})
    else:
        return None
    out["messages"] = msgs
    out["_reasoning_method"] = "image_grounded_gpt52"
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-records", type=int, default=0)
    ap.add_argument("--resume", action="store_true",
                    help="If --out exists, skip records whose id already appears in it.")
    ap.add_argument("--base-dir", type=Path, default=Path.cwd(),
                    help="Resolve repo-relative image paths against this directory.")
    ap.add_argument("--image-max-size", type=int, default=768)
    ap.add_argument("--image-jpeg-quality", type=int, default=85)
    ap.add_argument("--model", default="gpt-5.2")
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--max-tokens", type=int, default=4500)
    ap.add_argument("--max-attempts", type=int, default=4)
    args = ap.parse_args()

    records = []
    with args.inp.open() as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    if args.max_records:
        records = records[: args.max_records]
    done_ids: set[str] = set()
    if args.resume and args.out.exists():
        with args.out.open() as f:
            for line in f:
                if line.strip():
                    try:
                        done_ids.add(json.loads(line)["id"])
                    except Exception:
                        pass
        records = [r for r in records if r["id"] not in done_ids]
        print(f"resume: {len(done_ids)} already done; {len(records)} remaining",
              file=sys.stderr)

    print(f"processing {len(records)} records → {args.out}", file=sys.stderr)
    client = make_client()
    write_lock = Lock()
    mode = "a" if args.resume else "w"
    fh = args.out.open(mode, encoding="utf-8")
    done = 0
    failed = 0
    t0 = time.time()

    def worker(rec):
        try:
            return process_record(
                rec, client=client, base_dir=args.base_dir,
                image_max_size=args.image_max_size,
                image_jpeg_quality=args.image_jpeg_quality,
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                max_attempts=args.max_attempts,
            )
        except Exception as e:
            print(f"[worker-err] {rec.get('id')}: {e!r}", file=sys.stderr)
            traceback.print_exc()
            return None

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(worker, r) for r in records]
        for fut in as_completed(futs):
            rec = fut.result()
            with write_lock:
                if rec is None:
                    failed += 1
                else:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fh.flush()
                done += 1
                if done % 10 == 0 or done == len(records):
                    elapsed = time.time() - t0
                    rate = done / max(elapsed, 1)
                    eta = (len(records) - done) / max(rate, 0.01)
                    print(f"[{done}/{len(records)}] failed={failed} "
                          f"elapsed={elapsed:.0f}s rate={rate:.2f}/s eta={eta:.0f}s",
                          file=sys.stderr)
    fh.close()
    print(f"wrote {args.out} ({done - failed} ok, {failed} failed)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
