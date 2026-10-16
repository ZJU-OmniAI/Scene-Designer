"""Convert Scene-OTC JSONL into portable per-record Critic SFT JSONs and images."""
import argparse
import json
import re
import shutil
from pathlib import Path


def convert(record, source_root, output, index):
    messages = record['messages']
    user = next(m for m in messages if m['role'] == 'user')['content']
    completion = next(m for m in messages if m['role'] == 'assistant')['content']
    reasoning = re.search(r'<reasoning>(.*?)</reasoning>', completion, re.S)
    answer = re.search(r'<answer>(.*?)</answer>', completion, re.S)
    if not reasoning or not reasoning.group(1).strip() or not answer:
        raise ValueError(f'Record {index}: missing reasoning or answer')
    images = [b.get('path', b.get('image')) for b in user if b['type'] == 'image']
    prompt = '\n'.join(b['text'] for b in user if b['type'] == 'text')
    ranking = re.sub(r'\s+', '', answer.group(1)).upper()
    labels = 'ABCD'[:len(images)]
    if len(images) not in (2, 3, 4) or not prompt:
        raise ValueError(f'Record {index}: invalid candidate count or prompt')
    # Legacy binary records supervise the winner alone; preserve that task.
    if not (len(images) == 2 and ranking in labels and len(ranking) == 1):
        if sorted(ranking) != sorted(labels):
            raise ValueError(f'Record {index}: ranking is not a permutation of {labels}')
    truth = record.get('ground_truth', {})
    expected = truth.get('ranking_best_to_worst') or truth.get('better_scene')
    if expected and ranking != ''.join(expected):
        raise ValueError(f'Record {index}: answer disagrees with ground truth')
    paths = []
    for slot, raw in enumerate(images):
        source = Path(raw)
        if not source.is_absolute(): source = source_root / source
        if not source.is_file(): raise FileNotFoundError(source)
        rel = Path('images') / f'{index:07d}_{labels[slot]}{source.suffix.lower()}'
        (output/rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, output/rel)
        paths.append(rel.as_posix())
    result = dict(stage='critic', prompt=prompt, reasoning=reasoning.group(1).strip(),
                  answer=ranking, images=paths)
    (output/'sft').mkdir(exist_ok=True)
    (output/'sft'/f'critic_{index:07d}.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', type=Path, required=True, nargs='+')
    ap.add_argument('--source-root', type=Path, default=Path('.'), help='Base of relative image paths in JSONL')
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Use an empty output directory to avoid stale samples')
    args.output.mkdir(parents=True, exist_ok=True)
    count = 0
    for path in args.input:
        for line in path.read_text().splitlines():
            if not line.strip(): continue
            convert(json.loads(line), args.source_root, args.output, count)
            count += 1
    if not count: raise ValueError('No records')
    print(f'Converted {count} records; train with --sft-root {args.output}/sft --repo-root {args.output}')

if __name__ == '__main__': main()
