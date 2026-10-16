"""Check portable release layout and selected optional runtime dependencies."""
import argparse
import importlib.util
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.runtime import ROOT, SCENESMITH_ROOT


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime',action='store_true')
    p.add_argument('--model',type=Path)
    args=p.parse_args()
    for name in ['src/train_sft.py','src/train_grpo.py','src/rollout_client.py','trajectory_compression/cli.py','src_critic/build_dataset.py']:
        assert (ROOT/name).is_file(),name
    if args.runtime:
        for name in ['torch','transformers','PIL','openai','bpy','pydrake','hydra','fastapi','scenesmith']:
            if importlib.util.find_spec(name) is None:raise SystemExit(f'Missing dependency: {name}')
        assert (SCENESMITH_ROOT/'configurations/scene_designer.yaml').is_file(), 'Run scripts/setup_scenesmith.py first'
    if args.model:
        folder=args.model
        for name in ['config.json','tokenizer_config.json','preprocessor_config.json']:
            assert (folder/name).is_file(),name
        index=folder/'model.safetensors.index.json'
        weights=set(json.loads(index.read_text())['weight_map'].values()) if index.exists() else {'model.safetensors'}
        for name in weights:assert (folder/name).is_file(),name
    print('Preflight passed.')
if __name__=='__main__':main()
