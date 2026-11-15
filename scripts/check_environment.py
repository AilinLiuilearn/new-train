"""Preflight only: never starts a training run or writes model weights."""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime import require_myenv, local_paths, verify_resources, environment

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--paths', default='paths.local.json')
    args = p.parse_args()
    require_myenv()
    report = {'environment': environment(), 'resources': verify_resources(local_paths(args.paths))}
    print(json.dumps(report, ensure_ascii=False, indent=2))
