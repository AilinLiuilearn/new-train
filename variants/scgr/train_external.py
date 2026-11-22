"""Train complete MLRF + TCSR on an external patient-disjoint PNG cohort."""
import json
import math
from pathlib import Path
import traceback

from runtime import REPO, parser, sha
from variants.tsbci import train as engine
from .model import Model
from .train import diagnostics


def main():
    cli = parser('scgr_T')
    cli.set_defaults(config=str(REPO / 'configs' / 'scgr_T.json'))
    cli.add_argument('--split-directory', type=Path, required=True)
    cli.add_argument('--text-cache', type=Path, required=True)
    args = cli.parse_args()
    config = json.loads(Path(args.config).read_text())
    if (config.get('recipe') != 'T' or not config.get('text_enabled') or config.get('score_enabled')
            or config.get('tcsr_ablation') or config.get('mlrf_ablation') or args.arm != 'full'):
        raise ValueError('External evaluation uses complete MLRF + TCSR with BCE + Dice')
    text = json.loads(args.text_cache.read_text())
    prompt = text['prompts']['fusion']
    vector, = text['embeddings']
    if not text['frozen'] or set(text['prompts']) != {'fusion'} or not prompt.strip():
        raise ValueError('Expected one fixed fusion prompt')
    if len(vector) != 512 or not math.isclose(math.hypot(*vector), 1, rel_tol=1e-5):
        raise ValueError('Expected a normalized 512-dimensional text embedding')

    def build(cfg, **kwargs):
        return Model(cfg, text_anchor=vector, **kwargs)

    def report(rows, complete=False):
        best = max(rows, key=lambda row: row['validation']['full']['dice'])
        engine.dump('RESULTS.json', {
            'complete': complete, 'best_epoch': best['epoch'], 'weights': best['weights'],
            'validation_metrics': best['validation']['full'],
            'selection': 'Patient-disjoint internal validation Dice',
        })

    try:
        engine.main(args, variant='scgr_external_T', model_factory=build, results_fn=report,
                    diagnostics_fn=diagnostics, protocol_extra={
                        'fixed_text': prompt,
                        'fixed_text_cache_sha256': sha(args.text_cache),
                        'scope': 'Internal training-cohort validation; held-out test never used for selection',
                    })
    except Exception as error:
        traceback.print_exc()
        if engine.OUT is not None:
            engine.status('failed', error=repr(error))
        raise SystemExit(1)


if __name__ == '__main__':
    main()
