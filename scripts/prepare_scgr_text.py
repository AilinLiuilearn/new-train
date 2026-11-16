"""Reuse the strictly loaded BioMedCLIP tower for the one SCGR task prompt."""
import argparse
from pathlib import Path
import prepare_gt_vcl_text as encoder

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--paths',default=str(encoder.REPO/'paths.local.json'))
    p.add_argument('--output',default=str(encoder.REPO/'resources/scgr_text.json'))
    encoder.PROMPTS={'fusion':'Pulmonary tumor lesion.'}
    encoder.main(p.parse_args())
