#!/usr/bin/env python3
"""Verify the exact Phase 3/4 release evidence, without reading protected splits."""
import hashlib, json, sys
from pathlib import Path

def main():
    root=Path(__file__).resolve().parents[1]
    obj=json.loads((root/'docs/phase3_phase4_release_manifest_20261009.json').read_text())
    mismatches=[]
    for rel,want in obj['new_files_sha256'].items():
        p=root/rel
        if not p.is_file(): mismatches.append(rel+': missing')
        elif hashlib.sha256(p.read_bytes()).hexdigest()!=want: mismatches.append(rel+': digest differs')
    forbidden=[p for p in obj['new_files_sha256'] if p.lower().endswith('.pdf')]
    if forbidden:mismatches.append('PDF present in repo release: '+str(forbidden))
    frozen=root/'results/stage_a_readout_transfer_kill_v1/analysis/screening_decision.csv'
    if not frozen.is_file():mismatches.append('Frozen screening file missing')
    elif hashlib.sha256(frozen.read_bytes()).hexdigest()!='c163e061ce45efbec211e9d0ee999bd965b6c545da5e56f04d95b63f91b4a1ed':
        mismatches.append('Frozen screening checksum unexpected')
    print('Phase3/4 checked:',len(obj['new_files_sha256']),'new paths; failures:',len(mismatches))
    for m in mismatches: print('ERROR:',m)
    return 1 if mismatches else 0
if __name__=='__main__':sys.exit(main())
