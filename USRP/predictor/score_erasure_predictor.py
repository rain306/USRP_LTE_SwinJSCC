"""Produce a new transport bundle with CPU-predicted int8 chunk ranks.

Default selection is only images not used to fit the predictor. The original
NPYs and measured .erasure.json labels are never modified or copied as scores.
"""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from USRP.predictor.erasure_predictor import (SCHEME, chunk_features,
                                             load_source, predict)


def score(npy_dir, model_path, out_dir, split='heldout'):
    model_path = Path(model_path)
    model_bytes = model_path.read_bytes()
    model = json.loads(model_bytes)
    if model.get('scheme') != SCHEME:
        raise ValueError('Unsupported predictor model')
    if Path(out_dir).exists():
        raise FileExistsError('Use a new output bundle directory: ' + str(out_dir))
    chunk = int(model['chunk_bytes'])
    training = set(model['training_image_sha256'])
    heldout = set(model['heldout_image_sha256'])
    paths = sorted(Path(npy_dir).glob('*.npy'))
    if not paths:
        raise ValueError('No source NPYs: ' + str(npy_dir))

    selected = []
    for path in paths:
        raw, meta = load_source(path)
        source_hash = hashlib.sha256(raw).hexdigest()
        if split == 'heldout' and source_hash not in heldout:
            continue
        if split == 'unseen' and source_hash in training:
            continue
        contract = model['contract']
        if (meta.get('model') != contract['model'] or
                meta.get('C') != contract['C'] or
                meta.get('snr_db') != contract['snr'] or
                meta.get('storage', {}).get('format') != contract['storage_format']):
            raise ValueError('NPY does not match predictor training contract: ' + str(path))
        features, eligible = chunk_features(raw, meta, chunk)
        scores = predict(model, features, eligible)
        sidecar = dict(complete=True, provenance=dict(
            scheme=SCHEME, source_sha256=source_hash,
            model_sha256=hashlib.sha256(model_bytes).hexdigest(),
            chunk_bytes=chunk, total_chunks=len(scores),
            storage_format=contract['storage_format'],
            checkpoint_sha256=contract['checkpoint_sha256'],
            decoder_checkpoint_sha256=contract['decoder_checkpoint_sha256'],
            trained_on_image=source_hash in training),
            scores=scores.tolist())
        selected.append((path, sidecar))
    if not selected:
        raise ValueError('No NPYs selected by --split ' + split)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True)
    for path, sidecar in selected:
        shutil.copy2(path, out_dir/path.name)
        meta = path.with_suffix('.meta.json')
        if meta.exists():
            shutil.copy2(meta, out_dir/meta.name)
        destination = out_dir/(path.stem + '.predictor.json')
        destination.write_text(json.dumps(sidecar, ensure_ascii=False,
                                          separators=(',', ':'), allow_nan=False),
                               encoding='utf-8')
        print('[score] {} chunks={} training_image={}'.format(
            path.name, len(sidecar['scores']),
            sidecar['provenance']['trained_on_image']), flush=True)
    print('[done] bundle={} images={}'.format(out_dir, len(selected)), flush=True)
    return len(selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--npy-dir', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--split', choices=['heldout', 'unseen', 'all'],
                        default='heldout', help='all includes fitting images and cannot test generalization')
    args = parser.parse_args()
    def absolute(value):
        path = Path(value)
        return path if path.is_absolute() else PROJECT_ROOT/path
    score(absolute(args.npy_dir), absolute(args.model), absolute(args.out_dir),
          args.split)


if __name__ == '__main__':
    main()
