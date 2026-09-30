"""Train a cheap int8 DATA-chunk ranker from complete measured erasure labels.

All splits are by *image*, never by chunk. The held-out images are not fitted.
This script does not call a decoder, transmit packets, or alter source NPYs.
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from USRP.predictor.erasure_predictor import (FEATURE_NAMES, FEATURE_VERSION,
                                             SCHEME, chunk_features, load_source,
                                             percentile_targets, predict, rank_report,
                                             read_labels, sender)


def train(npy_dir, out_dir, chunk=1200, heldout_count=6, seed=2026,
          alpha=0.05, top_fraction=0.10, expected_images=None):
    paths = sorted(Path(npy_dir).glob('*.npy'))
    if expected_images is not None and len(paths) != expected_images:
        raise ValueError('Expected {} NPY images, found {}'.format(expected_images, len(paths)))
    if len(paths) < 3 or not 1 <= heldout_count < len(paths)-1:
        raise ValueError('Need at least two training images and one held-out image')
    if alpha <= 0 or not math.isfinite(alpha) or not 0 < top_fraction < 1:
        raise ValueError('Invalid alpha/top fraction')
    if Path(out_dir).exists():
        raise FileExistsError('Use a new predictor output directory: ' + str(out_dir))

    rows = []
    contract = None
    for path in paths:
        raw, meta = load_source(path)
        features, eligible = chunk_features(raw, meta, chunk)
        labels, spec = read_labels(path, raw, meta, chunk, eligible)
        current = {key: spec.get(key) for key in
                   ('checkpoint_sha256', 'decoder_checkpoint_sha256',
                    'model', 'C', 'snr', 'storage_format', 'bytes_per_symbol')}
        if any(value is None for value in current.values()):
            raise ValueError('Erasure labels lack a common model/decoder identity: ' + str(path))
        if contract is None:
            contract = current
        elif contract != current:
            raise ValueError('Mixed encoder/decoder/format label sets: ' + str(path))
        rows.append(dict(path=path, sha256=hashlib.sha256(raw).hexdigest(),
                         raw=raw, features=features, eligible=eligible,
                         labels=labels))

    rng = np.random.RandomState(seed)
    shuffled = rng.permutation(len(rows))
    heldout_ids = set(int(i) for i in shuffled[:heldout_count])
    train_rows = [row for i, row in enumerate(rows) if i not in heldout_ids]
    heldout_rows = [row for i, row in enumerate(rows) if i in heldout_ids]

    x_parts, y_parts, w_parts = [], [], []
    for row in train_rows:
        usable = row['eligible']
        x_parts.append(row['features'][usable])
        y_parts.append(percentile_targets(row['labels'][usable]))
        w_parts.append(np.full(int(usable.sum()), 1.0/int(usable.sum())))
    x = np.concatenate(x_parts)
    y = np.concatenate(y_parts)
    weights = np.concatenate(w_parts)
    weight_sum = float(weights.sum())
    mean = np.average(x, axis=0, weights=weights)
    variance = np.average((x-mean)**2, axis=0, weights=weights)
    std = np.sqrt(variance)
    std[std < 1e-8] = 1.0
    z = (x-mean)/std
    intercept = float(np.average(y, weights=weights))
    gram = (z.T * weights) @ z / weight_sum
    rhs = (z.T * weights) @ (y-intercept) / weight_sum
    coef = np.linalg.solve(gram + alpha*np.eye(len(FEATURE_NAMES)), rhs)
    model = dict(scheme=SCHEME, feature_version=FEATURE_VERSION,
                 feature_names=FEATURE_NAMES, chunk_bytes=chunk,
                 contract=contract, alpha=alpha, seed=seed,
                 training_image_sha256=[row['sha256'] for row in train_rows],
                 heldout_image_sha256=[row['sha256'] for row in heldout_rows],
                 training_files=[row['path'].name for row in train_rows],
                 heldout_files=[row['path'].name for row in heldout_rows],
                 feature_mean=mean.tolist(), feature_std=std.tolist(),
                 coef=coef.tolist(), intercept=intercept,
                 target='within_image_rank_of_measured_single_chunk_delta_mse',
                 caveat='Not a measured erasure score, joint-loss optimum, or OTA result')

    reports = []
    for row in heldout_rows:
        predicted = predict(model, row['features'], row['eligible'])
        energy, _ = sender.energy_scores(row['raw'], chunk, 'int')
        energy_report = rank_report(energy, row['labels'], row['eligible'], top_fraction)
        predictor_report = rank_report(predicted, row['labels'], row['eligible'], top_fraction)
        reports.append(dict(file=row['path'].name, source_sha256=row['sha256'],
                            predictor=predictor_report, energy=energy_report))
        print('[heldout] {} top-recall predictor={:.1%} energy={:.1%} '
              'sum-delta predictor={:.4f} energy={:.4f}'.format(
                  row['path'].name, predictor_report['top_recall'],
                  energy_report['top_recall'],
                  predictor_report['selected_delta_sum'],
                  energy_report['selected_delta_sum']), flush=True)
    report = dict(scheme=SCHEME, training_images=len(train_rows),
                  heldout_images=len(heldout_rows), top_fraction=top_fraction,
                  note=('Held-out images are not fitted. Sum of single-chunk '
                        'delta MSE is a rank diagnostic, not multi-loss PSNR.'),
                  per_image=reports,
                  mean_top_recall_predictor=float(np.mean([
                      item['predictor']['top_recall'] for item in reports])),
                  mean_top_recall_energy=float(np.mean([
                      item['energy']['top_recall'] for item in reports])),
                  mean_selected_delta_predictor=float(np.mean([
                      item['predictor']['selected_delta_sum'] for item in reports])),
                  mean_selected_delta_energy=float(np.mean([
                      item['energy']['selected_delta_sum'] for item in reports])))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True)
    (out_dir/'model.json').write_text(json.dumps(model, ensure_ascii=False, indent=2,
                                               allow_nan=False), encoding='utf-8')
    (out_dir/'validation.json').write_text(json.dumps(report, ensure_ascii=False,
                                                    indent=2, allow_nan=False), encoding='utf-8')
    print('[done] train={} heldout={} mean Top recall predictor={:.1%} energy={:.1%}; '
          'mean selected single-chunk delta sum predictor={:.4f} energy={:.4f}'.format(
        len(train_rows), len(heldout_rows), report['mean_top_recall_predictor'],
        report['mean_top_recall_energy'], report['mean_selected_delta_predictor'],
        report['mean_selected_delta_energy']), flush=True)
    print('[model] ' + str(out_dir/'model.json'), flush=True)
    return model, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--npy-dir', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--chunk', type=int, default=1200)
    parser.add_argument('--heldout-count', type=int, default=6)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--top-frac', type=float, default=0.10)
    parser.add_argument('--expected-images', type=int)
    args = parser.parse_args()
    root = PROJECT_ROOT
    source = Path(args.npy_dir)
    target = Path(args.out_dir)
    train(source if source.is_absolute() else root/source,
          target if target.is_absolute() else root/target,
          args.chunk, args.heldout_count, args.seed, args.alpha,
          args.top_frac, args.expected_images)


if __name__ == '__main__':
    main()
