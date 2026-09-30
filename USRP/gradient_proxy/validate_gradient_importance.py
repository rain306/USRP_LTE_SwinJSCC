"""CPU-only rank check against already measured single-chunk erasure scores.

This is a diagnostic, not proof of OTA quality or a substitute for new-image
validation. It never changes NPY or importance sidecars.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from USRP.core_codec.latent_storage import validate as validate_int8
from udp import send_raw as sender


def average_ranks(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind='mergesort')
    ranks = np.empty(len(values), dtype=np.float64)
    index = 0
    while index < len(order):
        stop = index + 1
        while stop < len(order) and values[order[stop]] == values[order[index]]:
            stop += 1
        ranks[order[index:stop]] = (index + stop - 1)/2.0
        index = stop
    return ranks


def rank_report(approximate, measured, top_fraction=0.10, eligible=None):
    if len(approximate) != len(measured) or len(approximate) < 3:
        raise ValueError('rank arrays must have equal length >= 3')
    # Chunk 0 is always protected independently of its score.
    eligible = list(range(1, len(measured))) if eligible is None else list(eligible)
    if not eligible or any(i <= 0 or i >= len(measured) for i in eligible):
        raise ValueError('no valid DATA-chunk indices to rank')
    top = max(1, round(len(measured)*top_fraction))-1
    if top < 1:
        raise ValueError('too few chunks for a top-fraction ranking test')
    top = min(top, len(eligible))
    actual = set(sorted(eligible, key=lambda i: (-measured[i], i))[:top])
    predicted = set(sorted(eligible, key=lambda i: (-approximate[i], i))[:top])
    a = average_ranks([approximate[i] for i in eligible])
    b = average_ranks([measured[i] for i in eligible])
    rho = float(np.corrcoef(a, b)[0, 1]) if np.std(a) and np.std(b) else float('nan')
    return dict(spearman=rho, top_k=top,
                top_overlap=len(actual & predicted),
                top_recall=len(actual & predicted)/top,
                random_recall=top/len(eligible))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--npy-dir', required=True)
    parser.add_argument('--chunk', type=int, default=1200)
    parser.add_argument('--top-frac', type=float, default=0.10)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    if not 0 < args.top_frac < 1:
        parser.error('--top-frac must be in (0,1)')
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    directory = (PROJECT_ROOT / args.npy_dir).resolve()
    files = sorted(directory.glob('*.npy'))
    if args.limit:
        files = files[:args.limit]
    if not files:
        parser.error('No NPY files in --npy-dir')
    reports = []
    for path in files:
        raw = path.read_bytes()
        proxy, _ = sender.gradient_scores(raw, str(path), args.chunk)
        oracle, _ = sender.erasure_scores(raw, str(path), args.chunk)
        meta = json.loads(sender.meta_load(raw, str(path)))
        n_symbols, header, _ = validate_int8(meta)
        payload_end = header + 2*n_symbols
        eligible = [i for i in range(1, len(proxy))
                    if max(header, i*args.chunk) < min(payload_end, (i+1)*args.chunk)]
        report = rank_report(proxy, oracle, args.top_frac, eligible)
        reports.append(report)
        print('{}: Spearman={:.3f}, Top{} overlap={}/{} ({:.1%}), random={:.1%}'.format(
            path.name, report['spearman'], report['top_k'],
            report['top_overlap'], report['top_k'],
            report['top_recall'], report['random_recall']))
    if len(reports) > 1:
        print('Mean Spearman={:.3f}, mean Top recall={:.1%} ({} images)'.format(
            float(np.nanmean([r['spearman'] for r in reports])),
            float(np.mean([r['top_recall'] for r in reports])), len(reports)))


if __name__ == '__main__':
    main()
