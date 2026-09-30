# -*- coding: utf-8 -*-
"""Offline, paired importance comparison. No sockets and no model training.

Models only DATA erasures; decoding metadata is assumed reliably available.
All selected policies send the same shuffled primary sequence and the same number
of repeat slots halfway through it. Chunk zero is protected by every policy.
"""
import argparse
import hashlib
import io
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from USRP.core_codec.swinjscc_codec import (SwinJSCCCodec, compute_psnr_mse,
                                            apply_erasure_placeholder)
from USRP.core_codec.decode_dataset import _ssim
from USRP.erasure_training.measure_erasure_importance import (load_sample, file_hash,
                                                              provenance, validate_cache)
from udp import send_raw as sender


def seed_for(seed, *parts):
    text = '|'.join(map(str, (seed,) + parts)).encode()
    return int.from_bytes(hashlib.sha256(text).digest()[:8], 'big')


def erasures(count, rate, model, burst_length, seed):
    rng = random.Random(seed)
    if model == 'independent':
        return [rng.random() < rate for _ in range(count)]
    # Two-state erasure process: every packet in the bad state is lost.
    # Stationary loss = rate, expected bad-state run length = burst_length.
    leave_bad = 1.0 / burst_length
    enter_bad = rate / ((1.0 - rate) * burst_length)
    if enter_bad > 1:
        raise ValueError('loss rate too high for the requested mean burst length')
    bad = rng.random() < rate
    lost = []
    for _ in range(count):
        lost.append(bad)
        u = rng.random()
        bad = (u >= leave_bad) if bad else (u < enter_bad)
    return lost


def ranked_a(scores, count):
    # Equal header protection: choose exactly count-1 other chunks.
    ranked = sorted(range(1, len(scores)), key=lambda q: (-scores[q], q))
    return [0] + ranked[:count - 1]


def concentrated_repeats(scores, budget):
    """Keep the same one header repeat; spend remaining slots twice per top block."""
    ranked = sorted(range(1, len(scores)), key=lambda q: (-scores[q], q))
    return [0] + [q for q in ranked for _ in range(2)][:budget-1]


def evaluate(codec, symbols, meta, reference):
    recon = codec.decode(symbols, meta['feature_shape'], meta['input_shape'][2:4],
                         orig_power=meta['original_symbol_power'], restore_power=True)
    if not np.isfinite(recon).all():
        raise ValueError('nonfinite reconstruction; refusing to silently sanitize metrics')
    psnr, mse = compute_psnr_mse(reference, recon)
    ssim = _ssim(reference, recon)
    if not math.isfinite(ssim):
        raise ValueError('SSIM unavailable or nonfinite')
    return dict(psnr_db=psnr if math.isfinite(psnr) else None,
                perfect_reconstruction=(mse == 0), mse=mse, ssim=ssim)


def save_results(path, settings, records):
    groups = defaultdict(list)
    for row in records:
        groups[(row['method'], row['loss_rate'])].append(row)
    summary = []
    for (method, loss_rate), rows in groups.items():
        item = dict(method=method, loss_rate=loss_rate, count=len(rows))
        for key in ('psnr_db', 'ssim', 'receive_rate_pct', 'actual_loss_rate'):
            values = [r[key] for r in rows if r.get(key) is not None]
            item['mean_' + key] = float(np.mean(values)) if values else None
        by_trial = defaultdict(list)
        for r in rows:
            if r['psnr_db'] is not None:
                by_trial[r['trial']].append(r['psnr_db'])
        trial_means = [float(np.mean(v)) for v in by_trial.values()]
        item['trial_mean_psnr_std_db'] = float(np.std(trial_means, ddof=1)) if len(trial_means) > 1 else None
        summary.append(item)
    payload = dict(settings=settings, complete=len(records)==settings.get('expected_records'), assumptions=[
        'Control metadata is reliable; only DATA packet erasures are simulated.',
        'All policies protect chunk 0 and use equal primary/repeat slots.',
        'Loss masks and primary order are shared across policies per image/trial.',
        'Burst model is synthetic, not fitted to the LTE logs.',
        'Energy/gradient are proxies; erasure measures one erased chunk, not the multi-loss optimum.',
        'Equal DATA packet slots, not necessarily identical byte totals: the final file chunk can be shorter.',
        'Concentrated policy repeats header once and top non-header blocks twice within the same repeat slots.'
    ], summary=summary, per_image=records)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)
    return summary


def main():
    p = argparse.ArgumentParser(description='Paired offline random/energy/sensitivity comparison')
    p.add_argument('--npy-dir', default='USRP/npy_in', help='Original, untransmitted complex64 NPY plus importance sidecars')
    p.add_argument('--out-dir', required=True, help='New run directory; must not already exist')
    p.add_argument('--ref-dir', default='datasets/Kodak24')
    p.add_argument('--checkpoint', default='weights/SwinJSCC_wo_SAandRA_AWGN_HRimage_snr10_psnr_C96.model')
    p.add_argument('--decoder-checkpoint', default=None,
                   help='Override decoder only; deliberately keep original erasure ranking for a controlled comparison')
    p.add_argument('--model', default='SwinJSCC_w/o_SAandRA')
    p.add_argument('--C', type=int, default=96)
    p.add_argument('--snr', type=int, default=10)
    p.add_argument('--device', default=None)
    p.add_argument('--chunk', type=int, default=1200)
    p.add_argument('--top-frac', type=float, default=.1)
    p.add_argument('--erasure-fill', type=float, default=0.0,
                   help='显式擦除占位符取值；0.0 复现原生补零。非零占位符只有在解码器'
                        '经过擦除感知训练时才有望带来提升。')
    p.add_argument('--loss-rates', default='0.1,0.2')
    p.add_argument('--loss-model', choices=['independent', 'burst'], default='independent')
    p.add_argument('--burst-length', type=float, default=5.)
    p.add_argument('--trials', type=int, default=3)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--methods', default='random,energy,sensitivity',
                   help='Comma-separated random,energy,sensitivity,erasure,erasure_concentrated; erasure policies require measured sidecars')
    p.add_argument('--ranking-erasure-fill', type=float, default=None,
                   help='Explicitly keep scores measured with this fill (e.g. 0) while evaluating another fill; default matches evaluation')
    a = p.parse_args()
    if not math.isfinite(a.erasure_fill):
        p.error('erasure-fill must be finite')
    ranking_fill = a.erasure_fill if a.ranking_erasure_fill is None else a.ranking_erasure_fill
    if not math.isfinite(ranking_fill):
        p.error('ranking-erasure-fill must be finite')
    methods = list(dict.fromkeys(x.strip() for x in a.methods.split(',')))
    if not methods or any(x not in ('random', 'energy', 'sensitivity', 'erasure', 'erasure_concentrated') for x in methods):
        p.error('Unknown or empty --methods')
    needs_erasure = any(x.startswith('erasure') for x in methods)
    rates = list(dict.fromkeys(float(x) for x in a.loss_rates.split(',')))
    if not rates or any(not 0 <= x < 1 for x in rates):
        p.error('loss rates must be in [0, 1)')
    if a.trials < 1 or a.chunk < 8 or a.chunk % 8 or not 0 < a.top_frac <= 1:
        p.error('positive trials, chunk multiple of 8, and top-frac in (0, 1] required')
    if not math.isfinite(a.burst_length) or a.burst_length < 1 or (a.limit is not None and a.limit < 1):
        p.error('burst-length >= 1 and limit >= 1 required')
    for rate in rates:
        erasures(1, rate, a.loss_model, a.burst_length, a.seed)
    root = lambda name: (PROJECT_ROOT / name).resolve()
    files = sorted(root(a.npy_dir).glob('*.npy'))
    if a.limit:
        files = files[:a.limit]
    if not files:
        p.error('No source NPY files found')
    for f in files:
        if 'sensitivity' in methods and not f.with_suffix('.importance.json').is_file():
            p.error('Missing importance file: ' + str(f.with_suffix('.importance.json')))
        if needs_erasure and not f.with_suffix('.erasure.json').is_file():
            p.error('Missing erasure file: ' + str(f.with_suffix('.erasure.json')))
    erasure_cache_hashes = {}
    if needs_erasure:
        checkpoint_hash = file_hash(root(a.checkpoint))
        for f in files:
            raw, _, _, meta, reference = load_sample(f, root(a.ref_dir))
            expected = provenance(raw, meta, reference, a.chunk, checkpoint_hash, a.model, a.C, a.snr, ranking_fill)
            cache = f.with_suffix('.erasure.json')
            validate_cache(json.loads(cache.read_text(encoding='utf-8')), expected, complete=True)
            erasure_cache_hashes[f.name] = file_hash(cache)
    # Detect missing/broken SSIM before loading the model.
    _ssim(np.zeros((8, 8, 3)), np.zeros((8, 8, 3)))
    out = root(a.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    codec = SwinJSCCCodec(a.checkpoint, model=a.model, C=a.C, snr=a.snr, device=a.device,
                         decoder_checkpoint=a.decoder_checkpoint)
    records = []
    settings = vars(a).copy()
    settings['evaluation_decoder_sha256'] = file_hash(root(a.decoder_checkpoint or a.checkpoint))
    settings['ranking_policy'] = 'Original cached importance retained; decoder override does not recompute ranking'
    settings['files'] = [f.name for f in files]
    settings['source_sha256'] = {}
    settings['selected_methods'] = methods
    settings['ranking_erasure_fill_effective'] = ranking_fill
    settings['erasure_cache_sha256'] = erasure_cache_hashes
    settings['expected_records'] = len(files)*(1+len(rates)*a.trials*len(methods))
    print('images={} trials={} loss_rates={} model={}'.format(len(files), a.trials, rates, a.loss_model), flush=True)
    for f in files:
        raw = f.read_bytes()
        settings['source_sha256'][f.name] = hashlib.sha256(raw).hexdigest()
        meta_payload = sender.meta_load(raw, str(f))
        if not meta_payload:
            raise ValueError('Missing decode metadata: ' + str(f))
        meta = json.loads(meta_payload)
        body, _ = sender.split(raw)
        symbols = np.load(io.BytesIO(body), allow_pickle=False)
        if symbols.dtype != np.dtype('complex64') or symbols.ndim != 1:
            raise ValueError('Expected one-dimensional complex64 NPY')
        head = len(body) - symbols.nbytes
        if head <= 0:
            raise ValueError('Invalid NPY header offset')
        ref_stem = meta.get('image_stem', f.stem)
        ref = next((root(a.ref_dir) / (ref_stem + ext) for ext in ('.png', '.jpg', '.jpeg', '.bmp')
                    if (root(a.ref_dir) / (ref_stem + ext)).is_file()), None)
        if ref is None:
            raise ValueError('Reference image missing: ' + ref_stem)
        h, w = meta['input_shape'][2:4]
        with Image.open(ref) as im:
            im = im.convert('RGB')
            if im.width < w or im.height < h:
                raise ValueError('Reference smaller than encoded image')
            left, top = (im.width - w) // 2, (im.height - h) // 2
            reference = np.asarray(im.crop((left, top, left+w, top+h)), dtype=np.float64)
        n = (len(raw) + a.chunk - 1) // a.chunk
        k = max(1, round(n * a.top_frac))
        ranked = {}
        if 'energy' in methods:
            ranked['energy'] = ranked_a(sender.scores(raw, a.chunk), k)
        if 'sensitivity' in methods:
            ranked['sensitivity'] = ranked_a(sender.sensitivity_scores(raw, str(f), a.chunk)[0], k)
        if needs_erasure:
            scores = sender.erasure_scores(raw, str(f), a.chunk)[0]
            ranked['erasure'] = ranked_a(scores, k)
            ranked['erasure_concentrated'] = concentrated_repeats(scores, k)
        intact = evaluate(codec, symbols, meta, reference)
        records.append(dict(image=f.name, method='intact', loss_rate=0., trial=0,
                            receive_rate_pct=100., actual_loss_rate=0., data_packets=n, **intact))
        for rate in rates:
            for trial in range(1, a.trials + 1):
                rng = random.Random(seed_for(a.seed, f.name, trial, 'schedule'))
                primary = list(range(1, n))
                rng.shuffle(primary)
                primary.insert(0, 0)
                perm = list(range(k))
                rng.shuffle(perm)
                random_a = [0] + rng.sample(list(range(1, n)), k - 1)
                mask_seed = seed_for(a.seed, f.name, rate, trial, a.loss_model)
                lost = erasures(n+k, rate, a.loss_model, a.burst_length, mask_seed)
                for method in methods:
                    ids = random_a if method == 'random' else ranked[method]
                    ids = sorted(ids)
                    repeats = [ids[i] for i in perm]
                    mid = max(1, n // 2)
                    schedule = primary[:mid] + repeats + primary[mid:]
                    got = {q for q, drop in zip(schedule, lost) if not drop}
                    damaged = bytearray(len(raw))
                    for q in got:
                        start = q * a.chunk
                        damaged[start:start+a.chunk] = raw[start:start+a.chunk]
                    # Original header/metadata are known; preserve zeros in lost
                    # payloads even when seq=0 or the embedded trailer is lost.
                    received = np.frombuffer(damaged, dtype=symbols.dtype,
                                             count=symbols.size, offset=head).copy()
                    if a.erasure_fill != 0.0:
                        lost_chunks = [q for q in range(n) if q not in got]
                        received = apply_erasure_placeholder(received, head, a.chunk,
                                                             lost_chunks, fill=a.erasure_fill)
                    score = evaluate(codec, received, meta, reference)
                    row = dict(image=f.name, method=method, loss_rate=rate, trial=trial,
                               mask_seed=mask_seed, data_packets=n+k, total_chunks=n,
                               unique_chunks=len(got), a_chunks=len(set(ids)), repeat_packets=k,
                               data_payload_bytes=sum(min(a.chunk, len(raw)-q*a.chunk) for q in schedule),
                               receive_rate_pct=100*len(got)/n,
                               a_receive_rate_pct=100*len(set(ids) & got)/len(set(ids)),
                               actual_loss_rate=sum(lost)/len(lost), **score)
                    records.append(row)
                    print('{} p={:.2f} trial={} {} recv={:.1f}% PSNR={} SSIM={:.4f}'.format(
                        f.stem, rate, trial, method, row['receive_rate_pct'],
                        '{:.2f}'.format(score['psnr_db']) if score['psnr_db'] is not None else 'inf', score['ssim']), flush=True)
            save_results(out / 'metrics.json', settings, records)
    summary = save_results(out / 'metrics.json', settings, records)
    for row in summary:
        print(json.dumps(row, ensure_ascii=False), flush=True)
    print('Saved: ' + str(out / 'metrics.json'), flush=True)


if __name__ == '__main__':
    main()
