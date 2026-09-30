"""Paired, offline int8 DATA-loss comparison: energy versus predictor.

This does not simulate the LTE stack or prove OTA superiority. Both methods
share a packet-loss mask and the same DATA repeat budget for each image/trial.
Control META and header interpretation are assumed available to the decoder.
"""
import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from USRP.core_codec.decode_dataset import _ssim
from USRP.predictor.erasure_predictor import sender
from USRP.core_codec.latent_storage import load_int8
from USRP.erasure_training.measure_erasure_importance import file_hash, load_sample
from USRP.core_codec.swinjscc_codec import SwinJSCCCodec, compute_psnr_mse


def seed_for(seed, *parts):
    payload = '|'.join(map(str, (seed,) + parts)).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], 'big')


def selected(scores, count):
    return [0] + sorted(range(1, len(scores)),
                        key=lambda index: (-scores[index], index))[:count-1]


def receive_symbols(raw, meta, chunk, schedule, lost):
    got = {seq for seq, missing in zip(schedule, lost) if not missing}
    damaged = bytearray(len(raw))
    for seq in got:
        lo = seq*chunk
        damaged[lo:lo+chunk] = raw[lo:lo+chunk]
    return load_int8(bytes(damaged), meta), len(got)


def metrics(codec, symbols, meta, reference):
    h, w = meta['input_shape'][2:4]
    reconstruction = codec.decode(symbols, meta['feature_shape'], (h, w),
                                  orig_power=meta['original_symbol_power'],
                                  restore_power=True)
    if not np.isfinite(reconstruction).all():
        raise ValueError('Non-finite reconstruction')
    psnr, mse = compute_psnr_mse(reference, reconstruction)
    ssim = _ssim(reference, reconstruction)
    if not math.isfinite(mse) or not math.isfinite(ssim):
        raise ValueError('Non-finite reconstruction metric')
    return dict(psnr_db=float(psnr) if math.isfinite(psnr) else None,
                mse=float(mse), ssim=float(ssim))


def evaluate(npy_dir, ref_dir, out_dir, checkpoint, decoder_checkpoint,
             chunk=1200, top_fraction=.10, loss_rates=(.05, .10), trials=3,
             seed=2026, device='cuda', limit=None):
    if Path(out_dir).exists():
        raise FileExistsError('Use a new simulation output directory: ' + str(out_dir))
    if chunk <= 0 or chunk % 8 or not 0 < top_fraction < 1 or trials < 1:
        raise ValueError('Invalid chunk/top fraction/trials')
    if not loss_rates or any(not 0 <= rate < 1 for rate in loss_rates):
        raise ValueError('Loss rates must lie in [0,1)')
    files = sorted(Path(npy_dir).glob('*.npy'))
    if limit is not None:
        if limit < 1:
            raise ValueError('--limit must be positive')
        files = files[:limit]
    if not files:
        raise ValueError('No held-out NPYs')
    encoder_hash = file_hash(checkpoint)
    decoder_hash = file_hash(decoder_checkpoint)
    prepared = []
    for path in files:
        raw, symbols, _, meta, reference = load_sample(path, Path(ref_dir),
                                                       allow_int8=True)
        if meta.get('storage', {}).get('format') != 'swinjscc_int8_v1':
            raise ValueError('Expected original int8 NPY: ' + str(path))
        predicted, _ = sender.predictor_scores(raw, str(path), chunk)
        sidecar = json.loads(path.with_suffix('.predictor.json').read_text(encoding='utf-8'))
        if sidecar['provenance'].get('trained_on_image') is not False:
            raise ValueError('Refusing training-image simulation as held-out evidence: ' + str(path))
        if (sidecar['provenance'].get('checkpoint_sha256') != encoder_hash or
                sidecar['provenance'].get('decoder_checkpoint_sha256') != decoder_hash):
            raise ValueError('Predictor was trained for different codec weights: ' + str(path))
        energy, _ = sender.energy_scores(raw, chunk, 'int')
        prepared.append((path, raw, symbols, meta, reference, energy, predicted))

    codec = SwinJSCCCodec(str(checkpoint), model='SwinJSCC_w/o_SAandRA',
                         C=96, snr=10, device=device,
                         decoder_checkpoint=str(decoder_checkpoint))
    records = []
    for path, raw, symbols, meta, reference, energy, predicted in prepared:
        n = (len(raw)+chunk-1)//chunk
        repeats = max(1, round(n*top_fraction))
        choices = dict(energy=selected(energy, repeats),
                       predictor=selected(predicted, repeats))
        intact = metrics(codec, symbols, meta, reference)
        print('[image] {} chunks={} clean_PSNR={}'.format(
            path.name, n, intact['psnr_db']), flush=True)
        for rate in loss_rates:
            for trial in range(1, trials+1):
                rng = random.Random(seed_for(seed, path.name, trial, 'schedule'))
                primary = list(range(1, n))
                rng.shuffle(primary)
                primary.insert(0, 0)
                repeat_order = list(range(repeats))
                rng.shuffle(repeat_order)
                mask_rng = random.Random(seed_for(seed, path.name, rate,
                                                   trial, 'loss'))
                lost = [mask_rng.random() < rate for _ in range(n+repeats)]
                for method, protected in choices.items():
                    secondary = [protected[index] for index in repeat_order]
                    mid = max(1, n//2)
                    schedule = primary[:mid] + secondary + primary[mid:]
                    received, unique = receive_symbols(raw, meta, chunk,
                                                       schedule, lost)
                    result = metrics(codec, received, meta, reference)
                    records.append(dict(file=path.name, method=method,
                                        loss_rate=rate, trial=trial,
                                        total_chunks=n, unique_chunks=unique,
                                        repeat_packets=repeats, **result))
                    print('[trial] {} p={} t={} {} PSNR={:.3f} SSIM={:.4f}'.format(
                        path.name, rate, trial, method,
                        result['psnr_db'] or float('nan'), result['ssim']), flush=True)
    summary = []
    for rate in loss_rates:
        pairs = {}
        for item in records:
            if item['loss_rate'] == rate:
                pairs.setdefault((item['file'], item['trial']), {})[item['method']] = item
        if any(set(pair) != {'energy', 'predictor'} for pair in pairs.values()):
            raise ValueError('Missing paired method result')
        summary.append(dict(loss_rate=rate, pairs=len(pairs),
                            mean_psnr_delta_db=float(np.mean([
                                pair['predictor']['psnr_db']-pair['energy']['psnr_db']
                                for pair in pairs.values()])),
                            mean_ssim_delta=float(np.mean([
                                pair['predictor']['ssim']-pair['energy']['ssim']
                                for pair in pairs.values()]))))
    output = dict(assumptions=[
        'Synthetic independent DATA packet losses; not measured from LTE.',
        'Paired loss masks and equal DATA repeat slots for both policies.',
        'Control META and header interpretation assumed available.',
        'Source files are original encoded int8 NPYs, not OTA received NPYs.'
    ], settings=dict(chunk_bytes=chunk, top_fraction=top_fraction,
                     loss_rates=list(loss_rates), trials=trials, seed=seed,
                     checkpoint=str(checkpoint),
                     decoder_checkpoint=str(decoder_checkpoint),
                     source_files=[path.name for path, *_ in prepared]),
                  summary=summary, per_image_trial=records)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True)
    (out_dir/'metrics.json').write_text(json.dumps(output, ensure_ascii=False,
                                                   indent=2, allow_nan=False),
                                        encoding='utf-8')
    for item in summary:
        print('[summary] p={} paired={} delta_PSNR={:+.4f}dB '
              'delta_SSIM={:+.5f}'.format(
                  item['loss_rate'], item['pairs'], item['mean_psnr_delta_db'],
                  item['mean_ssim_delta']), flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--npy-dir', required=True)
    parser.add_argument('--ref-dir', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--decoder-checkpoint', required=True)
    parser.add_argument('--chunk', type=int, default=1200)
    parser.add_argument('--top-frac', type=float, default=.10)
    parser.add_argument('--loss-rates', default='0.05,0.10')
    parser.add_argument('--trials', type=int, default=3)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--limit', type=int, help='First N images for a small smoke run')
    args = parser.parse_args()
    def absolute(value):
        path = Path(value)
        return path if path.is_absolute() else PROJECT_ROOT/path
    rates = [float(item) for item in args.loss_rates.split(',')]
    evaluate(absolute(args.npy_dir), absolute(args.ref_dir), absolute(args.out_dir),
             absolute(args.checkpoint), absolute(args.decoder_checkpoint),
             args.chunk, args.top_frac, rates, args.trials, args.seed,
             args.device, args.limit)


if __name__ == '__main__':
    main()
