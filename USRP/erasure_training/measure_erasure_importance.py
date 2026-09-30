"""Measure single DATA-chunk erasure distortion, without changing source NPYs.

Control/header metadata is assumed available. Scores are MSE(erased)-MSE(intact),
not an upper bound or an optimum for simultaneous erasures. No training occurs.
"""
import argparse
import hashlib
import io
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from USRP.core_codec.latent_storage import is_int8, load_int8, validate as validate_int8
from USRP.core_codec.swinjscc_codec import (SwinJSCCCodec, compute_psnr_mse,
                                            apply_erasure_placeholder)
from udp import send_raw as sender

SCHEME = 'single_chunk_erasure_delta_mse_v1'


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_sample(path, ref_dir, allow_int8=False):
    raw = path.read_bytes()
    payload = sender.meta_load(raw, str(path))
    if not payload:
        raise ValueError('Missing decode metadata: ' + str(path))
    meta = json.loads(payload)
    body, _ = sender.split(raw)
    if is_int8(meta):
        if not allow_int8:
            raise ValueError('This workflow requires original complex64 input: ' + str(path))
        array = np.load(io.BytesIO(body), allow_pickle=False)
        n, head, _ = validate_int8(meta)
        if array.dtype != np.int8 or array.shape != (n, 2) or len(body) != head+array.nbytes:
            raise ValueError('Int8 NPY does not match metadata')
        symbols = load_int8(raw, meta)
    else:
        symbols = np.load(io.BytesIO(body), allow_pickle=False)
        if symbols.ndim != 1 or symbols.dtype != np.dtype('complex64'):
            raise ValueError('Expected one-dimensional complex64 NPY: ' + str(path))
        head = len(body) - symbols.nbytes
        if head <= 0 or head % 8:
            raise ValueError('Invalid or unaligned NPY header')
    stem = meta.get('image_stem', path.stem)
    ref = next((ref_dir / (stem + ext) for ext in ('.png', '.jpg', '.jpeg', '.bmp')
                if (ref_dir / (stem + ext)).is_file()), None)
    if ref is None:
        raise ValueError('Reference image missing: ' + stem)
    h, w = meta['input_shape'][2:4]
    with Image.open(ref) as im:
        im = im.convert('RGB')
        if im.width < w or im.height < h:
            raise ValueError('Reference smaller than encoded image')
        left, top = (im.width-w)//2, (im.height-h)//2
        reference = np.asarray(im.crop((left, top, left+w, top+h)), dtype=np.float64)
    return raw, symbols, head, meta, reference


def provenance(raw, meta, reference, chunk, checkpoint_sha256, model, channels, snr,
               erasure_fill=0.0):
    d = dict(scheme=SCHEME, source_sha256=hashlib.sha256(raw).hexdigest(),
             chunk_bytes=chunk, total_chunks=(len(raw)+chunk-1)//chunk,
             checkpoint_sha256=checkpoint_sha256, model=model, C=channels, snr=snr,
             metadata_sha256=hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest(),
             reference_sha256=hashlib.sha256(reference.astype('<f8').tobytes()).hexdigest(),
             restore_power=True, pixel_range=255, clipping='codec.decode [0,255]')
    # 只记录非默认占位符，保证既有 fill=0 缓存继续有效。
    if erasure_fill != 0.0:
        d['erasure_fill'] = erasure_fill
    return d


def validate_cache(data, expected, complete=False):
    if data.get('provenance') != expected:
        raise ValueError('Erasure cache settings/input mismatch; regenerate with --overwrite')
    n = expected['total_chunks']
    scores = data.get('scores')
    if not isinstance(scores, list) or len(scores) > n or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in scores):
        raise ValueError('Invalid erasure scores')
    if not isinstance(data.get('intact_mse'), (int, float)) or not math.isfinite(data['intact_mse']):
        raise ValueError('Invalid intact MSE')
    if complete and (data.get('complete') is not True or len(scores) != n):
        raise ValueError('Incomplete erasure cache')


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def erased_symbols(symbols, head, chunk, seq, fill=0.0, bytes_per_symbol=8):
    """Erase only the latent bytes overlapping this exact file-byte DATA chunk.

    fill == 0.0 复现原生补零；fill != 0.0 写入显式擦除占位符（见
    swinjscc_codec.apply_erasure_placeholder）。
    """
    if bytes_per_symbol not in (2, 8) or chunk <= 0 or chunk % bytes_per_symbol or head % bytes_per_symbol:
        raise ValueError('Erasure requires aligned chunk/header')
    if bytes_per_symbol == 2 and fill != 0:
        raise ValueError('int8 supports only zero fill')
    lo = max(head, seq*chunk)
    hi = min(head+symbols.size*bytes_per_symbol, (seq+1)*chunk)
    damaged = symbols.copy()
    if hi > lo:
        damaged[(lo-head)//bytes_per_symbol:(hi-head)//bytes_per_symbol] = 0
    if fill != 0.0:
        damaged = apply_erasure_placeholder(damaged, head, chunk, [seq], fill=fill)
    return damaged


def reconstruction_mse(codec, symbols, meta, reference):
    recon = codec.decode(symbols, meta['feature_shape'], meta['input_shape'][2:4],
                         orig_power=meta['original_symbol_power'], restore_power=True)
    if not np.isfinite(recon).all():
        raise ValueError('Nonfinite reconstruction')
    mse = float(compute_psnr_mse(reference, recon)[1])
    if not math.isfinite(mse):
        raise ValueError('Nonfinite MSE')
    return mse


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--npy-dir', default='USRP/npy_in')
    p.add_argument('--ref-dir', default='datasets/Kodak24')
    p.add_argument('--checkpoint', default='weights/SwinJSCC_wo_SAandRA_AWGN_HRimage_snr10_psnr_C96.model')
    p.add_argument('--model', default='SwinJSCC_w/o_SAandRA')
    p.add_argument('--C', type=int, default=96)
    p.add_argument('--snr', type=int, default=10)
    p.add_argument('--device', default=None)
    p.add_argument('--decoder-checkpoint', default=None, help='Optional decoder override for measuring current reconstruction loss')
    p.add_argument('--chunk', type=int, default=1200)
    p.add_argument('--limit', type=int)
    p.add_argument('--progress-every', type=int, default=25)
    p.add_argument('--overwrite', action='store_true', help='Explicitly replace incompatible or existing caches')
    p.add_argument('--erasure-fill', type=float, default=0.0,
                   help='显式擦除占位符取值；0.0 保持原生补零。非零占位符只有在解码器'
                        '经过擦除感知训练时才有望带来提升。')
    a = p.parse_args()
    if not math.isfinite(a.erasure_fill):
        p.error('erasure-fill must be finite')
    if a.chunk < 8 or a.chunk > 1400 or a.chunk % 8 or a.progress_every < 1 or (a.limit is not None and a.limit < 1):
        p.error('chunk must be a multiple of 8 in [8,1400]; progress-every and limit must be positive')
    root = lambda x: (PROJECT_ROOT / x).resolve()
    files = sorted(root(a.npy_dir).glob('*.npy'))
    if a.limit:
        files = files[:a.limit]
    if not files:
        p.error('No source NPY files')
    checkpoint_hash = file_hash(root(a.checkpoint))
    decoder_hash = file_hash(root(a.decoder_checkpoint)) if a.decoder_checkpoint else None
    codec = None
    print('images={} chunk={} ; original NPYs will not be modified'.format(len(files), a.chunk), flush=True)
    for path in files:
        raw, symbols, head, meta, reference = load_sample(path, root(a.ref_dir), allow_int8=True)
        width = 2 if is_int8(meta) else 8
        if width == 2 and a.erasure_fill != 0:
            p.error('int8 supports only --erasure-fill 0')
        expected = provenance(raw, meta, reference, a.chunk, checkpoint_hash, a.model, a.C, a.snr, a.erasure_fill)
        if width == 2:
            expected['storage_format'] = meta['storage']['format']
            expected['bytes_per_symbol'] = width
        if a.decoder_checkpoint:
            expected['decoder_checkpoint_sha256'] = decoder_hash
        dest = path.with_suffix('.erasure.json')
        partial = path.with_suffix('.erasure.partial.json')
        if dest.exists() and not a.overwrite:
            validate_cache(json.loads(dest.read_text(encoding='utf-8')), expected, complete=True)
            print('[skip] ' + path.name + ' complete and verified', flush=True)
            continue
        if partial.exists() and not a.overwrite:
            result = json.loads(partial.read_text(encoding='utf-8'))
            validate_cache(result, expected)
        else:
            result = None
        if codec is None:
            codec = SwinJSCCCodec(a.checkpoint, model=a.model, C=a.C, snr=a.snr, device=a.device,
                                 decoder_checkpoint=a.decoder_checkpoint)
        if result is None:
            intact = reconstruction_mse(codec, symbols, meta, reference)
            result = dict(provenance=expected, complete=False, intact_mse=intact, scores=[])
            atomic_json(partial, result)
        start = len(result['scores'])
        begun = time.perf_counter()
        n = expected['total_chunks']
        for seq in range(start, n):
            lo, hi = max(head, seq*a.chunk), min(head+symbols.size*width, (seq+1)*a.chunk)
            delta = 0.0 if hi <= lo else reconstruction_mse(
                codec, erased_symbols(symbols, head, a.chunk, seq, a.erasure_fill, width), meta, reference) - result['intact_mse']
            # Negative scores are meaningful; do not silently clip them.
            result['scores'].append(delta)
            if (seq+1) % a.progress_every == 0 or seq+1 == n:
                atomic_json(partial, result)
                elapsed = time.perf_counter()-begun
                eta = elapsed/max(1, seq+1-start)*(n-seq-1)
                print('[{}] {}/{} elapsed={:.1f}s ETA={:.1f}s'.format(path.stem, seq+1, n, elapsed, eta), flush=True)
        result['complete'] = True
        atomic_json(dest, result)
        # Keep the recoverable partial checkpoint; transmitters only read .erasure.json.
        print('[saved] {} ; intact MSE={:.6f}'.format(dest, result['intact_mse']), flush=True)


if __name__ == '__main__':
    main()
