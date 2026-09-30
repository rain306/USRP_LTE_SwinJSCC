"""Offline float32/float16/int8 latent round-trip probe; no training or sockets.

The measured packed container is experimental, NOT accepted by send_raw/receive_raw.
Quantization is on normalized I/Q, before restoring original_symbol_power.
"""
import argparse
import hashlib
import json
import math
import struct
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from USRP.erasure_training.measure_erasure_importance import atomic_json, file_hash, load_sample
from USRP.core_codec.swinjscc_codec import SwinJSCCCodec, compute_psnr_mse

MAGIC = b'QPROBE01'
METHODS = ('float32', 'float16', 'int8')


def pack(symbols, meta, method):
    if method not in METHODS:
        raise ValueError('Unknown representation')
    symbols = np.asarray(symbols)
    if symbols.dtype != np.dtype('complex64') or symbols.ndim != 1 or not symbols.size:
        raise ValueError('Expected nonempty one-dimensional complex64')
    values = np.empty((symbols.size, 2), dtype=np.float32)
    values[:, 0], values[:, 1] = symbols.real, symbols.imag
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite source latent')
    header = dict(version=1, method=method, complex_count=int(symbols.size),
                  layout='interleaved_IQ', metadata=meta)
    if method == 'float32':
        encoded = values.astype('<f4').tobytes()
    elif method == 'float16':
        if np.max(np.abs(values)) > np.finfo(np.float16).max:
            raise ValueError('float16 overflow; source not modified')
        encoded = values.astype('<f2').tobytes()
    else:
        # One per-image symmetric scale, no percentile clipping or calibration.
        maximum = float(np.max(np.abs(values.astype(np.float64))))
        scale = maximum / 127.0 if maximum else 1.0
        header['scale'] = scale
        header['zero_point'] = 0
        encoded = np.clip(np.rint(values.astype(np.float64) / scale),
                          -127, 127).astype('i1').tobytes()
    metadata = json.dumps(header, ensure_ascii=False, separators=(',', ':'),
                          allow_nan=False).encode('utf-8')
    blob = MAGIC + struct.pack('<I', len(metadata)) + metadata + encoded
    return blob, len(encoded)


def unpack(blob):
    if len(blob) < 12 or blob[:8] != MAGIC:
        raise ValueError('Invalid probe container')
    length = struct.unpack('<I', blob[8:12])[0]
    if length > len(blob) - 12:
        raise ValueError('Truncated probe header')
    header = json.loads(blob[12:12+length].decode('utf-8'))
    method, count = header['method'], header['complex_count']
    if (header['version'] != 1 or header['layout'] != 'interleaved_IQ'
            or method not in METHODS or type(count) is not int or count <= 0):
        raise ValueError('Unsupported probe header')
    dtype = {'float32': '<f4', 'float16': '<f2', 'int8': 'i1'}[method]
    data = blob[12+length:]
    if len(data) != count * 2 * np.dtype(dtype).itemsize:
        raise ValueError('Payload size mismatch')
    values = np.frombuffer(data, dtype=dtype).astype(np.float64)
    if method == 'int8':
        scale = float(header['scale'])
        if not math.isfinite(scale) or scale <= 0 or header['zero_point'] != 0:
            raise ValueError('Invalid int8 scale')
        values *= scale
    values = values.reshape(count, 2).astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite recovered latent')
    symbols = np.empty(count, dtype=np.complex64)
    symbols.real, symbols.imag = values[:, 0], values[:, 1]
    return symbols, header['metadata']


def self_test():
    rng = np.random.RandomState(2026)
    cases = [np.zeros(7, np.complex64),
             np.array([1+2j, -3-4j, 0j], np.complex64),
             (rng.randn(257) + 1j*rng.randn(257)).astype(np.complex64)]
    for source in cases:
        for method in METHODS:
            blob, size = pack(source, {'input_shape': [1, 3, 512, 768]}, method)
            restored, meta = unpack(blob)
            assert meta['input_shape'] == [1, 3, 512, 768]
            assert size == source.size * {'float32': 8, 'float16': 4, 'int8': 2}[method]
            if method == 'float32':
                assert source.tobytes() == restored.tobytes()
            elif method == 'float16':
                np.testing.assert_array_equal(restored.real, source.real.astype('f2').astype('f4'))
                np.testing.assert_array_equal(restored.imag, source.imag.astype('f2').astype('f4'))
            else:
                maximum = max(float(np.max(np.abs(source.real))), float(np.max(np.abs(source.imag))))
                bound = maximum / 254.0 + 1e-6
                assert np.max(np.abs(restored.real-source.real)) <= bound
                assert np.max(np.abs(restored.imag-source.imag)) <= bound
            try:
                unpack(blob[:-1])
            except ValueError:
                pass
            else:
                raise AssertionError('Truncated payload accepted')
    for source, method in [(np.array([np.nan+0j], np.complex64), 'int8'),
                           (np.array([70000+0j], np.complex64), 'float16')]:
        try:
            pack(source, {}, method)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid latent accepted')
    print('[self-test] PASS: layouts, exact float32, float16 rounding, int8 bounds, invalid inputs')


def aggregate(rows):
    output = []
    for method in METHODS:
        group = [r for r in rows if r['method'] == method]
        if not group:
            continue
        item = dict(method=method, count=len(group))
        for key in ('payload_bytes', 'packed_bytes', 'payload_ratio', 'packed_ratio',
                    'roundtrip_seconds', 'decode_seconds', 'psnr_db', 'ssim',
                    'delta_psnr_db', 'delta_ssim'):
            values = [r[key] for r in group if r.get(key) is not None]
            item['mean_' + key] = float(np.mean(values)) if values else None
        for key in ('delta_psnr_db', 'delta_ssim'):
            valid = [r for r in group if r.get(key) is not None]
            if valid:
                worst = min(valid, key=lambda r: r[key])
                item['min_' + key] = worst[key]
                item['worst_image_' + key] = worst['image']
        item['total_packed_bytes'] = sum(r['packed_bytes'] for r in group)
        output.append(item)
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--npy-dir', default='USRP/npy_in')
    p.add_argument('--ref-dir', default='datasets/Kodak24')
    p.add_argument('--out-dir')
    p.add_argument('--checkpoint', default='weights/SwinJSCC_wo_SAandRA_AWGN_HRimage_snr10_psnr_C96.model')
    p.add_argument('--decoder-checkpoint', default='USRP/decoder_finetune_clean50_01/best.model')
    p.add_argument('--device', default=None)
    p.add_argument('--limit', type=int)
    p.add_argument('--pack-only', action='store_true', help='Check real files/byte counts without model decoding')
    p.add_argument('--self-test', action='store_true')
    a = p.parse_args()
    if a.self_test:
        self_test()
        return
    if not a.out_dir or (a.limit is not None and a.limit < 1):
        p.error('--out-dir required; --limit must be positive')
    root = lambda value: (PROJECT_ROOT / value).resolve()
    out = root(a.out_dir)
    if out.exists():
        p.error('Output exists; choose a NEW directory to preserve earlier results')
    files = sorted(root(a.npy_dir).glob('*.npy'))
    if a.limit:
        files = files[:a.limit]
    if not files:
        p.error('No NPY inputs')
    # Preflight all inputs before any expensive decoding; retain only paths/meta.
    specs = []
    for path in files:
        raw, symbols, _, meta, reference = load_sample(path, root(a.ref_dir))
        if (meta.get('model') != 'SwinJSCC_w/o_SAandRA' or meta.get('C') != 96
                or meta.get('snr_db') != 10):
            raise ValueError('Probe is fixed to current C96/SNR10 model: ' + path.name)
        if int(np.prod(meta['feature_shape'])) != 2*symbols.size:
            raise ValueError('Feature shape mismatch')
        if not math.isfinite(float(meta['original_symbol_power'])) or meta['original_symbol_power'] <= 0:
            raise ValueError('Invalid original power')
        # Check both lossy methods now, so float16 overflow cannot silently skip a file.
        for method in METHODS:
            restored, recovered_meta = unpack(pack(symbols, meta, method)[0])
            if recovered_meta != meta or restored.shape != symbols.shape:
                raise ValueError('Metadata or shape round-trip mismatch')
        specs.append((tuple(meta['input_shape'][2:4]), path,
                      hashlib.sha256(raw).hexdigest(),
                      hashlib.sha256(reference.astype('<f8').tobytes()).hexdigest()))
    codec = None
    if not a.pack_only:
        from USRP.core_codec.decode_dataset import _ssim
        # Fail early if the installed SSIM implementation is unavailable.
        probe = np.zeros((16, 16, 3), np.float64)
        if not math.isfinite(_ssim(probe, probe)):
            raise ValueError('SSIM unavailable')
        codec = SwinJSCCCodec(str(root(a.checkpoint)), decoder_checkpoint=str(root(a.decoder_checkpoint)),
                             C=96, snr=10, device=a.device)
    settings = vars(a).copy()
    settings['source_sha256'] = {path.name: digest for _, path, digest, _ in specs}
    settings['reference_crop_sha256'] = {path.name: digest for _, path, _, digest in specs}
    if codec:
        settings['checkpoint_sha256'] = file_hash(root(a.checkpoint))
        settings['decoder_checkpoint_sha256'] = file_hash(root(a.decoder_checkpoint))
    out.mkdir(parents=True, exist_ok=False)
    rows = []

    def save(complete, error=None):
        atomic_json(out / 'metrics.json', dict(
            complete=complete, quality_evaluated=bool(codec), error=error,
            expected_records=len(files)*len(METHODS), settings=settings,
            assumptions=['No packet loss, UEP or radio transmission in this probe.',
                         'Packed bytes include a real serialized experimental header plus latent payload.',
                         'Probe container is NOT compatible with the online transport scripts.',
                         'int8 uses one per-image symmetric scale, zero point 0, no clipping calibration.',
                         'Metrics use float reconstructions, before PNG rounding; original power is restored.',
                         'Timing includes warmup/resolution changes; not a performance benchmark.'],
            summary=aggregate(rows), per_image=rows))

    save(False)
    print('[probe] {} images, {} decodes; output={}'.format(len(files), 0 if not codec else len(files)*3, out), flush=True)
    try:
        for _, path, expected_hash, _ in sorted(specs):
            raw, source, _, meta, reference = load_sample(path, root(a.ref_dir))
            if hashlib.sha256(raw).hexdigest() != expected_hash:
                raise ValueError('Input changed during run')
            baseline = None
            for method in METHODS:
                started = time.perf_counter()
                blob, payload_size = pack(source, meta, method)
                recovered, recovered_meta = unpack(blob)
                row = dict(image=path.name, method=method, payload_bytes=payload_size,
                           packed_bytes=len(blob), header_bytes=len(blob)-payload_size,
                           source_npy_bytes=len(raw), payload_ratio=payload_size/source.nbytes,
                           roundtrip_seconds=time.perf_counter()-started)
                difference = recovered.astype(np.complex128)-source.astype(np.complex128)
                row['latent_mse_per_real_component'] = float(np.mean(np.abs(difference)**2)/2)
                if method == 'float32' and recovered.tobytes() != source.tobytes():
                    raise ValueError('Float32 baseline is not byte-exact')
                if codec:
                    started = time.perf_counter()
                    recon = codec.decode(recovered, recovered_meta['feature_shape'],
                                         recovered_meta['input_shape'][2:4],
                                         orig_power=recovered_meta['original_symbol_power'], restore_power=True)
                    row['decode_seconds'] = time.perf_counter()-started
                    if not np.isfinite(recon).all():
                        raise ValueError('Nonfinite reconstruction')
                    psnr, mse = compute_psnr_mse(reference, recon)
                    ssim = float(_ssim(reference, recon))
                    if not math.isfinite(ssim) or not math.isfinite(mse):
                        raise ValueError('Invalid image metrics')
                    row.update(psnr_db=float(psnr) if math.isfinite(psnr) else None,
                               perfect_reconstruction=(mse == 0), mse=float(mse), ssim=ssim)
                if baseline is None:
                    baseline = row.copy()
                row['packed_ratio'] = row['packed_bytes']/baseline['packed_bytes']
                if codec:
                    row['delta_psnr_db'] = (row['psnr_db']-baseline['psnr_db']
                                            if row['psnr_db'] is not None and baseline['psnr_db'] is not None else None)
                    row['delta_ssim'] = row['ssim']-baseline['ssim']
                rows.append(row)
                save(False)
                print('[{}] {} payload={} packed={} PSNR={} SSIM={} delta_PSNR={}'.format(
                    path.stem, method, payload_size, len(blob), row.get('psnr_db'),
                    row.get('ssim'), row.get('delta_psnr_db')), flush=True)
        save(True)
    except BaseException as exc:
        save(False, '{}: {}'.format(type(exc).__name__, exc))
        raise
    for item in aggregate(rows):
        print('[summary] ' + json.dumps(item, ensure_ascii=False), flush=True)
    print('[saved] ' + str(out / 'metrics.json'), flush=True)


if __name__ == '__main__':
    main()
