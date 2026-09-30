"""Versioned symmetric int8 I/Q in ordinary NPY + existing META trailer.

The transport remains byte-oriented. int8 zero bytes represent feature zero.
Only this module interprets quantization; UDP hosts do not need this module.
"""
import io
import math
import numpy as np

FORMAT = 'swinjscc_int8_v1'


def is_int8(meta):
    storage = (meta or {}).get('storage')
    if storage is None:
        return False
    if not isinstance(storage, dict) or storage.get('format') != FORMAT:
        raise ValueError('Unsupported latent storage format')
    return True


def quantize(symbols, meta):
    symbols = np.asarray(symbols)
    if symbols.dtype != np.dtype('complex64') or symbols.ndim != 1 or not symbols.size:
        raise ValueError('Expected original complex64 symbols')
    values = np.stack([symbols.real, symbols.imag], axis=1).astype(np.float64)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite source symbols')
    peak = float(np.max(np.abs(values)))
    scale = peak / 127.0 if peak else 1.0
    array = np.clip(np.rint(values / scale), -127, 127).astype(np.int8)
    buf = io.BytesIO()
    np.save(buf, array, allow_pickle=False)
    header_bytes = len(buf.getvalue()) - array.nbytes
    # Keep decoding metadata compact enough for the existing 1400-byte META.
    keys = ('image_stem', 'image_name', 'model', 'trainset', 'model_size', 'C',
            'rate', 'snr_db', 'input_shape', 'feature_shape', 'original_symbol_power',
            'target_power', 'cbr')
    result = {k: meta[k] for k in keys if k in meta}
    result.update(num_complex_symbols=int(symbols.size), dtype='int8',
                  storage=dict(format=FORMAT, scale=scale, zero_point=0,
                               shape=[int(symbols.size), 2], header_bytes=header_bytes))
    validate(result)
    return array, result


def validate(meta):
    if not is_int8(meta):
        raise ValueError('Missing int8 metadata')
    s = meta['storage']
    n = meta['num_complex_symbols']
    head = s['header_bytes']
    scale = float(s['scale'])
    if (type(n) is not int or n <= 0 or s['shape'] != [n, 2]
            or type(head) is not int or head < 10 or head % 2
            or not math.isfinite(scale) or scale <= 0 or s['zero_point'] != 0
            or int(np.prod(meta['feature_shape'])) != 2*n or meta.get('dtype') != 'int8'):
        raise ValueError('Invalid int8 dimensions/scale/header metadata')
    return n, head, scale


def load_int8(raw, meta):
    n, head, scale = validate(meta)
    if len(raw) < head + 2*n:
        raise ValueError('Truncated int8 NPY payload')
    # Offset comes from protected META, never len(file)-payload: trailers can be lost.
    values = np.frombuffer(raw, dtype=np.int8, count=2*n, offset=head).reshape(n, 2)
    restored = (values.astype(np.float64) * scale).astype(np.float32)
    if not np.isfinite(restored).all():
        raise ValueError('Nonfinite dequantized symbols')
    symbols = np.empty(n, dtype=np.complex64)
    symbols.real, symbols.imag = restored[:, 0], restored[:, 1]
    return symbols
