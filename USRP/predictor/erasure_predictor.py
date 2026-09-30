"""Cheap int8 chunk features and a CPU ridge ranker for erasure labels.

The target is a *within-image rank* of measured single-chunk delta MSE.
Neither these features nor the ranker's output are measured erasure losses.
"""
import hashlib
import io
import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from USRP.core_codec.latent_storage import validate as validate_int8
from udp import send_raw as sender


SCHEME = 'int8_erasure_rank_ridge_v1'
FEATURE_VERSION = 'int8_chunk_statistics_v1'
FEATURE_NAMES = [
    'log_mean_square', 'log_mean_abs', 'log_std_abs', 'log_p90_abs',
    'log_max_abs', 'zero_fraction', 'high_fraction', 'mean_abs_i',
    'mean_abs_q', 'mean_i', 'mean_q', 'energy_over_image',
    'log_neighbor_energy', 'energy_percentile', 'relative_position',
    'position_sin', 'position_cos', 'log_image_area', 'aspect_ratio',
    'log_quant_scale', 'log_original_power',
]


def load_source(path):
    """Require an original, embedded-META int8 NPY, never a received NPY."""
    path = Path(path)
    raw = path.read_bytes()
    body, meta = sender.split(raw)
    if not isinstance(meta, dict):
        raise ValueError('Embedded META is required: ' + str(path))
    n, head, _ = validate_int8(meta)
    array = np.load(io.BytesIO(body), allow_pickle=False)
    if array.dtype != np.int8 or array.shape != (n, 2) or len(body) != head + 2*n:
        raise ValueError('NPY/header does not match embedded int8 META: ' + str(path))
    if not math.isfinite(float(meta['original_symbol_power'])):
        raise ValueError('Invalid original symbol power: ' + str(path))
    return raw, meta


def chunk_features(raw, meta, chunk):
    """Return features and eligibility aligned to exact on-wire DATA indices."""
    if chunk <= 0 or chunk > 1400 or chunk % 8:
        raise ValueError('chunk must be a multiple of 8 in [8,1400]')
    body, _ = sender.split(raw)
    n, head, scale = validate_int8(meta)
    if len(body) != head + 2*n:
        raise ValueError('int8 payload/header mismatch')
    h, w = (int(v) for v in meta['input_shape'][2:4])
    if h <= 0 or w <= 0:
        raise ValueError('Invalid source dimensions')
    payload = np.frombuffer(body, dtype=np.int8, count=2*n, offset=head)
    count = (len(raw) + chunk - 1)//chunk
    features = np.zeros((count, len(FEATURE_NAMES)), dtype=np.float64)
    eligible = np.zeros(count, dtype=bool)
    energy = np.zeros(count, dtype=np.float64)
    values = [None] * count
    for seq in range(count):
        lo = max(head, seq*chunk)
        hi = min(head + 2*n, (seq+1)*chunk)
        if seq == 0 or hi <= lo:
            continue  # Header is protected independently; trailer is not latent.
        q = payload[lo-head:hi-head].astype(np.float64)
        if len(q) % 2:
            raise ValueError('Unaligned int8 I/Q DATA chunk')
        eligible[seq] = True
        values[seq] = q
        energy[seq] = float(np.mean(q*q))
    usable = energy[eligible]
    if not len(usable):
        raise ValueError('No eligible latent DATA chunks')
    global_energy = float(np.mean(usable))
    order = np.argsort(usable, kind='mergesort')
    energy_pct = np.empty(len(usable), dtype=np.float64)
    energy_pct[order] = np.arange(len(usable), dtype=np.float64)/max(1, len(usable)-1)
    pct_by_seq = np.zeros(count, dtype=np.float64)
    pct_by_seq[eligible] = energy_pct
    for seq in np.flatnonzero(eligible):
        q = values[seq]
        absolute = np.abs(q)
        pos = (seq + 0.5)/count
        neighbors = energy[max(0, seq-1):min(count, seq+2)]
        neighbor_energy = float(np.mean(neighbors))
        features[seq] = [
            math.log1p(energy[seq]), math.log1p(float(absolute.mean())),
            math.log1p(float(absolute.std())),
            math.log1p(float(np.percentile(absolute, 90))),
            math.log1p(float(absolute.max())),
            float(np.mean(q == 0)), float(np.mean(absolute >= 120)),
            float(np.mean(np.abs(q[0::2])))/127,
            float(np.mean(np.abs(q[1::2])))/127,
            float(np.mean(q[0::2]))/127, float(np.mean(q[1::2]))/127,
            energy[seq]/max(1e-12, global_energy),
            math.log1p(neighbor_energy), pct_by_seq[seq], pos,
            math.sin(2*math.pi*pos), math.cos(2*math.pi*pos),
            math.log1p(h*w), h/w, math.log(scale),
            math.log1p(float(meta['original_symbol_power'])),
        ]
    if not np.isfinite(features).all():
        raise ValueError('Non-finite predictor features')
    return features, eligible


def read_labels(path, raw, meta, chunk, eligible):
    sidecar = Path(path).with_suffix('.erasure.json')
    data = json.loads(sidecar.read_text(encoding='utf-8'))
    spec = data['provenance']
    count = len(eligible)
    if (data.get('complete') is not True or
            spec.get('scheme') != 'single_chunk_erasure_delta_mse_v1' or
            spec.get('source_sha256') != hashlib.sha256(raw).hexdigest() or
            spec.get('chunk_bytes') != chunk or
            spec.get('total_chunks') != count or
            spec.get('storage_format') != 'swinjscc_int8_v1' or
            spec.get('bytes_per_symbol') != 2 or
            spec.get('model') != meta.get('model') or
            spec.get('C') != meta.get('C') or
            spec.get('snr') != meta.get('snr_db')):
        raise ValueError('Incomplete or mismatched erasure labels: ' + str(sidecar))
    scores = np.asarray(data['scores'], dtype=np.float64)
    if scores.shape != (count,) or not np.isfinite(scores).all():
        raise ValueError('Invalid measured scores: ' + str(sidecar))
    return scores, spec


def percentile_targets(scores):
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(scores, kind='mergesort')
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start+1
        while end < len(order) and scores[order[end]] == scores[order[start]]:
            end += 1
        ranks[order[start:end]] = (start+end-1)/(2*max(1, len(order)-1))
        start = end
    return ranks


def predict(model, features, eligible):
    if model.get('scheme') != SCHEME or model.get('feature_version') != FEATURE_VERSION:
        raise ValueError('Unsupported predictor model')
    if model.get('feature_names') != FEATURE_NAMES:
        raise ValueError('Predictor feature schema mismatch')
    mean = np.asarray(model['feature_mean'], dtype=np.float64)
    std = np.asarray(model['feature_std'], dtype=np.float64)
    coef = np.asarray(model['coef'], dtype=np.float64)
    if any(a.shape != (len(FEATURE_NAMES),) for a in (mean, std, coef)) or np.any(std <= 0):
        raise ValueError('Invalid predictor coefficients')
    result = np.full(len(eligible), -1e30, dtype=np.float64)
    result[eligible] = ((features[eligible] - mean)/std) @ coef + float(model['intercept'])
    if not np.isfinite(result).all():
        raise ValueError('Non-finite predictor output')
    return result


def rank_report(predicted, measured, eligible, top_fraction):
    ids = np.flatnonzero(eligible)
    k = max(1, round(len(measured)*top_fraction)-1)  # Chunk zero consumes one repeat slot.
    k = min(k, len(ids))
    actual = set(sorted(ids, key=lambda i: (-measured[i], i))[:k])
    selected = set(sorted(ids, key=lambda i: (-predicted[i], i))[:k])
    x = percentile_targets(np.asarray(predicted)[ids])
    y = percentile_targets(np.asarray(measured)[ids])
    rho = (float(np.corrcoef(x, y)[0, 1])
           if len(ids) > 1 and np.std(x) > 0 and np.std(y) > 0 else None)
    return dict(top_k=k, top_recall=len(actual & selected)/k,
                random_recall=k/len(ids), spearman=rho,
                selected_delta_sum=float(np.sum(measured[list(selected)])))
