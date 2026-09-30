"""Fast, image-specific int8 DATA-chunk importance proxy.

One frozen-decoder backward pass scores all DATA chunks. Scores are gradient
proxies, NOT measured single-chunk erasure losses. Source NPYs are unchanged.
"""
import argparse
import hashlib
import json
import math
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from USRP.core_codec.latent_storage import validate as validate_int8
from USRP.erasure_training.measure_erasure_importance import atomic_json, file_hash, load_sample
from USRP.core_codec.swinjscc_codec import (get_model_kwargs,
                                            load_decoder_weights, symbols_to_feature)
from udp import send_raw as sender

SCHEME = 'int8_decoder_gradient_chunk_proxy_v1'


def score_chunks(raw, meta, feature_gradient, chunk):
    """Map restored-feature gradients onto exact interleaved int8 file bytes.

    The decoder sees [all real features, all imaginary features], whereas the
    on-disk NPY payload is [I0,Q0,I1,Q1,...]. The gradient is with respect to
    restored features, so multiply by sqrt(original_symbol_power) before
    combining it with dequantized symbol values.
    """
    n, head, scale = validate_int8(meta)
    if chunk <= 0 or chunk > 1400 or chunk % 8:
        raise ValueError('chunk must be a multiple of 8 in [8,1400]')
    body, _ = sender.split(raw)
    if len(body) != head + 2*n:
        raise ValueError('int8 payload length does not match metadata')
    grad = np.asarray(feature_gradient, dtype=np.float64).reshape(-1)
    if grad.size != 2*n or not np.isfinite(grad).all():
        raise ValueError('invalid restored-feature gradient')
    power = float(meta['original_symbol_power'])
    if not math.isfinite(power) or power < 0:
        raise ValueError('invalid original_symbol_power')
    q = np.frombuffer(body, dtype=np.int8, count=2*n, offset=head).reshape(n, 2)
    contribution = np.empty(2*n, dtype=np.float64)
    factor = scale * math.sqrt(power)
    contribution[0::2] = q[:, 0].astype(np.float64) * grad[:n] * factor
    contribution[1::2] = q[:, 1].astype(np.float64) * grad[n:] * factor
    absolute = np.abs(contribution)
    scores, signed = [], []
    for seq in range((len(raw)+chunk-1)//chunk):
        lo = max(head, seq*chunk)
        hi = min(head+2*n, (seq+1)*chunk)
        if hi <= lo:
            scores.append(0.0)
            signed.append(0.0)
        else:
            scores.append(float(absolute[lo-head:hi-head].sum()))
            signed.append(float(-contribution[lo-head:hi-head].sum()))
    if not np.isfinite(scores).all() or not np.isfinite(signed).all():
        raise ValueError('non-finite chunk importance')
    return scores, signed


class FrozenDecoder:
    def __init__(self, checkpoint, device, channels, snr, trainset, model_size, amp=False):
        import torch
        from net.decoder import create_decoder

        self.torch = torch
        self.device = device
        self.snr = snr
        self.model = 'SwinJSCC_w/o_SAandRA'
        self.channels = channels
        self.trainset = trainset
        self.model_size = model_size
        self.checkpoint = checkpoint
        self.amp = bool(amp)
        if self.amp and not str(device).startswith('cuda'):
            raise ValueError('--amp requires a CUDA device')
        self.decoder = None
        self.hw = None
        self.create_decoder = create_decoder

    def ensure_resolution(self, h, w):
        if self.decoder is None:
            from USRP.core_codec.swinjscc_config import get_downsample
            _, kwargs, _ = get_model_kwargs(self.trainset, self.model,
                                             self.model_size, self.channels,
                                             img_size=(h, w))
            self.decoder = self.create_decoder(**kwargs).to(self.device).eval()
            load_decoder_weights(self.decoder, self.checkpoint)
            for parameter in self.decoder.parameters():
                parameter.requires_grad_(False)
            self.downsample = get_downsample(self.trainset)
        if self.hw != (h, w):
            divisor = 2 ** self.downsample
            self.decoder.update_resolution(h//divisor, w//divisor)
            self.hw = (h, w)

    def gradient(self, symbols, meta, reference):
        torch = self.torch
        h, w = (int(v) for v in meta['input_shape'][2:4])
        self.ensure_resolution(h, w)
        feature = symbols_to_feature(symbols, meta['feature_shape'],
                                     orig_power=meta['original_symbol_power'],
                                     restore_power=True)
        target = np.transpose(reference.astype(np.float32)/255.0,
                              (2, 0, 1))[None, ...].copy()
        variable = torch.from_numpy(feature).to(self.device).requires_grad_(True)
        target_tensor = torch.from_numpy(target).to(self.device)
        autocast = torch.cuda.amp.autocast() if self.amp else nullcontext()
        with torch.enable_grad():
            with autocast:
                reconstruction = self.decoder(variable, self.snr, self.model)
                if reconstruction.shape != target_tensor.shape:
                    raise ValueError('decoder output/reference shape mismatch')
                loss = torch.mean((torch.clamp(reconstruction.float(), 0.0, 1.0)
                                   - target_tensor) ** 2)
            gradient, = torch.autograd.grad(loss, variable, retain_graph=False,
                                             create_graph=False)
        result = gradient.detach().cpu().numpy()
        mse_255 = float(loss.detach().cpu().item()) * 255.0**2
        if not math.isfinite(mse_255):
            raise ValueError('non-finite intact MSE')
        return result, mse_255


def expected_provenance(raw, meta, reference, chunk, checkpoint_hash,
                        decoder_hash, amp=False):
    return dict(scheme=SCHEME,
                source_sha256=hashlib.sha256(raw).hexdigest(),
                chunk_bytes=chunk,
                total_chunks=(len(raw)+chunk-1)//chunk,
                checkpoint_sha256=checkpoint_hash,
                decoder_checkpoint_sha256=decoder_hash,
                metadata_sha256=hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest(),
                reference_sha256=hashlib.sha256(reference.astype(np.uint8).tobytes()).hexdigest(),
                storage_format=meta['storage']['format'],
                model='SwinJSCC_w/o_SAandRA',
                C=meta['C'], snr=meta['snr_db'],
                score_rule='sum_abs_quantized_symbol_times_decoder_mse_gradient',
                loss='clipped_RGB_MSE_0_to_1',
                amp=bool(amp),
                original_power_restored=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--npy-dir', required=True)
    parser.add_argument('--ref-dir', required=True)
    parser.add_argument('--checkpoint', default='weights/SwinJSCC_wo_SAandRA_AWGN_HRimage_snr10_psnr_C96.model')
    parser.add_argument('--decoder-checkpoint', default='USRP/decoder_finetune_clean50_01/best.model')
    parser.add_argument('--C', type=int, default=96)
    parser.add_argument('--snr', type=int, default=10)
    parser.add_argument('--trainset', default='DIV2K')
    parser.add_argument('--model-size', default='base', choices=['small', 'base', 'large'])
    parser.add_argument('--chunk', type=int, default=1200)
    parser.add_argument('--device', default=None)
    parser.add_argument('--amp', action='store_true', help='CUDA fp16 autocast to reduce activation memory; validate separately from fp32')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    if args.chunk <= 0 or args.chunk > 1400 or args.chunk % 8:
        parser.error('--chunk must be a multiple of 8 in [8,1400]')
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')

    resolve = lambda value: (PROJECT_ROOT / value).resolve()
    npy_dir, ref_dir = resolve(args.npy_dir), resolve(args.ref_dir)
    files = sorted(npy_dir.glob('*.npy'))
    if args.limit:
        files = files[:args.limit]
    if not files:
        parser.error('No NPY files in --npy-dir')
    checkpoint, decoder_checkpoint = resolve(args.checkpoint), resolve(args.decoder_checkpoint)
    checkpoint_hash = file_hash(checkpoint)
    decoder_hash = file_hash(decoder_checkpoint)
    import torch
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    if args.amp and not str(device).startswith('cuda'):
        parser.error('--amp requires a CUDA device')
    print('[gradient] files={} chunk={} device={} (one frozen-decoder backward per new image)'.format(
        len(files), args.chunk, device), flush=True)
    decoder = None
    for path in files:
        raw, symbols, head, meta, reference = load_sample(path, ref_dir, allow_int8=True)
        if meta.get('storage', {}).get('format') != 'swinjscc_int8_v1':
            raise ValueError('gradient scorer requires int8: ' + str(path))
        if (meta.get('model') != 'SwinJSCC_w/o_SAandRA' or
                meta.get('C') != args.C or meta.get('snr_db') != args.snr or
                meta.get('trainset') != args.trainset or
                meta.get('model_size') != args.model_size):
            raise ValueError('NPY model/C/SNR/trainset/model_size mismatch: ' + str(path))
        if head != meta['storage']['header_bytes']:
            raise ValueError('int8 header mismatch: ' + str(path))
        expected = expected_provenance(raw, meta, reference, args.chunk,
                                       checkpoint_hash, decoder_hash, args.amp)
        destination = path.with_suffix('.gradient.json')
        if destination.exists() and not args.overwrite:
            existing = json.loads(destination.read_text(encoding='utf-8'))
            cached_scores = existing.get('scores')
            if (existing.get('complete') is not True or
                    existing.get('provenance') != expected or
                    not isinstance(cached_scores, list) or
                    len(cached_scores) != expected['total_chunks'] or
                    any(isinstance(value, bool) or not isinstance(value, (int, float))
                        or not math.isfinite(value) or value < 0
                        for value in cached_scores)):
                raise ValueError('Incompatible gradient sidecar; inspect before --overwrite: ' + str(destination))
            print('[skip] verified ' + destination.name, flush=True)
            continue
        if decoder is None:
            decoder = FrozenDecoder(decoder_checkpoint, device, args.C, args.snr,
                                    args.trainset, args.model_size, args.amp)
        started = time.perf_counter()
        gradient, mse = decoder.gradient(symbols, meta, reference)
        scores, signed = score_chunks(raw, meta, gradient, args.chunk)
        result = dict(complete=True, provenance=expected, intact_mse=mse,
                      scores=scores, signed_first_order=signed)
        atomic_json(destination, result)
        print('[saved] {} chunks={} {:.1f}s intact_MSE={:.3f}'.format(
            destination, len(scores), time.perf_counter()-started, mse), flush=True)


if __name__ == '__main__':
    main()
