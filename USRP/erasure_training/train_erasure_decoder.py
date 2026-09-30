"""Frozen-encoder block-erasure decoder adaptation; never trains on Kodak.

Synthetic losses here are residual missing chunks in serialized file order,
not LTE packet-time losses or a simulation of retransmissions. Output stays in
USRP. Validation uses fixed crops/masks and selects by mean reconstruction MSE.
"""
import argparse
import io
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from USRP.core_codec.swinjscc_codec import SwinJSCCCodec, chunk_symbol_slice
from USRP.erasure_training.measure_erasure_importance import atomic_json, file_hash
from USRP.offline_comparison.compare_importance import erasures, seed_for
from USRP.core_codec.decode_dataset import _ssim


def images(root):
    return sorted(p for p in root.rglob('*') if p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp'))


def crop_tensor(path, size, rng, device):
    with Image.open(path) as im:
        im = im.convert('RGB')
        if min(im.size) < size:
            raise ValueError('Image smaller than crop: ' + str(path))
        if rng is None:
            left, top = (im.width-size)//2, (im.height-size)//2
        else:
            left, top = rng.randrange(im.width-size+1), rng.randrange(im.height-size+1)
        arr = np.array(im.crop((left, top, left+size, top+size)), dtype=np.float32)/255.
        if rng is not None and rng.random() < .5:
            arr = arr[:, ::-1].copy()
    return torch.from_numpy(arr.transpose(2, 0, 1).copy()).unsqueeze(0).to(device)


def feature_mask(shape, chunk, rate, kind, burst_length, seed):
    """NPY offsets -> complex symbol mask -> original real/imag halves."""
    count = int(np.prod(shape))
    if count % 2 or chunk % 8:
        raise ValueError('Expected even feature length and aligned chunks')
    half = count//2
    stream = io.BytesIO()
    np.save(stream, np.zeros(half, dtype=np.complex64), allow_pickle=False)
    head = stream.tell()-half*8
    n = (stream.tell()+chunk-1)//chunk
    lost = erasures(n, rate, kind, burst_length, seed)
    keep = np.ones(half, dtype=np.float32)
    for seq, drop in enumerate(lost):
        if drop:
            bounds = chunk_symbol_slice(head, chunk, seq, half*8)
            if bounds is not None:
                keep[bounds[0]:bounds[1]] = 0
    return np.concatenate((keep, keep)).reshape(shape)


def erase_feature(feature, mask, fill=0.0):
    """Fill is per I/Q component in unit-power serialized complex64 domain.

    The decoder restores sqrt(original complex power). A normalized -1 sentinel
    therefore becomes -sqrt(2*mean(feature**2)), not -1 in raw feature space.
    This does NOT make -1 out-of-range for unconstrained SwinJSCC features.
    """
    keep = torch.as_tensor(mask, dtype=feature.dtype, device=feature.device)
    erased = feature*keep
    if fill != 0.0:
        scale = torch.sqrt(torch.mean(feature.detach()**2)*2)
        erased = erased+(1-keep)*float(fill)*scale
    return erased


def validation(codec, paths, a):
    codec.decoder.eval()
    rows = []
    with torch.no_grad():
        for path in paths:
            x = crop_tensor(path, a.crop, None, codec.device)
            feature = codec.encoder(x, a.snr, a.C, codec.model).float()
            # On the existing float32 file path, normalization and restored power
            # cancel. Zeroing the original feature is the same erased input.
            for kind, rate in [('clean', 0.)] + [(k, r) for k in ('independent', 'burst') for r in a.rates]:
                mask = feature_mask(tuple(feature.shape), a.chunk, rate,
                                    'independent' if kind == 'clean' else kind,
                                    a.burst_length, seed_for(a.seed if a.validation_seed is None else a.validation_seed, path.name, kind, rate, 'validation'))
                recon = codec.decoder(erase_feature(feature, mask, a.erasure_fill), a.snr, codec.model).clamp(0, 1)
                mse = float(torch.mean((recon-x)**2).item())
                if not math.isfinite(mse):
                    raise ValueError('Nonfinite validation MSE')
                target = x[0].permute(1, 2, 0).cpu().numpy()*255
                pred = recon[0].permute(1, 2, 0).cpu().numpy()*255
                rows.append(dict(image=path.name, loss_model=kind, residual_loss_rate=rate,
                                 mse=mse, psnr_db=-10*math.log10(mse) if mse else None,
                                 ssim=float(_ssim(target, pred))))
            print('[validation] '+path.name+' complete', flush=True)
    codec.decoder.train()
    summary = []
    for kind, rate in dict.fromkeys((r['loss_model'], r['residual_loss_rate']) for r in rows):
        subset = [r for r in rows if r['loss_model'] == kind and r['residual_loss_rate'] == rate]
        psnrs = [r['psnr_db'] for r in subset if r['psnr_db'] is not None]
        summary.append(dict(loss_model=kind, residual_loss_rate=rate, count=len(subset),
                            mean_psnr_db=float(np.mean(psnrs)) if psnrs else None,
                            mean_ssim=float(np.mean([r['ssim'] for r in subset]))))
    return dict(selection_mse=float(np.mean([r['mse'] for r in rows])), summary=summary, per_image=rows)


def save_torch(path, data):
    temp = path.with_suffix(path.suffix+'.tmp')
    torch.save(data, str(temp))
    temp.replace(path)


def training_condition(rng, args, model_seed=None):
    # Consume the legacy choice even for an override, preserving paired crops,
    # rate choices and mask seeds across mixed/independent training runs.
    kind = rng.choice(('independent', 'burst'))
    if args.train_loss_model != 'mixed':
        kind = args.train_loss_model
    elif args.train_burst_prob is not None:
        if model_seed is None:
            raise ValueError('Weighted mixture requires deterministic per-step seed')
        probability = args.train_burst_prob
        draw = random.Random(model_seed).random()
        # Keep the original 50/50 decisions at p=.5. For smaller p, retain a
        # subset of those burst examples without shifting the main RNG stream.
        if probability < .5 and kind == 'burst' and draw >= 2*probability:
            kind = 'independent'
        elif probability > .5 and kind == 'independent' and draw < 2*probability-1:
            kind = 'burst'
    return kind, rng.choice(args.rates)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', default='datasets/DIV2K_valid')
    p.add_argument('--test-dir', default='datasets/Kodak24', help='Excluded by content hash; never used for model selection')
    p.add_argument('--out-dir', required=True)
    p.add_argument('--checkpoint', default='weights/SwinJSCC_wo_SAandRA_AWGN_HRimage_snr10_psnr_C96.model')
    p.add_argument('--C', type=int, default=96)
    p.add_argument('--snr', type=int, default=10)
    p.add_argument('--device', default=None)
    p.add_argument('--crop', type=int, default=256)
    p.add_argument('--chunk', type=int, default=1200)
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--clean-weight', type=float, default=.25)
    p.add_argument('--erasure-fill', type=float, default=0., help='Missing I/Q value BEFORE power restoration; default zero, experiment: -1')
    p.add_argument('--loss-rates', default='0.05,0.1,0.2,0.3', help='Residual erased fraction, NOT transmitted packet loss')
    p.add_argument('--burst-length', type=float, default=5.)
    p.add_argument('--train-loss-model', choices=['mixed', 'independent', 'burst'], default='mixed',
                   help='Training only; mixed retains legacy 50/50. Validation still tests both models.')
    p.add_argument('--train-burst-prob', type=float, default=None,
                   help='Mixed training burst probability [0,1]; omitted preserves legacy 50/50. Validation unchanged.')
    p.add_argument('--val-count', type=int, default=10)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--split-seed', type=int, default=None, help='Fixed data split seed; defaults to --seed for legacy runs')
    p.add_argument('--validation-seed', type=int, default=None, help='Fixed validation erasures; defaults to --seed')
    p.add_argument('--limit', type=int, help='Limit source images for a smoke test')
    p.add_argument('--steps-per-epoch', type=int, help='Optional training-step cap; defaults to all training images')
    p.add_argument('--resume', action='store_true', help='Resume last completed epoch in the same output directory')
    a = p.parse_args()
    if a.train_burst_prob is not None and (not math.isfinite(a.train_burst_prob) or not 0 <= a.train_burst_prob <= 1 or a.train_loss_model != 'mixed'):
        p.error('train-burst-prob must be finite in [0,1] and requires --train-loss-model mixed')
    if not math.isfinite(a.erasure_fill) or abs(a.erasure_fill) > np.finfo(np.float32).max:
        p.error('erasure-fill must fit finite float32')
    a.rates = list(dict.fromkeys(float(v) for v in a.loss_rates.split(',')))
    if not a.rates or any(not 0 < r < 1 for r in a.rates):
        p.error('loss-rates must be finite and in (0,1)')
    if a.crop < 128 or a.crop % 128 or a.chunk < 8 or a.chunk > 1400 or a.chunk % 8:
        p.error('crop must be a positive multiple of 128; chunk a multiple of 8 in [8,1400]')
    if a.epochs < 1 or a.val_count < 1 or not 0 <= a.clean_weight <= 1 or not math.isfinite(a.lr) or a.lr <= 0:
        p.error('Invalid epochs, val-count, clean-weight or learning rate')
    if not math.isfinite(a.burst_length) or a.burst_length < 1 or (a.steps_per_epoch is not None and a.steps_per_epoch < 1):
        p.error('Invalid burst-length or steps-per-epoch')
    if a.limit is not None and a.limit <= a.val_count:
        p.error('limit must exceed val-count')
    for rate in a.rates:
        erasures(1, rate, 'burst', a.burst_length, a.seed)
    root = lambda x: (PROJECT_ROOT/x).resolve()
    out = root(a.out_dir)
    if (PROJECT_ROOT/'USRP').resolve() not in out.parents:
        p.error('Training output must be inside USRP, not official source/weights folders')
    paths = images(root(a.data_dir))
    if a.limit:
        paths = paths[:a.limit]
    if len(paths) <= a.val_count:
        p.error('Not enough training images for the validation split')
    excluded = images(root(a.test_dir))
    if not excluded:
        p.error('Test directory missing; cannot check held-out leakage')
    test_hashes = {file_hash(f) for f in excluded}
    hashes = {str(f): file_hash(f) for f in paths}
    if any(v in test_hashes for v in hashes.values()) or len(set(hashes.values())) != len(paths):
        p.error('Duplicate images or overlap with held-out test set')
    for path in paths:
        with Image.open(path) as im:
            if min(im.size) < a.crop:
                p.error('Image smaller than crop: '+str(path))
    random.Random(a.seed if a.split_seed is None else a.split_seed).shuffle(paths)
    val_paths, train_paths = paths[:a.val_count], paths[a.val_count:]
    config = vars(a).copy()
    if a.train_burst_prob is None:
        config.pop('train_burst_prob')
    if a.train_loss_model == 'mixed':
        config.pop('train_loss_model')  # Preserve existing clean50 resume configs.
    for key in ('split_seed', 'validation_seed'):
        if config[key] is None:
            config.pop(key)
    # Retain compatibility with pre-sentinel zero-fill resume checkpoints.
    if a.erasure_fill == 0.0:
        config.pop('erasure_fill')
    for k in ('resume', 'epochs', 'out_dir'):
        config.pop(k)
    config['checkpoint_sha256'] = file_hash(root(a.checkpoint))
    config['source_sha256'] = hashes
    config['validation_images'] = list(map(str, val_paths))
    config['training_images'] = list(map(str, train_paths))
    if a.resume:
        if not (out/'last.pt').is_file():
            p.error('No last.pt to resume')
    else:
        out.mkdir(parents=True, exist_ok=False)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)
    rng = random.Random(a.seed)
    codec = SwinJSCCCodec(a.checkpoint, C=a.C, snr=a.snr, device=a.device)
    codec._ensure_models(a.crop, a.crop)
    codec.encoder.eval()
    for param in codec.encoder.parameters():
        param.requires_grad_(False)
    encoder_before = {k: v.detach().cpu().clone() for k, v in codec.encoder.named_parameters()}
    optimizer = torch.optim.Adam(codec.decoder.parameters(), lr=a.lr)
    start = 0
    history = []

    def export_best():
        # A compatible encoder+decoder state dict; original checkpoint is read-only.
        state = {}
        for prefix, module in [('encoder.', codec.encoder), ('decoder.', codec.decoder)]:
            state.update({prefix+k: v.detach().cpu() for k, v in module.state_dict().items()
                          if not k.endswith('attn_mask')})
        save_torch(out/'best.model', state)

    if a.resume:
        saved = torch.load(str(out/'last.pt'), map_location=codec.device)
        if saved['config'] != config:
            raise ValueError('Resume settings or input hashes changed')
        codec.decoder.load_state_dict(saved['decoder'])
        optimizer.load_state_dict(saved['optimizer'])
        rng.setstate(saved['rng']); torch.set_rng_state(saved['torch_rng'].cpu())
        if saved['cuda_rng'] is not None:
            torch.cuda.set_rng_state_all([s.cpu() for s in saved['cuda_rng']])
        start, best, history = saved['epoch'], saved['best'], saved['history']
    else:
        atomic_json(out/'config.json', config)
        base = validation(codec, val_paths, a)
        atomic_json(out/'validation_original.json', base)
        best = base['selection_mse']
        export_best()
        atomic_json(out/'validation_best.json', dict(epoch=0, **base))
    print('train={} validation={} epochs={} start={} ; Kodak excluded'.format(len(train_paths), len(val_paths), a.epochs, start), flush=True)
    for epoch in range(start+1, a.epochs+1):
        codec.decoder.train()
        order = list(train_paths); rng.shuffle(order)
        if a.steps_per_epoch:
            order = order[:a.steps_per_epoch]
        losses = []
        for step, path in enumerate(order, 1):
            x = crop_tensor(path, a.crop, rng, codec.device)
            with torch.no_grad():
                feature = codec.encoder(x, a.snr, a.C, codec.model).float()
            kind, rate = training_condition(rng, a, seed_for(a.seed, epoch, step, 'training-mixture'))
            mask = feature_mask(tuple(feature.shape), a.chunk, rate, kind, a.burst_length, rng.getrandbits(63))
            erased = erase_feature(feature, mask, a.erasure_fill)
            optimizer.zero_grad(set_to_none=True)
            # Separate backward passes keep the two decoder graphs out of memory together.
            loss_value = 0.
            for value, weight in ((erased, 1-a.clean_weight), (feature, a.clean_weight)):
                if weight == 0:
                    continue
                pred = codec.decoder(value, a.snr, codec.model).clamp(0, 1)
                loss = torch.mean((pred-x)**2)*weight
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite training loss')
                loss.backward(); loss_value += float(loss.item())
            norm = torch.nn.utils.clip_grad_norm_(codec.decoder.parameters(), 1.)
            if not torch.isfinite(norm):
                raise ValueError('Nonfinite gradient')
            optimizer.step()
            losses.append(loss_value)
            if step == 1 or step % 10 == 0 or step == len(order):
                print('epoch={} step={}/{} loss={:.6f} {} residual={:.2f}'.format(epoch, step, len(order), loss_value, kind, rate), flush=True)
        for name, param in codec.encoder.named_parameters():
            if param.grad is not None or not torch.equal(param.detach().cpu(), encoder_before[name]):
                raise AssertionError('Frozen encoder changed: '+name)
        val = validation(codec, val_paths, a)
        improved = val['selection_mse'] < best
        if improved:
            best = val['selection_mse']; export_best()
            atomic_json(out/'validation_best.json', dict(epoch=epoch, **val))
        history.append(dict(epoch=epoch, training_mse=float(np.mean(losses)), validation_mse=val['selection_mse'], best=best, improved=improved))
        atomic_json(out/'history.json', history)
        save_torch(out/'last.pt', dict(config=config, epoch=epoch, best=best, history=history,
                   decoder=codec.decoder.state_dict(), optimizer=optimizer.state_dict(),
                   rng=rng.getstate(), torch_rng=torch.get_rng_state(),
                   cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None))
        print('epoch={} validation MSE={:.6f} best={:.6f} encoder unchanged=True'.format(epoch, val['selection_mse'], best), flush=True)
    print('Best decoder: '+str(out/'best.model')+' ; may remain original if validation never improves', flush=True)


if __name__ == '__main__':
    main()
