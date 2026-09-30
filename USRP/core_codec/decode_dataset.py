# -*- coding: utf-8 -*-
"""
decode_dataset.py
=================
批量解码 + 评估：对指定目录里的「复符号 .npy」逐个调用 SwinJSCC 解码器重建图像，
把重建图单独保存到一个文件夹，并计算 PSNR / MSE（可选 SSIM）汇总成指标文件。

设计为「接收端落盘 npy 后」的处理环节：
  接收端 udp/receive_raw.py 把 npy 落到 --npy-dir；本脚本读取该目录并解码。
  后期把来源从「目录」改成「端口」只需替换入口取数，解码/评估逻辑复用本文件。

用法
----
# 一次性处理整个目录
python USRP/core_codec/decode_dataset.py \
  --npy-dir recv_out --out-dir recon \
  --checkpoint weights/SwinJSCC_wo_SAandRA_AWGN_HRimage_snr10_psnr_C96.model \
  --model SwinJSCC_w/o_SAandRA --C 96 --snr 10 \
  --ref-dir datasets/Kodak24 --metrics-json recon/_metrics.json

# 持续监听目录（收到新 npy 就解码），配合实时传输
python USRP/core_codec/decode_dataset.py ... --watch --poll 2.0

元数据来源：优先读取与 npy 同目录同名的 <stem>.meta.json（编码端导出）；
缺失时用命令行 --input-size/--feature-shape 回退（此时不做功率恢复，除非 --orig-power）。
"""
import argparse
import glob
import json
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from USRP.core_codec.latent_storage import is_int8, load_int8  # noqa: E402
from USRP.core_codec.swinjscc_codec import (  # noqa: E402
    SwinJSCCCodec, compute_psnr_mse, load_npy_robust, downsample_of,
    load_embedded_meta, PROJECT_ROOT,
)

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None


def _abspath(p):
    return p if os.path.isabs(p) else str(PROJECT_ROOT / p)


def _load_meta(npy_path, meta_dir=None):
    # 1) 优先读 .npy 内嵌的元数据 trailer（--embed-meta 导出；随链路一起传输）。
    try:
        emb = load_embedded_meta(npy_path)
        if emb:
            return emb, npy_path + '#embedded'
    except Exception:  # noqa: BLE001
        pass
    # 2) 退回同目录/--meta-dir 的 sidecar json。
    stem = os.path.splitext(os.path.basename(npy_path))[0]
    candidates = []
    base_dirs = [os.path.dirname(npy_path)]
    if meta_dir:
        base_dirs.append(_abspath(meta_dir))
    for d in base_dirs:
        candidates.append(os.path.join(d, stem + '.meta.json'))
        candidates.append(os.path.join(d, stem + '.json'))
    for c in candidates:
        if os.path.exists(c):
            with open(c, 'r', encoding='utf-8') as f:
                return json.load(f), c
    return None, None


def _find_ref_image(npy_path, meta, ref_dir):
    stem = os.path.splitext(os.path.basename(npy_path))[0]
    if meta and meta.get('image_stem'):
        stem = meta['image_stem']
    if ref_dir:
        for ext in ('.png', '.jpg', '.jpeg', '.bmp'):
            cand = os.path.join(_abspath(ref_dir), stem + ext)
            if os.path.exists(cand):
                return cand
    if meta and meta.get('source_image') and os.path.exists(meta['source_image']):
        return meta['source_image']
    return None


def _ssim(orig, recon):
    """Calculate RGB SSIM with both current and legacy scikit-image APIs.

    scikit-image < 0.19 uses ``multichannel`` while newer releases use
    ``channel_axis``.  Do not hide import/API errors: callers need to know
    that a missing SSIM is an unavailable metric, not a measured value.
    """
    from skimage.metrics import structural_similarity as ssim_fn
    orig_array = np.asarray(orig, dtype=np.float64)
    recon_array = np.asarray(recon, dtype=np.float64)
    try:
        return float(ssim_fn(orig_array, recon_array,
                             channel_axis=-1, data_range=255.0))
    except TypeError:
        # Compatibility with older scikit-image releases.
        return float(ssim_fn(orig_array, recon_array,
                             multichannel=True, data_range=255.0))


def process_one(npy_path, args, codec):
    """解码单个 npy，返回 metrics dict（含 recon 图路径）。"""
    # 先读元数据（内嵌 trailer 在文件尾部，不受首块丢失影响），再据此健壮读取 npy：
    # 若 seq=0 块丢失被补零导致 npy 头损坏，可用已知符号数重建头（见 load_npy_robust）。
    meta, meta_path = _load_meta(npy_path, args.meta_dir)
    exp_sym = None
    if meta:
        exp_sym = meta.get('num_complex_symbols')
        if exp_sym is None and meta.get('feature_shape'):
            exp_sym = int(np.prod(meta['feature_shape'])) // 2
    if is_int8(meta):
        stats_path = os.path.splitext(npy_path)[0] + '.recv.json'
        if os.path.exists(stats_path):
            with open(stats_path, encoding='utf-8') as f:
                if json.load(f).get('erasure_fill', 0) != 0:
                    raise ValueError('int8 requires zero-filled missing bytes')
        with open(npy_path, 'rb') as f:
            sym = load_int8(f.read(), meta)
    else:
        sym = load_npy_robust(npy_path, expected_symbols=exp_sym).reshape(-1)
        if sym.dtype != np.dtype('complex64'):
            raise ValueError('Non-complex NPY requires valid int8 META; refusing to guess scale')

    if meta:
        feature_shape = meta['feature_shape']
        input_hw = (meta['input_shape'][2], meta['input_shape'][3])
        orig_power = meta.get('original_symbol_power')
    else:
        if not args.feature_shape or not args.input_size:
            raise ValueError('缺少元数据，需提供 --feature-shape 与 --input-size')
        feature_shape = [int(v) for v in args.feature_shape.replace('x', ',').split(',')]
        input_hw = (int(args.input_size), int(args.input_size))
        orig_power = args.orig_power

    restore = not args.no_restore_power and (orig_power is not None)
    recon_hwc = codec.decode(sym, feature_shape, input_hw, orig_power=orig_power,
                             restore_power=restore)
    # 模型在严重退化输入下可能输出 NaN/Inf：落盘与计指标前做一次安全清洗，
    # 避免 PNG 保存报 "invalid value encountered in cast"、PSNR 变 nan。
    _nbad = int((~np.isfinite(recon_hwc)).sum())
    if _nbad:
        print('  [warn] 重建含 {} 个非有限值，已清洗为 0/边界'.format(_nbad))
        recon_hwc = np.nan_to_num(recon_hwc, nan=0.0, posinf=255.0, neginf=0.0)

    stem = os.path.splitext(os.path.basename(npy_path))[0]
    out_png = os.path.join(args.out_dir, stem + '.png')
    if Image is not None:
        Image.fromarray(np.clip(recon_hwc, 0, 255).astype(np.uint8)).save(out_png)
    else:
        np.save(os.path.join(args.out_dir, stem + '.npy'), recon_hwc)

    rec = {
        'npy': os.path.basename(npy_path),
        'recon_png': os.path.basename(out_png),
        'num_complex_symbols': int(sym.shape[0]),
        'feature_shape': feature_shape,
        'input_hw': list(input_hw),
        'restore_power': bool(restore),
        'orig_power': float(orig_power) if orig_power is not None else None,
        'meta_file': os.path.basename(meta_path) if meta_path else None,
        'storage_format': meta.get('storage', {}).get('format', 'complex64') if meta else 'complex64',
    }
    ref = _find_ref_image(npy_path, meta, args.ref_dir)
    if ref and Image is not None:
        orig_img = Image.open(ref).convert('RGB')
        # 参考图需与重建图同尺寸：编码端对原图做了「最大 128 倍数」中心裁剪，
        # 因此这里对参考图做同样的中心裁剪再比较，否则高清图会 shape mismatch。
        rh, rw = recon_hwc.shape[0], recon_hwc.shape[1]
        ow, oh = orig_img.size
        if (oh, ow) != (rh, rw):
            if args.image_size:
                ch = cw = int(args.image_size)
                # 若图像各边小于目标，则退化为「最大 128 倍数」裁剪（不放大）。
                if oh < ch or ow < cw:
                    ch = (oh // 128) * 128 or oh
                    cw = (ow // 128) * 128 or ow
            else:
                ch = (oh // 128) * 128 or oh
                cw = (ow // 128) * 128 or ow
            left, top = (ow - cw) // 2, (oh - ch) // 2
            orig_img = orig_img.crop((left, top, left + cw, top + ch))
        orig = np.asarray(orig_img, dtype=np.float64)
        psnr, mse = compute_psnr_mse(orig, recon_hwc)
        rec.update({'ref_image': os.path.basename(ref), 'psnr_db': psnr, 'mse': mse})
        if args.ssim:
            try:
                rec['ssim'] = _ssim(orig, recon_hwc)
            except Exception as exc:  # noqa: BLE001
                rec['ssim_error'] = '{}: {}'.format(type(exc).__name__, exc)
                if not getattr(args, '_ssim_warning_emitted', False):
                    print('  [warn] SSIM 未计算：{}。请安装/升级 scikit-image。'.format(
                        rec['ssim_error']))
                    args._ssim_warning_emitted = True
    return rec, out_png


def _resolution_key(npy_path, args):
    """排序用：读元数据拿 (H, W)；读不到就排到最后，保持互不影响。"""
    try:
        meta, _ = _load_meta(npy_path, args.meta_dir)
        if meta and meta.get('input_shape'):
            return (int(meta['input_shape'][2]), int(meta['input_shape'][3]))
    except Exception:  # noqa: BLE001
        pass
    return (10 ** 9, 10 ** 9)


def decode_all(args, codec, pending_list, results):
    # 按分辨率分组：让同分辨率的图连续解码，避免「每换一张图就重建一次 Swin 移位窗口
    # 注意力掩码」。诊断实测：纯前向 ~2s，但每次分辨率切换要多花 4~12s 重建掩码；
    # 逐图乱序(6 种 2K 分辨率交织)会把时间放大数倍。分组后掩码最多重建 ~6 次。
    if not getattr(args, 'no_group', False) and len(pending_list) > 1:
        pending_list = sorted(pending_list, key=lambda p: _resolution_key(p, args))
        # 打印分组概览
        groups = OrderedDict()
        for p in pending_list:
            groups.setdefault(_resolution_key(p, args), []).append(p)
        if len(groups) > 1:
            print('[分组] 按分辨率归并解码，共 {} 组：'.format(len(groups)))
            for (h, w), lst in groups.items():
                label = '{}x{}'.format(h, w) if h != 10 ** 9 else '无元数据'
                print('    {} : {} 张'.format(label, len(lst)))
    for npy_path in pending_list:
        # npy 文件名本身已含编码端前缀；输出沿用同一 stem，切勿再次叠加 prefix。
        base = os.path.splitext(os.path.basename(npy_path))[0]
        out_png = os.path.join(args.out_dir, base + '.png')
        out_npy = os.path.join(args.out_dir, base + '.npy')
        if (os.path.exists(out_png) or os.path.exists(out_npy)) and not args.overwrite:
            print('跳过（已解码）', base)
            continue
        try:
            rec, _ = process_one(npy_path, args, codec)
            results[base] = rec
            msg = '[{}] {} 符号={}'.format(base, 'OK', rec['num_complex_symbols'])
            if 'psnr_db' in rec:
                msg += '  PSNR={:.2f}dB MSE={:.2f}'.format(rec['psnr_db'], rec['mse'])
                if 'ssim' in rec:
                    msg += '  SSIM={:.4f}'.format(rec['ssim'])
            print(msg)
        except Exception as e:  # noqa: BLE001
            print('[{}] 解码失败: {}'.format(base, e))


def dump_metrics(args, results):
    if not args.metrics_json:
        return
    rows = list(results.values())
    psnrs_all = [r['psnr_db'] for r in rows if 'psnr_db' in r]
    # 单张出现 NaN/Inf 不应污染整体均值（原实现用 np.mean 会被一票否决成 nan）。
    psnrs = [p for p in psnrs_all if np.isfinite(p)]
    ssims_all = [r['ssim'] for r in rows if 'ssim' in r]
    ssims = [s for s in ssims_all if np.isfinite(s)]
    mean_psnr = float(np.mean(psnrs)) if psnrs else None
    mean_ssim = float(np.mean(ssims)) if ssims else None
    summary = {
        'count': len(rows),
        'n_with_ref': len(psnrs_all),
        'n_finite_psnr': len(psnrs),
        'n_finite_ssim': len(ssims),
        'mean_psnr_db': mean_psnr,
        'mean_ssim': mean_ssim,
        'model': args.model,
        'checkpoint': os.path.abspath(_abspath(args.checkpoint)),
        'decoder_checkpoint': os.path.abspath(_abspath(args.decoder_checkpoint or args.checkpoint)),
        'C': args.C,
        'snr_db': args.snr,
        'npy_dir': os.path.abspath(args.npy_dir),
        'out_dir': os.path.abspath(args.out_dir),
        'ref_dir': os.path.abspath(_abspath(args.ref_dir)) if args.ref_dir else None,
        'per_image': rows,
    }
    path = _abspath(args.metrics_json)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print('指标已写入 {}  (n={}  平均PSNR={}  平均SSIM={})'.format(
        path, summary['count'],
        '{:.2f}dB'.format(mean_psnr) if mean_psnr is not None else 'N/A',
        '{:.4f}'.format(mean_ssim) if mean_ssim is not None else 'N/A'))


def main():
    ap = argparse.ArgumentParser(description='SwinJSCC 批量解码 + 评估（复符号 NPY -> 重建图）')
    ap.add_argument('--npy-dir', required=True, help='复符号 npy 所在目录（接收端落盘目录）')
    ap.add_argument('--out-dir', required=True, help='重建图像输出目录')
    ap.add_argument('--checkpoint', required=True, help='整网权重 .model')
    ap.add_argument('--decoder-checkpoint', default=None, help='Only override decoder weights; preserve original encoder/checkpoint')
    ap.add_argument('--model', default='SwinJSCC_w/o_SAandRA',
                    choices=['SwinJSCC_w/o_SAandRA', 'SwinJSCC_w/_SA',
                             'SwinJSCC_w/_RA', 'SwinJSCC_w/_SAandRA'])
    ap.add_argument('--trainset', default='DIV2K', choices=['DIV2K', 'CIFAR10'])
    ap.add_argument('--model_size', default='base', choices=['small', 'base', 'large'])
    ap.add_argument('--C', type=int, default=96)
    ap.add_argument('--rate', type=int, default=None)
    ap.add_argument('--snr', type=int, default=10)
    ap.add_argument('--ref-dir', default=None, help='参考原图目录（按文件名匹配算 PSNR）')
    ap.add_argument('--image-size', type=int, default=None,
                    help='编码时的裁剪尺寸（与编码端 --image-size 保持一致）；默认 None=自动按重建图尺寸中心裁剪参考图')
    ap.add_argument('--meta-dir', default=None, help='元数据目录（默认与 npy 同目录）')
    ap.add_argument('--metrics-json', default=None, help='指标汇总输出文件')
    ap.add_argument('--prefix', default='', help='npy 文件名前缀（与编码端一致）')
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--ssim', action='store_true', help='额外计算 SSIM（需 scikit-image）')
    ap.add_argument('--no-restore-power', action='store_true', help='解码前不做功率恢复')
    ap.add_argument('--orig-power', type=float, default=None, help='无元数据时的原始符号功率')
    ap.add_argument('--feature-shape', default=None, help='无元数据时的特征形状，如 1,256,96')
    ap.add_argument('--input-size', type=int, default=None, help='无元数据时的输入边长')
    ap.add_argument('--watch', action='store_true', help='持续监听目录，有新 npy 就解码')
    ap.add_argument('--poll', type=float, default=2.0, help='监听轮询间隔(秒)')
    ap.add_argument('--device', default=None)
    ap.add_argument('--amp', action='store_true',
                    help='fp16 混合精度解码（仅 CUDA；与编码端对称，显著加速）')
    ap.add_argument('--no-group', action='store_true',
                    help='不按分辨率分组（默认会分组，让同分辨率的图连续解码以复用注意力掩码）')
    args = ap.parse_args()

    args.npy_dir = _abspath(args.npy_dir)
    args.out_dir = _abspath(args.out_dir)
    args.metrics_json = _abspath(args.metrics_json) if args.metrics_json else None
    os.makedirs(args.out_dir, exist_ok=True)

    codec = SwinJSCCCodec(args.checkpoint, args.model, args.trainset, args.model_size,
                          args.C, args.rate, args.snr, args.device, args.amp,
                          decoder_checkpoint=args.decoder_checkpoint)
    results = OrderedDict()

    def scan():
        return sorted(glob.glob(os.path.join(args.npy_dir, args.prefix + '*.npy')))

    if not args.watch:
        files = scan()
        if not files:
            raise SystemExit('目录 {} 下没有 npy'.format(args.npy_dir))
        print('待解码 npy {} 个'.format(len(files)))
        decode_all(args, codec, files, results)
        dump_metrics(args, results)
        print('=== 解码完成 ===')
        return

    print('监听目录 {} 中（每 {:.1f}s 轮询，Ctrl-C 退出）...'.format(args.npy_dir, args.poll))
    seen = set()
    try:
        while True:
            new = [f for f in scan() if f not in seen]
            if new:
                decode_all(args, codec, new, results)
                seen.update(new)
                dump_metrics(args, results)
            time.sleep(args.poll)
    except KeyboardInterrupt:
        print('\n停止监听。')
        dump_metrics(args, results)


if __name__ == '__main__':
    main()
