# -*- coding: utf-8 -*-
"""
encode_dataset.py
=================
批量编码：对指定数据集目录里的每张图片调用 SwinJSCC 编码器，导出
「复基带符号 .npy + 元数据 .meta.json」，供后续 udp/send_raw.py 使用。

相比 export_latent_npy.py（单图）：本脚本一次加载模型，遍历整个目录，命令行可指定
数据集、输出目录、图片匹配、裁剪尺寸、模型/权重/SNR/C 等。

输出命名：默认用「数据集目录名」做前缀，例如 --data-dir datasets/Kodak24 ->
          <out-dir>/Kodak24_01.npy        复符号（complex64，内存即交错 I/Q）
          <out-dir>/Kodak24_01.meta.json  元数据（--no-sidecar 可关闭）
可关闭前缀：--no-dataset-tag；或自定义：--prefix pre_。

元数据内嵌：--embed-meta 把元数据 JSON 作为 trailer 追加到 .npy 末尾，一个文件同时
携带符号+元数据，随 UDP 一起传输，接收端落盘即可解码，无需再单独发 .json。

用法
----
python USRP/core_codec/encode_dataset.py \
  --data-dir datasets/Kodak24 --out-dir out_batch/kodak24 \
  --checkpoint weights/SwinJSCC_wo_SAandRA_AWGN_HRimage_snr10_psnr_C96.model \
  --model SwinJSCC_w/o_SAandRA --C 96 --snr 10 --image-size 256

可选：--limit N 只处理前 N 张；--pattern '*.png'；--prefix pre_；--qpsk-bits；
      --recursive；--overwrite（默认跳过已存在的 npy）。
"""
import argparse
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from USRP.core_codec.latent_storage import quantize  # noqa: E402
from USRP.core_codec.swinjscc_codec import (  # noqa: E402
    SwinJSCCCodec, list_images, qpsk_pack, embed_meta_into_bytes, PROJECT_ROOT,
)


def main():
    ap = argparse.ArgumentParser(description='SwinJSCC 批量编码：数据集目录 -> 复符号 NPY')
    ap.add_argument('--data-dir', required=True, help='待编码图片目录')
    ap.add_argument('--out-dir', required=True, help='输出目录（npy + meta.json）')
    ap.add_argument('--checkpoint', required=True, help='整网权重 .model')
    ap.add_argument('--model', default='SwinJSCC_w/o_SAandRA',
                    choices=['SwinJSCC_w/o_SAandRA', 'SwinJSCC_w/_SA',
                             'SwinJSCC_w/_RA', 'SwinJSCC_w/_SAandRA'])
    ap.add_argument('--trainset', default='DIV2K', choices=['DIV2K', 'CIFAR10'])
    ap.add_argument('--model_size', default='base', choices=['small', 'base', 'large'])
    ap.add_argument('--C', type=int, default=96, help='固定码率模型瓶颈通道数')
    ap.add_argument('--rate', type=int, default=None, help='RA 模型传输通道数')
    ap.add_argument('--snr', type=int, default=10, help='条件 SNR(dB)')
    ap.add_argument('--image-size', type=int, default=None,
                    help='可选：把图中心裁剪为 NxN 再编码（如 256，最快且对齐官方训练尺度）。'
                         '默认 None=按原图最大 128 倍数裁剪，支持高清图（如 CLIC2022）')
    ap.add_argument('--pattern', default=None, help='图片 glob，如 *.png；默认 png/jpg/jpeg/bmp')
    ap.add_argument('--recursive', action='store_true', help='递归搜索子目录')
    ap.add_argument('--prefix', default='', help='输出文件名前缀（给定时优先于数据集名）')
    ap.add_argument('--no-dataset-tag', action='store_true',
                    help='不用数据集目录名作前缀（默认会用，如 Kodak24_01）')
    ap.add_argument('--limit', type=int, default=None, help='只处理前 N 张（调试用）')
    ap.add_argument('--overwrite', action='store_true', help='即使 npy 已存在也重新编码')
    ap.add_argument('--embed-meta', action='store_true',
                    help='把元数据 JSON 作为 trailer 内嵌进 .npy（随链路一起传输，推荐）')
    ap.add_argument('--no-sidecar', action='store_true',
                    help='不另写 <stem>.meta.json（通常与 --embed-meta 搭配）')
    ap.add_argument('--power', type=float, default=1.0, help='目标复符号平均功率')
    ap.add_argument('--storage', choices=['complex64', 'int8'], default='complex64',
                    help='int8: symmetric quantized I/Q NPY, requires META for decoding')
    ap.add_argument('--qpsk-bits', action='store_true', help='额外导出 QPSK 打包比特 .bits.npy')
    ap.add_argument('--sensitivity-importance', action='store_true',
                    help='计算 decoder-gradient 重要性并输出 .importance.json，供 Level 2 发送端使用')
    ap.add_argument('--importance-group-symbols', type=int, default=150,
                    help='每个重要性组的复符号数；chunk=1200 时 150 与一个 complex64 payload 对齐')
    ap.add_argument('--device', default=None, help='cuda / cpu，默认自动')
    ap.add_argument('--amp', action='store_true',
                    help='CUDA 下启用 fp16 混合精度前向（更快，数值略有差异；默认关闭）')
    args = ap.parse_args()
    if args.storage == 'int8' and (not args.embed_meta or args.sensitivity_importance or args.qpsk_bits):
        ap.error('int8 requires --embed-meta and does not support sensitivity/qpsk exports')
    if args.sensitivity_importance and args.importance_group_symbols <= 0:
        ap.error('--importance-group-symbols 必须为正整数')

    data_dir = args.data_dir if os.path.isabs(args.data_dir) else str(PROJECT_ROOT / args.data_dir)
    out_dir = args.out_dir if os.path.isabs(args.out_dir) else str(PROJECT_ROOT / args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    images = list_images(data_dir, args.pattern, args.recursive)
    if args.limit:
        images = images[:args.limit]
    if not images:
        raise SystemExit('在 {} 下没找到图片（pattern={}）'.format(data_dir, args.pattern))
    print('待编码图片 {} 张，来自 {}'.format(len(images), data_dir))

    codec = SwinJSCCCodec(args.checkpoint, args.model, args.trainset, args.model_size,
                          args.C, args.rate, args.snr, args.device, args.amp)

    # 输出前缀：--prefix 优先；否则用数据集目录名（如 --data-dir datasets/Kodak24 -> Kodak24_）。
    if args.prefix:
        tag = args.prefix
    elif args.no_dataset_tag:
        tag = ''
    else:
        ds_name = os.path.basename(os.path.normpath(data_dir))
        tag = (ds_name + '_') if ds_name else ''
    if tag:
        print('输出前缀 = "{}"（例：{}01.npy）'.format(tag, tag))

    ok, skipped, failed = 0, 0, 0
    for i, img in enumerate(images, 1):
        stem = os.path.splitext(os.path.basename(img))[0]
        base = tag + stem
        npy_path = os.path.join(out_dir, base + '.npy')
        meta_path = os.path.join(out_dir, base + '.meta.json')
        if os.path.exists(npy_path) and not args.overwrite:
            skipped += 1
            print('[{}/{}] 跳过（已存在）{}'.format(i, len(images), base + '.npy'))
            continue
        try:
            t0 = time.time()
            sym, meta, _ = codec.encode(
                img, image_size=args.image_size, power=args.power,
                sensitivity_group_symbols=(args.importance_group_symbols
                                           if args.sensitivity_importance else None))
            elapsed = time.time() - t0
            stored = sym
            if args.storage == 'int8':
                stored, meta = quantize(sym, meta)
            meta['npy_file'] = os.path.basename(npy_path)
            meta['output_stem'] = base
            if args.qpsk_bits:
                packed, nbits = qpsk_pack(sym)
                bits_path = os.path.join(out_dir, base + '.bits.npy')
                np.save(bits_path, packed)
                meta['qpsk_bits_file'] = os.path.basename(bits_path)
                meta['qpsk_bits'] = int(nbits)
                meta['qpsk_bytes'] = int(os.path.getsize(bits_path))

            if args.embed_meta:
                # 一个 .npy 同时携带符号 + 元数据（trailer），随链路一起传输。
                buf = io.BytesIO()
                np.save(buf, stored)
                raw = embed_meta_into_bytes(buf.getvalue(), meta)
                with open(npy_path, 'wb') as f:
                    f.write(raw)
            else:
                np.save(npy_path, stored)
            meta['npy_bytes'] = int(os.path.getsize(npy_path))

            if args.sensitivity_importance:
                importance = codec.last_sensitivity_importance
                if not importance:
                    raise RuntimeError('未生成 sensitivity importance')
                importance_path = os.path.join(out_dir, base + '.importance.json')
                with open(importance_path, 'w', encoding='utf-8') as f:
                    json.dump(importance, f, ensure_ascii=False, separators=(',', ':'))
                # 只在普通 META 中记录 sidecar 的身份，不把 scores 内嵌进 .npy/META。
                meta['importance_sidecar'] = os.path.basename(importance_path)
                meta['importance_scheme'] = importance['scheme']
                meta['importance_group_symbols'] = importance['group_symbols']

            if not args.no_sidecar:
                # 默认写 sidecar；--no-sidecar 时只保留（可能内嵌的）元数据不落 json。
                with open(meta_path, 'w', encoding='utf-8') as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)
            ok += 1
            suffix = ' + sensitivity' if args.sensitivity_importance else ''
            print('[{}/{}] {} -> {} 个复符号, CBR={:.4f}, {:.1f} KB, {:.2f}s{}'.format(
                i, len(images), base, meta['num_complex_symbols'], meta['cbr'],
                meta['npy_bytes'] / 1024.0, elapsed, suffix))
        except Exception as e:  # noqa: BLE001
            failed += 1
            print('[{}/{}] 失败 {}: {}'.format(i, len(images), stem, e))

    print('=== 批量编码完成 === 成功 {}  跳过 {}  失败 {}  输出目录 {}'.format(
        ok, skipped, failed, out_dir))


if __name__ == '__main__':
    main()
