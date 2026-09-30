# -*- coding: utf-8 -*-
"""
swinjscc_config.py
==================
从官方 main.py 的 config 里**直接读取**编码器/解码器结构参数，避免把
embed_dims / depths / num_heads / window_size / downsample 等超参表复制一份到
本工程。这样网络结构始终以官方代码为唯一真源（single source of truth）。

原理
----
官方 main.py 在模块级执行 argparse 并定义 class config()，其属性
`encoder_kwargs` / `decoder_kwargs` / `downsample` / `image_dims` 正是构建
官方 SwinJSCC 网络所需的全部结构参数。main.py 的训练/测试流程被
`if __name__ == '__main__'` 保护，因此**导入 main 不会触发训练**。

为避免 main.py 的 argparse 误读本工程的命令行参数，这里在导入前临时把
sys.argv 换成一组受控参数（trainset / model / model_size / C），导入后再还原。
"""
import importlib
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 同一进程内按参数缓存官方 config，避免重复导入/重载。
_CACHE = {}


def _load_official_config(trainset='DIV2K', model='SwinJSCC_w/o_SAandRA',
                          model_size='base', C=96):
    """用受控 argv 导入官方 main，返回其 config 实例（含 encoder_kwargs 等）。"""
    key = (trainset, model, model_size, int(C) if C is not None else None)
    if key in _CACHE:
        return _CACHE[key]

    saved_argv = list(sys.argv)
    saved_cuda = os.environ.get('CUDA_VISIBLE_DEVICES')
    # 官方 main 只认这些开关；其余参数交给默认值，避免解析错误。
    sys.argv = ['main', '--trainset', str(trainset), '--model', str(model),
                '--model_size', str(model_size), '--C', str(C)]
    try:
        if 'main' in sys.modules:
            # 已导入过则按新参数重载，确保 config 反映当前参数。
            main = importlib.reload(sys.modules['main'])
        else:
            main = importlib.import_module('main')
    finally:
        sys.argv = saved_argv
        if saved_cuda is None:
            os.environ.pop('CUDA_VISIBLE_DEVICES', None)
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = saved_cuda

    cfg = main.config()
    _CACHE[key] = cfg
    return cfg


def get_model_kwargs(trainset='DIV2K', model='SwinJSCC_w/o_SAandRA',
                     model_size='base', C=96, img_size=None):
    """返回官方 (encoder_kwargs, decoder_kwargs, downsample)。

    img_size 非空时覆盖结构中的初始分辨率（官方默认 HR=256 / CIFAR=32）；
    实际分辨率仍由调用方在 forward 前用 update_resolution(H,W) 设置。
    """
    cfg = _load_official_config(trainset, model, model_size, C)
    enc = dict(cfg.encoder_kwargs)
    dec = dict(cfg.decoder_kwargs)
    if img_size is not None:
        # 官方 image_dims 形如 (3, H, W)；img_size 可能是 (H, W) 元组或单个 int。
        # 必须先判类型再取值，否则对元组调用 int() 会抛
        # "int() argument must be ... not 'tuple'"。
        if isinstance(img_size, (tuple, list)):
            hw = (int(img_size[0]), int(img_size[1]))
        else:
            n = int(img_size)
            hw = (n, n)
        enc['img_size'] = hw
        dec['img_size'] = hw
    return enc, dec, cfg.downsample


def get_downsample(trainset='DIV2K'):
    """编码器总下采样指数（HR=4 / CIFAR=2），直接取自官方 config。"""
    cfg = _load_official_config(trainset)
    return cfg.downsample


def get_image_dims(trainset='DIV2K'):
    """官方 image_dims，如 (3, 256, 256)。"""
    cfg = _load_official_config(trainset)
    return cfg.image_dims


if __name__ == '__main__':
    for ts in ('DIV2K', 'CIFAR10'):
        cfg = _load_official_config(ts)
        print('trainset={}  downsample={}  image_dims={}'.format(
            ts, getattr(cfg, 'downsample', None), getattr(cfg, 'image_dims', None)))
    enc, dec, ds = get_model_kwargs('DIV2K', 'SwinJSCC_w/o_SAandRA', 'base', 96, img_size=256)
    print('encoder_kwargs keys:', sorted(enc.keys()))
    print('decoder_kwargs keys:', sorted(dec.keys()))
    print('downsample =', ds)
