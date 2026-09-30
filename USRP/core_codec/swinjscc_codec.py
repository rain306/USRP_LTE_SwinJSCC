# -*- coding: utf-8 -*-
"""
swinjscc_codec.py
=================
主机端 SwinJSCC 编解码封装。

设计原则（**import 耦合，不复制官方逻辑**）：
  * 网络结构参数  -> 从官方 main.py 的 config 读取（见 swinjscc_config.get_model_kwargs）
  * 网络构建       -> 调用官方 net.encoder.create_encoder / net.decoder.create_decoder
  * 权重加载       -> 直接 load_state_dict 官方整网 .model 的 encoder.* / decoder.*
  * 功率归一化     -> 复用官方 net.channel.Channel.complex_normalize（不再自己重写公式）
  * 图像预处理     -> 复用官方 data.datasets.Datasets 的确定性 CenterCrop+ToTensor 逻辑
  * 解码幅度恢复   -> 与官方 Channel.forward 一致：feature = cat([real, imag]) * sqrt(pwr)

本文件只保留官方未以「可导入函数」形式暴露的最小薄层（复符号 real/imag 拼接、
PSNR/QPSK 等评估工具），其余一律调用官方实现。

依赖：numpy 顶层可用；torch / torchvision / PIL 及官方 net/data 模块在
encode()/decode() 调用时才导入，便于无 torch 环境做纯 numpy 自检。
"""
import io
import json
import os
import re
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from USRP.core_codec.swinjscc_config import get_model_kwargs, get_downsample  # noqa: E402


# ---------------------------------------------------------------------------
# 权重加载（官方整网 .model 拆出 encoder./decoder. 前缀）
# ---------------------------------------------------------------------------
def _extract_prefix(ckpt_path, prefix, drop_attn_mask=True):
    import torch
    sd = torch.load(ckpt_path, map_location='cpu')
    if sd.get('format') == 'swinjscc_explicit_mask_v1':
        raise ValueError('Experimental mask checkpoint is not supported by this clean50 pipeline')
    sub = {}
    for k, v in sd.items():
        if not k.startswith(prefix):
            continue
        key = k[len(prefix):]
        # attn_mask 是随分辨率重建的 buffer（update_resolution 会按实际分辨率重算），
        # 它在 checkpoint 里绑定的是「训练分辨率」，直接加载会让高分辨率输入报
        # size mismatch（load_state_dict 即使 strict=False 也会对形状不符报错）。
        # 这里跳过它，交由 update_resolution 重建 —— 于是任意 128 倍数分辨率都能编码。
        if drop_attn_mask and key.endswith('attn_mask'):
            continue
        sub[key] = v
    if not sub:
        raise RuntimeError('checkpoint 中找不到 {}*. 参数：{}'.format(prefix, ckpt_path))
    return sub


def load_encoder_weights(encoder, ckpt_path):
    sub = _extract_prefix(ckpt_path, 'encoder.')
    missing, unexpected = encoder.load_state_dict(sub, strict=False)
    if missing:
        print('[warn] 编码器缺失参数 {} 个（例如 {}）'.format(len(missing), missing[:3]))
    if unexpected:
        print('[warn] 编码器多余参数 {} 个（例如 {}）'.format(len(unexpected), unexpected[:3]))
    return sub


def load_decoder_weights(decoder, ckpt_path):
    sub = _extract_prefix(ckpt_path, 'decoder.')
    missing, unexpected = decoder.load_state_dict(sub, strict=False)
    if missing:
        print('[warn] 解码器缺失参数 {} 个（例如 {}）'.format(len(missing), missing[:3]))
    if unexpected:
        print('[warn] 解码器多余参数 {} 个（例如 {}）'.format(len(unexpected), unexpected[:3]))
    return sub


# ---------------------------------------------------------------------------
# 官方模块的薄封装
# ---------------------------------------------------------------------------
def make_official_channel(device='cpu'):
    """构造官方 Channel（channel_type='none'），仅用于其 complex_normalize 归一化。

    注意：不能用 `class _Cfg: device = device` 的写法——类体不捕获外层函数作用域，
    会让 RHS 的 `device` 变成未定义名（NameError: name 'device' is not defined）。
    官方 Channel 仅按属性名读取 config.CUDA / config.device / config.logger，
    因此这里用 SimpleNamespace 承载即可。
    """
    from types import SimpleNamespace
    from net.channel import Channel

    class _Args:
        channel_type = 'none'
        multiple_snr = '10'

    cfg = SimpleNamespace(CUDA=(str(device) != 'cpu'), device=device, logger=None)
    return Channel(_Args(), cfg)


def preprocess_image(image_path, trainset='DIV2K', image_size=None):
    """对齐官方 data/datasets.py::Datasets 的确定性预处理（CenterCrop + ToTensor）。

    官方 Datasets 把边长向下取到 128 的倍数后 CenterCrop；本函数在 HR 且未指定
    image_size 时沿用该规则；指定 image_size 时按 NxN 中心裁剪（对齐官方 HR 权重
    的 256 训练尺度）。返回 (numpy CHW [1,3,H,W] float32, HWC 0-255 原图)。
    """
    from PIL import Image
    from torchvision import transforms

    image = Image.open(image_path).convert('RGB')
    if trainset == 'CIFAR10':
        chw = transforms.ToTensor()(image).unsqueeze(0)
        return chw.numpy(), np.asarray(image, dtype=np.float32)

    w, h = image.size  # PIL: (宽, 高)
    if image_size:
        h = w = int(image_size)
    else:
        h -= h % 128
        w -= w % 128
    left, top = (image.size[0] - w) // 2, (image.size[1] - h) // 2
    crop = image.crop((left, top, left + w, top + h))
    chw = transforms.ToTensor()(crop).unsqueeze(0)
    return chw.numpy(), np.asarray(crop, dtype=np.float32)


# ---------------------------------------------------------------------------
# 纯 numpy 工具（无需 torch）
# ---------------------------------------------------------------------------
# 内嵌元数据 trailer 的魔数（8 字节）。文件布局：
#   [原始 npy 字节] + MAGIC(8B) + JSON长度L(8B, big-endian) + JSON(L 字节, utf-8)
# 这样「一个 .npy = 复符号 + 元数据」，随 UDP 一起传输、接收端落盘后即可直接解码，
# 无需再单独传 .json；std(numpy) 读取前由本模块剥离 trailer。
META_MAGIC = b'SJMETA01'


def embed_meta_into_bytes(npy_bytes, meta):
    """把 meta(dict) 作为 trailer 追加到 npy 字节流末尾，返回新的字节流。

    布局：[npy 字节] + [JSON(L)] + MAGIC(8B) + 长度L(8B, big-endian)
    尾部固定 16B 为 MAGIC+长度，便于从文件末尾快速定位并剥离。
    """
    payload = json.dumps(meta, ensure_ascii=False).encode('utf-8')
    return npy_bytes + payload + META_MAGIC + len(payload).to_bytes(8, 'big')


def split_meta_trailer(raw):
    """从原始字节流中剥离 trailer。返回 (npy_bytes, meta or None)。"""
    if len(raw) < 16:
        return raw, None
    L = int.from_bytes(raw[-8:], 'big')          # 末尾 8B = JSON 长度
    if L <= 0 or L + 16 > len(raw):
        return raw, None
    if raw[-16:-8] != META_MAGIC:                # 前 8B = 魔数
        return raw, None
    try:
        meta = json.loads(raw[-16 - L:-16].decode('utf-8'))
    except Exception:  # noqa: BLE001
        return raw, None
    return raw[:-16 - L], meta


def load_embedded_meta(path):
    """读取 .npy 内嵌元数据；没有则返回 None。"""
    with open(path, 'rb') as f:
        raw = f.read()
    _, meta = split_meta_trailer(raw)
    return meta


def _rebuild_npy_header(n_elem):
    """用 numpy 自身生成 (n_elem,) complex64 的规范 .npy 头（仅头，不含数据）。

    用途：UDP 传输中「承载 npy 魔数头的首个块（seq=0）整块丢失」时，接收端会把该
    块补零，导致文件头被零覆盖、numpy 报 `Cannot load file containing pickled data`。
    这里复用 numpy 写头实现，保证与发送端原始头「逐字节等长」，从而可用 len(header)
    精确定位数据体起点（见 load_npy_robust，避免错位）；头部若干元素退化为 0。
    """
    n_elem = int(n_elem)
    buf = io.BytesIO()
    np.save(buf, np.zeros(n_elem, dtype=np.complex64))
    full = buf.getvalue()
    return full[:len(full) - n_elem * 8]


def load_npy_robust(path, expected_symbols=None):
    """健壮读取 .npy：某些 FUSE 挂载直接 mmap 会在关闭阶段触发 EIO，退回内存解析。
    若文件带内嵌元数据 trailer（见 embed_meta_into_bytes），先剥离再解析。
    若头损坏（如首块丢失被补零），且已知符号数，则重建头后解析。"""
    try:
        with open(path, 'rb') as f:
            raw = f.read()
        data, emb = split_meta_trailer(raw)
        try:
            return np.load(io.BytesIO(data), allow_pickle=False)
        except ValueError:
            # 头损坏：用已知符号数反推头长并重建（data = 头 + 元素数×8）。
            n = expected_symbols
            if n is None and emb:
                n = emb.get('num_complex_symbols')
                if n is None and emb.get('feature_shape'):
                    n = int(np.prod(emb['feature_shape'])) // 2
            if n is None:
                raise
            n = int(n)
            # 不能用「len(data) − n×8」反推头长：若承载内嵌元数据的尾块也一起丢失，
            # 文件会多出 trailer 长度，反推的头长被撑大 → 数据体整体错位 → 解码 NaN。
            # 正确做法：用 numpy 规范头长度精确锚定，数据体=紧随规范头之后的 n×8 字节。
            header = _rebuild_npy_header(n)
            hlen = len(header)
            if len(raw) < hlen + n * 8:
                raise
            fixed = header + raw[hlen:hlen + n * 8]
            return np.load(io.BytesIO(fixed), allow_pickle=False)
    except OSError:
        return np.load(path, allow_pickle=False)


def symbols_to_feature(sym, feature_shape, orig_power=None, restore_power=True):
    """复符号 -> 编码器实特征（与官方 net/channel.py::Channel 的 real/imag 拼接逆运算）。

    官方 Channel.forward 把实特征 reshape(-1) 后：前一半作实部、后一半作虚部，
    信道输出再 cat([real, imag]) 并乘 sqrt(pwr) 还原幅度。本函数即该逆映射。
    """
    sym = np.asarray(sym).reshape(-1).astype(np.complex128)
    n_sym = sym.shape[0]
    feat = np.empty(2 * n_sym, dtype=np.float64)
    feat[:n_sym] = sym.real
    feat[n_sym:] = sym.imag
    if restore_power and orig_power is not None:
        feat = feat * np.sqrt(float(orig_power))
    n_expected = int(np.prod(feature_shape))
    if feat.shape[0] != n_expected:
        raise ValueError('特征元素数不匹配：符号 {} 个 -> 实数 {} 个，但目标 {} 需要 {} 个。'.format(
            n_sym, feat.shape[0], feature_shape, n_expected))
    return feat.reshape(feature_shape).astype(np.float32)


# ---------------------------------------------------------------------------
# 显式擦除占位符（erasure placeholder），参考 Block Erasure-Aware JSCC 的思路。
# 原生流程把丢失的 DATA 分块补零；0 是 latent 的合法取值，解码器无法区分
# “这块被擦除”与“这块本来就是 0”。这里提供可选的“显式占位符”：把被擦除
# 分块覆盖的复符号改写成固定哨兵值（论文用 -1），使擦除位置在解码器输入上
# 呈现可识别的常量模式。
# 注意：只有当解码器经过擦除感知训练时，占位符才可能带来稳定收益；未训练时
# 它等价于把补零换成另一个常量，仅用于对照实验，不保证提升。
ERASURE_PLACEHOLDER_DEFAULT = 0.0


def chunk_symbol_slice(head, chunk, seq, nbytes):
    """文件字节 DATA 分块 seq 覆盖的复符号下标区间 [lo, hi)，无覆盖时返回 None。"""
    lo = max(head, seq * chunk)
    hi = min(head + nbytes, (seq + 1) * chunk)
    if hi <= lo:
        return None
    return (lo - head) // 8, (hi - head) // 8


def apply_erasure_placeholder(symbols, head, chunk, erased_seqs, fill=0.0):
    """把被擦除分块覆盖的复符号写成显式占位符。

    fill == 0.0 复现原生补零行为；fill != 0.0 时实部与虚部都取该值，得到一个
    解码器理论上可识别的哨兵值。
    """
    if fill == 0.0:
        return symbols
    out = np.array(symbols, copy=True)
    marker = np.complex64(complex(float(fill), float(fill)))
    nbytes = out.nbytes
    for seq in erased_seqs:
        sl = chunk_symbol_slice(head, chunk, seq, nbytes)
        if sl is not None:
            out[sl[0]:sl[1]] = marker
    return out


def compute_psnr_mse(orig, recon):
    """两幅同尺寸 0-255 图像（HWC）的 PSNR(dB) 与 MSE。"""
    orig = np.asarray(orig, dtype=np.float64)
    recon = np.asarray(recon, dtype=np.float64)
    if orig.shape != recon.shape:
        raise ValueError('尺寸不一致：{} vs {}'.format(orig.shape, recon.shape))
    mse = float(np.mean((orig - recon) ** 2))
    if mse <= 0:
        return float('inf'), 0.0
    return float(10.0 * np.log10((255.0 ** 2) / mse)), mse


def qpsk_pack(sym):
    """单位功率复符号按象限硬判决为 QPSK：1 符号 -> 2 bit（I,Q），MSB-first 打包 uint8。"""
    bits = np.empty(2 * sym.shape[0], dtype=np.uint8)
    bits[0::2] = (sym.real >= 0).astype(np.uint8)
    bits[1::2] = (sym.imag >= 0).astype(np.uint8)
    pad = (-bits.shape[0]) % 8
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    return np.packbits(bits), int(bits.shape[0])


def compute_cbr(model, feature_numel, input_numel, rate, trainset):
    """CBR 公式取自官方 net/network.py::SwinJSCC.forward（该处为内联表达式，无独立函数可导入）。"""
    if model in ('SwinJSCC_w/o_SAandRA', 'SwinJSCC_w/_SA'):
        return feature_numel / 2 / input_numel
    downsample = get_downsample(trainset)
    return rate / (2 * 3 * 2 ** (downsample * 2))


def downsample_of(trainset):
    """编码器总下采样指数，取自官方 config。"""
    return get_downsample(trainset)


def list_images(data_dir, pattern='*.png', recursive=False):
    """枚举目录下的图片（默认 png/jpg/jpeg/bmp），按名字排序。"""
    import glob
    data_dir = str(data_dir)
    pats = [pattern] if pattern else ['*.png', '*.jpg', '*.jpeg', '*.bmp']
    files = []
    for p in pats:
        if recursive:
            files += glob.glob(os.path.join(data_dir, '**', p), recursive=True)
        else:
            files += glob.glob(os.path.join(data_dir, p))
    return sorted(set(files))


# ---------------------------------------------------------------------------
# 编解码器（结构/权重/归一化全部走官方实现）
# ---------------------------------------------------------------------------
class SwinJSCCCodec:
    """一次构建 encoder+decoder（官方结构+官方权重），供整批图片复用。"""

    def __init__(self, checkpoint, model='SwinJSCC_w/o_SAandRA', trainset='DIV2K',
                 model_size='base', C=96, rate=None, snr=10, device=None, amp=False,
                 decoder_checkpoint=None):
        import torch
        self.checkpoint = str(checkpoint)
        if not os.path.isabs(self.checkpoint):
            self.checkpoint = str(PROJECT_ROOT / self.checkpoint)
        if not os.path.exists(self.checkpoint):
            raise FileNotFoundError('checkpoint 不存在: {}'.format(self.checkpoint))
        self.decoder_checkpoint = self.checkpoint
        if decoder_checkpoint:
            self.decoder_checkpoint = str((PROJECT_ROOT / decoder_checkpoint).resolve())
            if not os.path.isfile(self.decoder_checkpoint):
                raise FileNotFoundError(self.decoder_checkpoint)
        self.torch = torch
        self.model = model
        self.trainset = trainset
        self.model_size = model_size
        self.C = C
        self.snr = snr
        self.downsample = get_downsample(trainset)
        if model in ('SwinJSCC_w/_RA', 'SwinJSCC_w/_SAandRA'):
            if rate is None:
                raise ValueError('--model {} 需要 --rate'.format(model))
            self.rate = rate
        else:
            self.rate = C
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        # 可选混合精度（fp16 autocast）：显存/带宽减半，2K 前向通常有 1.5~2 倍加速；
        # 默认关闭以保证与官方 fp32 数值一致。仅 CUDA 生效。
        self.amp = bool(amp) and str(self.device).startswith('cuda')
        # conv 算法自动调优：对结果无数值影响，仅首次调用多花一点时间。
        if str(self.device).startswith('cuda'):
            torch.backends.cudnn.benchmark = True
        # 打印真实算力设备：若为 cpu，2K 图逐张前向会到秒级，属预期而非 bug。
        print('[device] 编码设备 = {}; amp = {}; torch.cuda.is_available() = {}; torch = {}'.format(
            self.device, self.amp, torch.cuda.is_available(), torch.__version__))
        if torch.cuda.is_available():
            print('[device] GPU = {}'.format(torch.cuda.get_device_name(0)))
        if model != 'SwinJSCC_w/o_SAandRA' and self.device == 'cpu':
            print('[warn] {} 分支内部使用 CUDA 张量，CPU 下可能报错。'.format(model))
        self.encoder = None
        self.decoder = None
        self.channel = None
        self.H = self.W = None
        # 由 encode(..., sensitivity_group_symbols=N) 生成；编码脚本将其写入
        # 独立 sidecar，避免把数百个分数塞进需要空口传输的 META。
        self.last_sensitivity_importance = None

    def _amp_ctx(self):
        """fp16 autocast 上下文，兼容新旧 torch：
        - torch>=1.11 : torch.autocast(device_type='cuda', dtype=float16)
        - torch>=1.10 : torch.cuda.amp.autocast(dtype=float16)
        - torch 1.6-1.9: torch.cuda.amp.autocast()  # 该版本 __init__ 不接受 dtype 参数
        都不满足则退回空上下文（fp32）。"""
        torch = self.torch
        if not self.amp:
            return nullcontext()
        autocast = getattr(torch, 'autocast', None)
        if autocast is not None:
            try:
                return autocast(device_type='cuda', dtype=torch.float16)
            except TypeError:
                pass
        amp = getattr(getattr(torch, 'cuda', None), 'amp', None)
        if amp is not None and hasattr(amp, 'autocast'):
            try:
                return amp.autocast(dtype=torch.float16)   # torch>=1.10
            except TypeError:
                return amp.autocast()                       # torch 1.6~1.9，默认 fp16
        return nullcontext()

    def channel_number(self):
        return self.C if self.model in ('SwinJSCC_w/o_SAandRA', 'SwinJSCC_w/_SA') else self.rate

    def _ensure_models(self, H, W):
        # 官方模型只构建/加载权重一次：分辨率变化统一用官方 update_resolution 适配。
        # 旧实现每当 (H,W) 变化就重建网络并重新 torch.load 上百 MB 的 checkpoint，
        # 批量处理不同尺寸的高清图时，这一开销会叠加在每张图上（~10s/张的主因）。
        if self.encoder is None:
            from net.encoder import create_encoder
            from net.decoder import create_decoder
            # 结构参数直接取自官方 config；img_size 仅为初值，随即被 update_resolution 覆盖。
            enc_kw, dec_kw, _ = get_model_kwargs(self.trainset, self.model, self.model_size,
                                                 self.C, img_size=(H, W))
            self.encoder = create_encoder(**enc_kw).to(self.device).eval()
            self.decoder = create_decoder(**dec_kw).to(self.device).eval()
            load_encoder_weights(self.encoder, self.checkpoint)
            load_decoder_weights(self.decoder, self.decoder_checkpoint)
            if self.channel is None:
                self.channel = make_official_channel(self.device)
            self.H = self.W = None
        if self.H != H or self.W != W:
            # 与官方 test() 一致：分辨率变化只重建随尺寸变化的 attention mask、同步各阶段尺寸。
            self.encoder.update_resolution(H, W)
            self.decoder.update_resolution(H // (2 ** self.downsample), W // (2 ** self.downsample))
            self.H, self.W = H, W

    # ---- 编码：图像 -> 复符号（归一化复用官方 Channel.complex_normalize） ----
    def _decoder_gradient_importance(self, normalized_feature, target, original_power,
                                     group_symbols):
        """按 decoder 重建损失的一阶敏感度计算复符号组重要性。

        对每个复符号使用 ``|z| * |d(MSE)/dz|`` 作为其被擦除时的近似损失；
        再按连续 symbol group 求均值。该值比 latent magnitude 更直接对应重建质量。
        """
        torch = self.torch
        group_symbols = int(group_symbols)
        if group_symbols <= 0:
            raise ValueError('sensitivity_group_symbols 必须为正整数')
        # decode() 会恢复 sqrt(original_power)，这里保持完全相同的 decoder 输入尺度。
        # pwr may be a CUDA scalar. Extract its value explicitly instead of
        # asking NumPy to convert a GPU tensor to an array.
        if torch.is_tensor(original_power):
            power_value = float(original_power.detach().cpu().item())
        else:
            power_value = float(original_power)
        if not np.isfinite(power_value) or power_value < 0:
            raise ValueError('original_power 必须为有限非负标量')
        with torch.enable_grad():
            variable = normalized_feature.detach().float().clone().requires_grad_(True)
            # Keep the scale operation in the gradient context as well, so
            # callers using no_grad() do not disconnect variable from loss.
            restored = variable * power_value ** 0.5
            reconstruction = self.decoder(restored, self.snr, self.model)
            loss = torch.mean((reconstruction - target.float()) ** 2)
            gradient, = torch.autograd.grad(loss, variable, retain_graph=False,
                                             create_graph=False, only_inputs=True)
        values = variable.detach().reshape(-1).cpu().numpy().astype(np.float64)
        grads = gradient.detach().reshape(-1).cpu().numpy().astype(np.float64)
        half = values.size // 2
        magnitude = np.hypot(values[:half], values[half:])
        derivative = np.hypot(grads[:half], grads[half:])
        symbol_scores = magnitude * derivative
        groups = [float(np.mean(symbol_scores[start:start + group_symbols]))
                  for start in range(0, symbol_scores.size, group_symbols)]
        return {
            'scheme': 'decoder_gradient_times_symbol_magnitude',
            'loss': 'image_mse',
            'group_symbols': group_symbols,
            'num_symbols': int(symbol_scores.size),
            'scores': groups,
        }

    def encode(self, image_path, image_size=256, power=1.0,
               sensitivity_group_symbols=None):
        torch = self.torch
        chw, orig_hwc = preprocess_image(image_path, self.trainset, image_size)
        _, _, H, W = chw.shape
        if self.trainset != 'CIFAR10' and (H % 128 or W % 128):
            raise ValueError('HR 输入尺寸需为 128 的整数倍，得到 {}x{}'.format(H, W))
        self._ensure_models(H, W)
        x = torch.from_numpy(chw).to(self.device)
        self.last_sensitivity_importance = None
        with torch.no_grad():
            # 可选 fp16 前向；输出立即转回 fp32，保证归一化与后续复符号构造精度不变。
            ctx = self._amp_ctx()
            with ctx:
                if self.model in ('SwinJSCC_w/_RA', 'SwinJSCC_w/_SAandRA'):
                    feature, mask = self.encoder(x, self.snr, self.rate, self.model)
                    mask_numel = int(mask.numel())
                else:
                    feature = self.encoder(x, self.snr, self.rate, self.model)
                    mask_numel = 0
            feature = feature.float()
            # 官方归一化：返回单位功率特征与原始复符号平均功率 pwr。
            norm_feature, pwr = self.channel.complex_normalize(feature, power)
        if sensitivity_group_symbols:
            self.last_sensitivity_importance = self._decoder_gradient_importance(
                norm_feature, x, pwr, sensitivity_group_symbols)
        f = norm_feature.detach().to('cpu').numpy().reshape(-1).astype(np.float64)
        L = f.shape[0]
        # 与官方 Channel.forward 一致：前一半作实部、后一半作虚部 -> 复符号。
        sym = (f[:L // 2] + 1j * f[L // 2:]).astype(np.complex64)
        cbr = compute_cbr(self.model, int(feature.numel()), int(x.numel()), self.rate, self.trainset)
        meta = {
            'source_image': os.path.abspath(image_path),
            'image_name': os.path.basename(image_path),
            'image_stem': os.path.splitext(os.path.basename(image_path))[0],
            'checkpoint': self.checkpoint,
            'model': self.model,
            'trainset': self.trainset,
            'model_size': self.model_size,
            'C': self.C,
            'rate': self.rate,
            'snr_db': self.snr,
            'input_shape': [int(v) for v in (1, 3, H, W)],
            'feature_shape': [int(v) for v in feature.shape],
            'mask_numel': mask_numel,
            'num_real_elements': int(feature.numel()),
            'num_complex_symbols': int(sym.shape[0]),
            'cbr': float(cbr),
            'target_power': float(power),
            'original_symbol_power': float(pwr),
            'dtype': 'complex64',
            'byte_order': 'little (numpy 默认；complex64 = 8 字节/符号: float32 I + float32 Q)',
            'layout': 'interleaved I/Q, complex64',
        }
        return sym, meta, orig_hwc

    # ---- 解码：复符号 -> 重建图 ----
    def decode(self, sym, feature_shape, input_hw, orig_power=None, restore_power=True):
        torch = self.torch
        H, W = int(input_hw[0]), int(input_hw[1])
        self._ensure_models(H, W)
        feat = symbols_to_feature(sym, feature_shape, orig_power=orig_power,
                                  restore_power=restore_power)
        x = torch.from_numpy(feat).to(self.device)
        with torch.no_grad():
            ctx = self._amp_ctx()
            with ctx:
                recon = self.decoder(x, self.snr, self.model)
        recon_np = np.clip(recon.float().detach().to('cpu').numpy(), 0.0, 1.0)
        return np.transpose(recon_np[0], (1, 2, 0)) * 255.0


# ---------------------------------------------------------------------------
# 纯 numpy 自检（无需 torch）
# ---------------------------------------------------------------------------
def _self_test():
    rng = np.random.default_rng(0)

    # 复符号 <-> 特征 往返（含功率恢复）
    feat = rng.standard_normal((1, 256, 96)).astype(np.float32)
    L = feat.size
    sym = (feat.reshape(-1)[:L // 2] + 1j * feat.reshape(-1)[L // 2:]).astype(np.complex64)
    pwr = float(np.mean(np.abs(sym) ** 2))
    back = symbols_to_feature(sym, list(feat.shape), orig_power=pwr, restore_power=True)
    # sym 由未归一化 feat 直接构造，解码端乘 sqrt(pwr) 还原，故参考值应为 feat*sqrt(pwr)。
    err = float(np.max(np.abs(back - feat * np.sqrt(pwr))))
    assert err < 1e-4, 'real/imag 拼接往返失败'
    print('复符号 {} 个, 原始平均功率 {:.6f}, 往返误差 {:.2e}'.format(sym.shape[0], pwr, err))

    # 归一化一致性（若 torch 可用，验证官方 complex_normalize 得到单位功率）
    try:
        import torch
        ch = make_official_channel('cpu')
        norm_f, p = ch.complex_normalize(torch.from_numpy(feat), 1.0)
        unit = float(torch.mean(torch.abs(norm_f.reshape(-1)[:L // 2]
                                          + 1j * norm_f.reshape(-1)[L // 2:]) ** 2))
        print('官方 Channel.complex_normalize 单位功率 = {:.6f}'.format(unit))
    except Exception as e:  # noqa: BLE001
        print('(跳过官方 Channel 校验：{})'.format(e))

    packed, nbits = qpsk_pack(sym)
    print('QPSK: {} 符号 -> {} bit -> {} 字节'.format(sym.shape[0], nbits, packed.nbytes))

    a = rng.integers(0, 255, (32, 32, 3)).astype(np.float32)
    psnr, mse = compute_psnr_mse(a, a)
    print('PSNR(自比) = {}  MSE = {}'.format(psnr, mse))
    print('[OK] swinjscc_codec 纯 numpy 自检通过。')


if __name__ == '__main__':
    if '--self-test' in sys.argv or len(sys.argv) == 1:
        _self_test()
    else:
        print('用法: python swinjscc_codec.py --self-test')
