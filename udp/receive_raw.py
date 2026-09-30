#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""UDP receiver for raw latent files, with low-overhead live progress logs."""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
import os
import re
import socket
import struct
import tempfile
import time

H = b'RWH1'
D = b'RWD1'
M = b'RWM1'
W = (1 << 16) - 1


def atomic(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.tmp_', dir=os.path.dirname(path))
    with os.fdopen(fd, 'wb') as handle:
        handle.write(data)
        handle.flush()
    os.replace(temporary, path)


def decode_bitset(value, count):
    try:
        raw = base64.b64decode(value, validate=True)
    except Exception:
        return set()
    return {index for index in range(count)
            if index // 8 < len(raw) and raw[index // 8] & (1 << (index % 8))}


def output_name(name, grouped):
    if not name or name in ('.', '..') or any(c in name for c in ('/', '\\', ':', '\0')):
        raise ValueError('Unsafe incoming filename')
    if grouped:
        match = re.fullmatch(r'((?:energy|erasure|sensitivity|baseline)_\d{2,})__(.+)', name)
        if not match or match[2] in ('.', '..'):
            raise ValueError('Expected round-prefixed filename; sender must use --rounds > 1 or --alternate')
        return os.path.join(match[1], match[2])
    return name


def utc_time(value):
    return datetime.fromtimestamp(value, timezone.utc).isoformat() if value is not None else None


def transport_info(payload, fid, chunk_size, grouped):
    """New META can recover a session when all HELLO packets were lost."""
    try:
        info = json.loads(payload.decode('utf-8')).get('transport')
        if not isinstance(info, dict) or info.get('version') != 1:
            return None
        name, size = info['file_name'], info['file_bytes']
        if (type(info.get('file_id')) is not int or info['file_id'] != fid or
                type(info.get('chunk_bytes')) is not int or info['chunk_bytes'] != chunk_size or
                type(size) is not int or size <= 0 or size > 1 << 40 or
                not isinstance(name, str)):
            return None
        output_name(name, grouped)
        return name, size
    except (ValueError, KeyError, TypeError, UnicodeError, AttributeError):
        return None


class Session:
    def __init__(self, fid, name, size, chunk_size, peer, erasure_fill=0.0):
        self.f = fid
        self.n = name
        self.z = size
        self.ch = chunk_size
        self.peer = peer
        self.fill = erasure_fill
        self.t = (size + chunk_size - 1) // chunk_size if size else 0
        self.c = {}
        self.m = None
        self.last = time.time()
        self.started = self.last
        self.first_hello = None
        self.first_meta = None
        self.first_data = None
        self.last_data = None
        self.dup = 0
        self.bad = 0
        self.hello = 0
        self.mp = 0
        self.data_packets = 0
        self.data_bytes = 0

    def add(self, sequence, payload):
        self.data_packets += 1
        self.data_bytes += len(payload)
        if sequence < 0 or sequence >= self.t:
            self.bad += 1
            return
        expected = self.ch if sequence < self.t - 1 else self.z - (self.t - 1) * self.ch
        if not payload or len(payload) != expected:
            self.bad += 1
            return
        if sequence in self.c:
            self.dup += 1
            return
        self.c[sequence] = payload
        self.last = time.time()
        if self.first_data is None:
            self.first_data = self.last
        self.last_data = self.last

    def blob(self):
        if self.m and self.fill != 0:
            try:
                metadata = json.loads(self.m.decode())
            except (ValueError, UnicodeError):
                metadata = {}
            if metadata.get('storage', {}).get('format') == 'swinjscc_int8_v1' and self.fill != 0:
                raise ValueError('int8 requires --erasure-fill 0')
        result = []
        missing = 0
        zero = b'\0'
        # 预计算显式擦除占位符：complex64 = 实部/虚部各一个 float32，共 8 字节。
        marker = (struct.pack('<f', float(self.fill)) * 2) if self.fill != 0.0 else None
        for sequence in range(self.t):
            if sequence in self.c:
                result.append(self.c[sequence])
            else:
                length = self.ch if sequence < self.t - 1 else self.z - (self.t - 1) * self.ch
                length = max(0, length)
                if marker is None:
                    result.append(zero * length)
                else:
                    reps = (length + len(marker) - 1) // len(marker)
                    result.append((marker * reps)[:length])
                missing += 1
        return b''.join(result), missing


def semantic_stats(session):
    try:
        metadata = json.loads(session.m.decode()) if session.m else None
        config = metadata.get('semantic_uep') if isinstance(metadata, dict) else None
    except Exception:
        config = None
    if not isinstance(config, dict):
        return None

    count = min(session.t, int(config.get('total_chunks', session.t)))
    tier_a = decode_bitset(config.get('tier_a_bits', ''), count)
    tier_b = decode_bitset(config.get('tier_b_bits', ''), count)
    tier_c = set(range(count)) - tier_a - tier_b
    received = set(session.c)

    def percentage(tier):
        return None if not tier else 100 * len(tier & received) / len(tier)

    return {
        'scheme': config.get('scheme'),
        'metric': config.get('metric'),
        'tier_a_pct': percentage(tier_a),
        'tier_b_pct': percentage(tier_b),
        'tier_c_pct': percentage(tier_c),
        'tier_a_count': len(tier_a),
        'tier_b_count': len(tier_b),
        'tier_c_count': len(tier_c),
        'a_rep': config.get('a_rep'),
        'b_rep': config.get('b_rep'),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--grouped-runs', action='store_true',
                        help='Store method_NN__filename in method_NN/filename; use a NEW output root per batch')
    parser.add_argument('--meta-dir')
    parser.add_argument('--stats-dir')
    parser.add_argument('--port', type=int, default=60000,
                        help='UDP listening port (default: 60000)')
    parser.add_argument('--bind', default='0.0.0.0')
    parser.add_argument('--chunk', type=int, default=1200)
    parser.add_argument('--buf', type=int, default=65535)
    parser.add_argument('--rcvbuf', type=int, default=32 << 20)
    parser.add_argument('--quiet', type=float, default=5)
    parser.add_argument('--status-interval', type=float, default=1.0,
                        help='seconds between live status lines; 0 disables them (default: 1.0)')
    parser.add_argument('--no-stats', action='store_true')
    parser.add_argument('--erasure-fill', type=float, default=0.0,
                        help='显式擦除占位符取值；0.0 保持原生补零。非零占位符只有在'
                             '解码器经过擦除感知训练时才有望带来提升。')
    args = parser.parse_args()
    if args.status_interval < 0:
        parser.error('--status-interval must be 0 or greater')
    if args.grouped_runs and any(os.path.exists(p) for p in
                                (args.out_dir, args.meta_dir, args.stats_dir) if p):
        parser.error('--grouped-runs requires NEW output directories to prevent overwriting previous batches')

    meta_dir = args.meta_dir or args.out_dir
    stats_dir = args.stats_dir or args.out_dir
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)
    os.makedirs(stats_dir, exist_ok=True)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, args.rcvbuf)
    sock.bind((args.bind, args.port))
    sock.settimeout(.1)
    print(f'[监听] UDP {args.bind}:{args.port}  状态间隔={args.status_interval:g}s  静默落盘={args.quiet:g}s', flush=True)

    sessions = {}
    orphan_data = {}
    orphan_meta = {}
    completed = set()
    last_report_at = time.time()
    last_report_packets = 0
    total_data_packets = 0

    def flush(session, reason):
        blob, missing = session.blob()
        target_name = output_name(session.n, args.grouped_runs)
        atomic(os.path.join(args.out_dir, target_name), blob)
        if session.m:
            atomic(os.path.join(meta_dir, os.path.splitext(target_name)[0] + '.meta.json'), session.m)
        receive_pct = 100 * len(session.c) / session.t if session.t else 0
        tiers = semantic_stats(session)
        format_pct = lambda value: 'n/a' if value is None else f'{value:.1f}%'
        tier_text = (f' A={format_pct(tiers["tier_a_pct"])}'
                     f' B={format_pct(tiers["tier_b_pct"])}'
                     f' C={format_pct(tiers["tier_c_pct"])}') if tiers else ''
        print(f'[落盘] {session.n} {reason} {len(session.c)}/{session.t} '
              f'({receive_pct:.1f}%) missing={missing} dup={session.dup}{tier_text}', flush=True)
        if not args.no_stats:
            stats = {
                'file': session.n, 'fid': session.f, 'size': session.z,
                'total_chunks': session.t, 'unique_chunks': len(session.c),
                'chunk_bytes': session.ch,
                'missing_sequences': [q for q in range(session.t) if q not in session.c],
                'erasure_fill': session.fill,
                'receive_rate_pct': receive_pct, 'missing_chunks': missing,
                'packet_duplicates': session.dup, 'invalid_packets': session.bad,
                'hello_packets': session.hello, 'meta_packets': session.mp,
                'data_packets': session.data_packets, 'data_payload_bytes': session.data_bytes,
                'md5': hashlib.md5(blob).hexdigest(), 'peer': list(session.peer),
                'semantic': tiers,
                'timing_utc': {
                    'session_started': utc_time(session.started),
                    'first_hello': utc_time(session.first_hello),
                    'first_meta': utc_time(session.first_meta),
                    'first_data': utc_time(session.first_data),
                    'last_data': utc_time(session.last_data),
                    'saved': utc_time(time.time()),
                },
            }
            atomic(os.path.join(stats_dir, os.path.splitext(target_name)[0] + '.recv.json'),
                   json.dumps(stats, ensure_ascii=False, indent=2).encode())
        completed.add((session.n, session.z))

    def report(now):
        nonlocal last_report_at, last_report_packets
        if not args.status_interval or now - last_report_at < args.status_interval:
            return
        elapsed = max(now - last_report_at, 1e-9)
        packet_rate = (total_data_packets - last_report_packets) / elapsed
        last_report_at = now
        last_report_packets = total_data_packets
        if not sessions:
            return
        current = max(sessions.values(), key=lambda item: item.last)
        unique = len(current.c)
        receive_pct = 100 * unique / current.t if current.t else 0
        print(f'[RX] DATA总包={total_data_packets}  当前={current.n}  '
              f'唯一chunk={unique}/{current.t} ({receive_pct:.1f}%)  '
              f'重复={current.dup}  速率={packet_rate:.1f} pkt/s', flush=True)

    try:
        while True:
            try:
                datagram, address = sock.recvfrom(args.buf)
            except socket.timeout:
                datagram = None
                address = None
            now = time.time()
            for fid, session in list(sessions.items()):
                if now - session.last >= args.quiet:
                    flush(session, 'QUIET')
                    sessions.pop(fid, None)
            report(now)
            if datagram is None:
                continue

            if datagram.startswith(H):
                if len(datagram) < 16:
                    continue
                fid = int.from_bytes(datagram[4:6], 'big')
                size = int.from_bytes(datagram[6:14], 'big')
                name_length = int.from_bytes(datagram[14:16], 'big')
                if 16 + name_length > len(datagram):
                    continue
                name = datagram[16:16 + name_length].decode('utf8', 'ignore')
                try:
                    output_name(name, args.grouped_runs)
                except ValueError as error:
                    print(f'[忽略 HELLO] {error}', flush=True)
                    continue
                if (name, size) in completed:
                    continue
                # HELLO is deliberately repeated by the sender as a control
                # packet.  Re-creating the Session here would discard every
                # DATA chunk received before the repeated HELLO.
                existing = sessions.get(fid)
                if existing and existing.n == name and existing.z == size:
                    existing.hello += 1
                    if existing.first_hello is None:
                        existing.first_hello = now
                    existing.last = now
                    continue

                if existing:
                    print(f'[忽略 HELLO] fid={fid} 与现有会话不一致', flush=True)
                    continue

                session = Session(fid, name, size, args.chunk, address, args.erasure_fill)
                session.hello = 1
                session.first_hello = now
                for sequence, payload in orphan_data.pop(fid, {}).items():
                    session.add(sequence, payload)
                session.m = orphan_meta.pop(fid, None)
                session.mp = 1 if session.m else 0
                if session.m:
                    session.first_meta = now
                sessions[fid] = session
                print(f'[RX] HELLO fid={fid} 文件={name} size={size} chunks={session.t} '
                      f'backfill={len(session.c)} peer={address[0]}:{address[1]}', flush=True)
                continue

            if datagram.startswith(D) and len(datagram) >= 10:
                fid = int.from_bytes(datagram[4:6], 'big')
                sequence = int.from_bytes(datagram[6:10], 'big')
                if fid == W:
                    continue
                payload = datagram[10:]
                total_data_packets += 1
                session = sessions.get(fid)
                if session:
                    session.add(sequence, payload)
                else:
                    orphan_data.setdefault(fid, {})[sequence] = payload
                continue

            if datagram.startswith(M) and len(datagram) >= 10:
                fid = int.from_bytes(datagram[4:6], 'big')
                size = int.from_bytes(datagram[6:10], 'big')
                payload = datagram[10:10 + size]
                if len(payload) != size:
                    continue
                session = sessions.get(fid)
                identity = transport_info(payload, fid, args.chunk, args.grouped_runs)
                if session:
                    if identity and identity != (session.n, session.z):
                        print(f'[忽略 META] fid={fid} 与现有会话不一致', flush=True)
                        continue
                    first_meta = session.m is None
                    session.m = payload
                    session.mp += 1
                    if session.first_meta is None:
                        session.first_meta = now
                    session.last = now
                    if first_meta:
                        print(f'[RX] META fid={fid} 文件={session.n} bytes={size}', flush=True)
                else:
                    if identity and identity not in completed:
                        name, file_size = identity
                        session = Session(fid, name, file_size, args.chunk, address, args.erasure_fill)
                        session.m = payload
                        session.mp = 1
                        session.first_meta = now
                        for sequence, chunk in orphan_data.pop(fid, {}).items():
                            session.add(sequence, chunk)
                        sessions[fid] = session
                        print(f'[RX] META建档 fid={fid} 文件={name} chunks={session.t} '
                              f'backfill={len(session.c)}（HELLO尚未收到）', flush=True)
                    elif not identity:
                        first_orphan = fid not in orphan_meta
                        orphan_meta[fid] = payload
                        if first_orphan:
                            print(f'[RX] META fid={fid} bytes={size}（等待 HELLO）', flush=True)
    except KeyboardInterrupt:
        for session in list(sessions.values()):
            try:
                flush(session, 'CTRL-C')
            except Exception as exc:
                print('[warn]', exc, flush=True)
    finally:
        sock.close()


if __name__ == '__main__':
    main()
