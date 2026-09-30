"""CPU integration tests: packetization, erasures, META fallback and decoding input."""
import contextlib
import base64
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from USRP.erasure_training import measure_erasure_importance as measure
from USRP.core_codec.latent_storage import quantize, load_int8
from USRP.offline_comparison.evaluate_latent_quantization import pack, unpack
from USRP.erasure_training.measure_erasure_importance import erased_symbols
from USRP.gradient_proxy.score_gradient_importance import score_chunks
from USRP.gradient_proxy.validate_gradient_importance import rank_report
from USRP.core_codec.swinjscc_codec import embed_meta_into_bytes
from USRP.core_codec.decode_dataset import process_one
from udp import send_raw as tx
from udp import receive_raw as rx


class Socket:
    def __init__(self): self.packets = []
    def setsockopt(self, *args): pass
    def bind(self, *args): pass
    def settimeout(self, *args): pass
    def close(self): pass
    def sendto(self, packet, address): self.packets.append(packet)
    def recvfrom(self, size):
        if not self.packets: raise KeyboardInterrupt
        return self.packets.pop(0), ('127.0.0.1', 50000)


class Int8TransportTests(unittest.TestCase):
    def test_project_root_after_directory_classification(self):
        from USRP.core_codec import swinjscc_codec, swinjscc_config
        from USRP.predictor import erasure_predictor

        for module in (swinjscc_codec, swinjscc_config, erasure_predictor):
            self.assertEqual(module.PROJECT_ROOT, PROJECT_ROOT)
        self.assertTrue((PROJECT_ROOT / 'main.py').is_file())
        self.assertTrue((PROJECT_ROOT / 'net').is_dir())
        self.assertTrue((PROJECT_ROOT / 'udp' / 'send_raw.py').is_file())

    def setUp(self):
        rng = np.random.RandomState(2026)
        self.source = (rng.randn(2000)+1j*rng.randn(2000)).astype(np.complex64)
        self.meta = dict(image_stem='example', model='SwinJSCC_w/o_SAandRA', C=96,
                         snr_db=10, num_complex_symbols=2000, feature_shape=[1, 4000],
                         input_shape=[1, 3, 16, 16], original_symbol_power=2.)
        array, self.compact = quantize(self.source, self.meta)
        buf = io.BytesIO(); np.save(buf, array, allow_pickle=False)
        self.raw = embed_meta_into_bytes(buf.getvalue(), self.compact)
        self.clean = load_int8(self.raw, self.compact)

    def test_matches_evaluated_quantizer(self):
        expected, _ = unpack(pack(self.source, self.meta, 'int8')[0])
        np.testing.assert_array_equal(self.clean, expected)
        self.assertEqual(np.load(io.BytesIO(self.raw), allow_pickle=False).dtype, np.int8)

    def test_energy_auto_detects_int8_and_protects_header(self):
        chunk = 1200
        values = np.load(io.BytesIO(tx.split(self.raw)[0]), allow_pickle=False).reshape(-1)
        head = self.compact['storage']['header_bytes']
        actual, metric = tx.energy_scores(self.raw, chunk)
        expected = []
        for q in range((len(self.raw)+chunk-1)//chunk):
            lo = max(q*chunk, head)
            hi = min((q+1)*chunk, head+values.nbytes)
            part = values[lo-head:hi-head].astype(np.float64)
            expected.append(float(np.mean(part*part)) if part.size else -1e30)
        self.assertEqual(actual, expected)
        self.assertEqual(metric, 'int8_mean_square_proxy_v1')
        self.assertEqual(tx.energy_scores(self.raw, chunk, 'int')[0], actual)
        with self.assertRaises(ValueError):
            tx.energy_scores(self.raw, chunk, 'float')
        _, repeats, _, tiers = tx.plan(self.raw, chunk, 0, 'semantic', .5, .1, 2, 1,
                                       importance='energy')
        selected = rx.decode_bitset(tiers['tier_a_bits'], len(actual))
        ranked = sorted(range(1, len(actual)), key=lambda q: (-actual[q], q))
        self.assertEqual(selected, {0, *ranked[:tiers['tier_a_count']-1]})
        self.assertEqual(set(repeats), selected)

    def test_gradient_scores_follow_interleaved_int8_byte_boundaries(self):
        chunk = 1200
        count = len(self.source)
        gradient = np.concatenate((np.ones(count), 2*np.ones(count)))
        scores, signed = score_chunks(self.raw, self.compact, gradient, chunk)
        body, _ = tx.split(self.raw)
        head = self.compact['storage']['header_bytes']
        values = np.frombuffer(body, dtype=np.int8, count=2*count, offset=head)
        scale = self.compact['storage']['scale'] * np.sqrt(self.compact['original_symbol_power'])
        weight = np.tile([1., 2.], count)
        expected = []
        for seq in range((len(self.raw)+chunk-1)//chunk):
            lo = max(head, seq*chunk)-head
            hi = min(head+2*count, (seq+1)*chunk)-head
            expected.append(float(np.sum(np.abs(values[lo:hi].astype(float)*weight[lo:hi]*scale))))
        np.testing.assert_allclose(scores, expected, rtol=1e-12)
        self.assertEqual(len(signed), len(scores))
        with self.assertRaises(ValueError):
            score_chunks(self.raw, self.compact, gradient[:-1], chunk)

    def test_gradient_sidecar_protects_header_and_rejects_stale_source(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'example.npy'; path.write_bytes(self.raw)
            count = (len(self.raw)+1199)//1200
            sidecar = dict(complete=True, provenance=dict(
                scheme='int8_decoder_gradient_chunk_proxy_v1',
                storage_format='swinjscc_int8_v1',
                source_sha256=hashlib.sha256(self.raw).hexdigest(),
                chunk_bytes=1200, total_chunks=count),
                scores=[float(i) for i in range(count)])
            path.with_suffix('.gradient.json').write_text(json.dumps(sidecar))
            scores, metric = tx.gradient_scores(self.raw, str(path), 1200)
            self.assertEqual(scores, sidecar['scores'])
            self.assertEqual(metric, sidecar['provenance']['scheme'])
            _, repeats, _, tiers = tx.plan(self.raw, 1200, 0, 'semantic', .5, .1, 2, 1,
                                           importance='gradient', path=str(path))
            chosen = rx.decode_bitset(tiers['tier_a_bits'], count)
            self.assertIn(0, chosen)
            self.assertEqual(set(repeats), chosen)
            with self.assertRaises(ValueError):
                tx.gradient_scores(self.raw+b'x', str(path), 1200)
            with self.assertRaises(ValueError):
                tx.gradient_scores(self.raw, str(path), 600)
            delivered = Socket()
            argv = ['send', '--dir', temp, '--host', '127.0.0.1', '--mode', 'semantic',
                    '--importance', 'gradient', '--channel-profile', 'medium',
                    '--gap', '0', '--warmup', '0', '--once']
            with patch.object(sys, 'argv', argv), patch.object(tx.socket, 'socket', return_value=delivered), contextlib.redirect_stdout(io.StringIO()):
                tx.main()
            output = Path(temp)/'received'
            argv = ['receive', '--out-dir', str(output), '--status-interval', '0']
            with patch.object(sys, 'argv', argv), patch.object(rx.socket, 'socket', return_value=delivered), contextlib.redirect_stdout(io.StringIO()):
                rx.main()
            self.assertEqual((output/'example.npy').read_bytes(), self.raw)
            stats = json.loads((output/'example.recv.json').read_text())
            self.assertEqual(stats['semantic']['metric'], 'int8_decoder_gradient_chunk_proxy_v1')
            sock = Socket()
            path.write_bytes(self.raw+b'x')
            argv = ['send', '--dir', temp, '--host', '127.0.0.1', '--mode', 'semantic',
                    '--importance', 'gradient', '--once']
            with patch.object(sys, 'argv', argv), patch.object(tx.socket, 'socket', return_value=sock):
                with self.assertRaises(ValueError):
                    tx.main()
            self.assertEqual(sock.packets, [])

    def test_gradient_rank_report_excludes_forced_header(self):
        actual = [100, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        report = rank_report(actual, actual, .2)
        self.assertEqual(report['top_k'], 1)
        self.assertEqual(report['top_overlap'], 1)
        self.assertEqual(report['top_recall'], 1.0)
        self.assertAlmostEqual(report['spearman'], 1.0)

    def test_int8_energy_sends_without_erasure_sidecar(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); source = root/'source'; source.mkdir()
            (source/'example.npy').write_bytes(self.raw)
            sock = Socket()
            argv = ['send', '--dir', str(source), '--host', '127.0.0.1', '--mode', 'semantic',
                    '--importance', 'energy', '--channel-profile', 'medium', '--gap', '0',
                    '--warmup', '0', '--once']
            with patch.object(sys, 'argv', argv), patch.object(tx.socket, 'socket', return_value=sock), contextlib.redirect_stdout(io.StringIO()):
                tx.main()
            count = (len(self.raw)+1199)//1200
            metas = [p for p in sock.packets if p.startswith(tx.M)]
            self.assertTrue(metas)
            metadata = json.loads(metas[0][10:])
            self.assertEqual(metadata['semantic_uep']['metric'], 'int8_mean_square_proxy_v1')
            self.assertEqual(metadata['semantic_uep']['tier_a_count'], max(1, round(count*.1)))
            self.assertIn(0, rx.decode_bitset(metadata['semantic_uep']['tier_a_bits'], count))
            self.assertEqual(sum(p.startswith(tx.D) for p in sock.packets),
                             count+metadata['semantic_uep']['tier_a_count'])
            received = root/'received'
            argv = ['receive', '--out-dir', str(received), '--status-interval', '0']
            with patch.object(sys, 'argv', argv), patch.object(rx.socket, 'socket', return_value=sock), contextlib.redirect_stdout(io.StringIO()):
                rx.main()
            self.assertEqual((received/'example.npy').read_bytes(), self.raw)
            stats = json.loads((received/'example.recv.json').read_text())
            self.assertEqual(stats['semantic']['metric'], 'int8_mean_square_proxy_v1')
            self.assertEqual(stats['missing_sequences'], [])

    def test_explicit_wrong_energy_dtype_stops_before_warmup(self):
        with tempfile.TemporaryDirectory() as temp:
            (Path(temp)/'example.npy').write_bytes(self.raw)
            sock = Socket()
            argv = ['send', '--dir', temp, '--host', '127.0.0.1', '--mode', 'semantic',
                    '--importance', 'energy', '--energy-dtype', 'float', '--once']
            with patch.object(sys, 'argv', argv), patch.object(tx.socket, 'socket', return_value=sock):
                with self.assertRaises(ValueError):
                    tx.main()
            self.assertEqual(sock.packets, [])

    def test_int8_energy_alternates_with_erasure(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp)
            path = source/'example.npy'; path.write_bytes(self.raw)
            count = (len(self.raw)+1199)//1200
            sidecar = dict(complete=True, provenance=dict(scheme='single_chunk_erasure_delta_mse_v1',
                           source_sha256=hashlib.sha256(self.raw).hexdigest(), chunk_bytes=1200,
                           total_chunks=count), scores=list(range(count)))
            path.with_suffix('.erasure.json').write_text(json.dumps(sidecar))
            sock = Socket()
            argv = ['send', '--dir', str(source), '--host', '127.0.0.1', '--mode', 'semantic',
                    '--importance', 'energy', '--alternate', '--rounds', '1',
                    '--round-gap', '0', '--channel-profile', 'medium', '--gap', '0',
                    '--warmup', '0', '--once']
            with patch.object(sys, 'argv', argv), patch.object(tx.socket, 'socket', return_value=sock), contextlib.redirect_stdout(io.StringIO()):
                tx.main()
            names = list(dict.fromkeys(packet[16:].decode() for packet in sock.packets
                                       if packet.startswith(tx.H)))
            self.assertEqual(names, ['energy_01__example.npy',
                                     'erasure_01__example.npy'])
            metrics = [json.loads(packet[10:])['semantic_uep']['metric']
                       for packet in sock.packets if packet.startswith(tx.M)]
            self.assertEqual(list(dict.fromkeys(metrics)),
                             ['int8_mean_square_proxy_v1',
                              'single_chunk_erasure_delta_mse_v1'])
            received = source/'received'
            argv = ['receive', '--out-dir', str(received), '--grouped-runs',
                    '--status-interval', '0']
            with patch.object(sys, 'argv', argv), patch.object(rx.socket, 'socket', return_value=sock), contextlib.redirect_stdout(io.StringIO()):
                rx.main()
            for method, metric in [('energy', 'int8_mean_square_proxy_v1'),
                                   ('erasure', 'single_chunk_erasure_delta_mse_v1')]:
                folder = received/f'{method}_01'
                self.assertEqual((folder/'example.npy').read_bytes(), self.raw)
                stats = json.loads((folder/'example.recv.json').read_text())
                self.assertEqual(stats['semantic']['metric'], metric)

    def test_large_int8_control_meta_keeps_decode_and_tier_fields(self):
        name = 'CLIC2022_Test_' + 'a' * 64 + '.npy'
        bitset = base64.b64encode(bytes((1312 + 7) // 8)).decode()
        full = dict(self.compact)
        full.update(image_stem='a' * 64, image_name='a' * 64 + '.png',
                    npy_file=name, output_stem=name[:-4],
                    transport=dict(version=1, file_id=0, file_name=name,
                                   file_bytes=len(self.raw), chunk_bytes=1200),
                    semantic_uep=dict(scheme='semantic', metric='single_chunk_erasure_delta_mse_v1',
                                      metric_note='measured single DATA chunk erasure delta MSE; not multi-loss optimum',
                                      total_chunks=1312, top_fraction=.1, mid_fraction=.1,
                                      a_rep=2, b_rep=1, tier_a_bits=bitset, tier_b_bits=bitset,
                                      tier_a_count=131, tier_b_count=131, tier_c_count=1050,
                                      header_protected=True, channel_profile='medium'))
        self.assertGreater(len(json.dumps(full, separators=(',', ':')).encode()), 1400)
        compact = tx.compact_int8_meta(full)
        self.assertLessEqual(len(json.dumps(compact, separators=(',', ':')).encode()), 1400)
        self.assertEqual(compact['semantic_uep']['tier_a_bits'], bitset)
        self.assertEqual(compact['semantic_uep']['tier_b_bits'], bitset)
        self.assertEqual(rx.transport_info(json.dumps(compact).encode(), 0, 1200, False),
                         (name, len(self.raw)))
        np.testing.assert_array_equal(load_int8(self.raw, compact), self.clean)

    def test_each_erased_chunk_matches_measurement(self):
        chunk = 1200
        head = self.compact['storage']['header_bytes']
        count = (len(self.raw)+chunk-1)//chunk
        for lost in range(count):
            session = rx.Session(0, 'example.npy', len(self.raw), chunk, ('local', 50000))
            for q in range(count):
                if q != lost: session.add(q, self.raw[q*chunk:(q+1)*chunk])
            blob, missing = session.blob()
            self.assertEqual(len(blob), len(self.raw))
            self.assertEqual(missing, 1)
            actual = load_int8(blob, self.compact)
            expected = erased_symbols(self.clean, head, chunk, lost, bytes_per_symbol=2)
            np.testing.assert_array_equal(actual, expected)

    def test_short_packet_is_missing_not_shifted(self):
        session = rx.Session(0, 'x.npy', 2405, 1200, ('local', 50000))
        session.add(0, b'x'*1199)
        session.add(1, b'y'*1200)
        session.add(2, b'z'*5)
        blob, missing = session.blob()
        self.assertEqual(blob, b'\0'*1200+b'y'*1200+b'z'*5)
        self.assertEqual(missing, 1)

    def test_full_sender_receiver_and_metadata_fallback(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); source = root/'source'; source.mkdir()
            path = source/'example.npy'; path.write_bytes(self.raw)
            count = (len(self.raw)+1199)//1200
            score = dict(complete=True, provenance=dict(scheme='single_chunk_erasure_delta_mse_v1',
                         source_sha256=hashlib.sha256(self.raw).hexdigest(), chunk_bytes=1200,
                         total_chunks=count), scores=list(range(count)))
            path.with_suffix('.erasure.json').write_text(json.dumps(score))
            sock = Socket()
            args = ['send', '--dir', str(source), '--host', '127.0.0.1', '--mode', 'semantic',
                    '--importance', 'erasure', '--channel-profile', 'medium', '--gap', '0',
                    '--warmup', '0', '--once']
            with patch.object(sys, 'argv', args), patch.object(tx.socket, 'socket', return_value=sock), contextlib.redirect_stdout(io.StringIO()):
                tx.main()
            metas = [p for p in sock.packets if p.startswith(tx.M)]
            self.assertTrue(metas)
            self.assertLessEqual(max(len(p)-10 for p in metas), 1400)
            lost = {0, count-1}  # Lose both NPY header and embedded META; standalone META survives.
            sock.packets = [p for p in sock.packets if not (p.startswith(tx.D) and int.from_bytes(p[6:10], 'big') in lost)]
            received = root/'received'
            args = ['receive', '--out-dir', str(received), '--status-interval', '0']
            with patch.object(sys, 'argv', args), patch.object(rx.socket, 'socket', return_value=sock), contextlib.redirect_stdout(io.StringIO()):
                rx.main()
            stats = json.loads((received/'example.recv.json').read_text())
            self.assertEqual(stats['missing_sequences'], sorted(lost))
            self.assertIsNotNone(stats['semantic'])
            expected = self.clean.copy()
            for q in lost:
                expected = erased_symbols(expected, self.compact['storage']['header_bytes'], 1200, q, bytes_per_symbol=2)
            class Codec:
                def decode(self, symbols, *args, **kwargs):
                    np.testing.assert_array_equal(symbols, expected)
                    return np.zeros((16, 16, 3), np.float32)
            output = root/'recon'; output.mkdir()
            args = SimpleNamespace(meta_dir=None, no_restore_power=False, out_dir=str(output),
                                   ref_dir=None, image_size=None, ssim=False)
            rec, _ = process_one(str(received/'example.npy'), args, Codec())
            self.assertEqual(rec['storage_format'], 'swinjscc_int8_v1')
            self.assertTrue(rec['meta_file'].endswith('.meta.json'))

    def test_int8_rejects_unaligned_sensitivity_and_nonzero_fill(self):
        with self.assertRaises(ValueError):
            tx.plan(self.raw, 1200, 0, 'semantic', .1, .1, 2, 1, importance='sensitivity')
        self.assertEqual(tx.plan(self.raw, 1200, 0, 'semantic', .1, .1, 2, 1,
                                 importance='energy')[3]['metric'], 'int8_mean_square_proxy_v1')
        session = rx.Session(0, 'x.npy', len(self.raw), 1200, ('local', 50000), -1)
        session.m = json.dumps(self.compact).encode()
        with self.assertRaises(ValueError): session.blob()

    def test_energy_auto_supports_other_numeric_precisions(self):
        for dtype in (np.float16, np.float32, np.float64, np.complex64,
                      np.complex128, np.uint8, np.int16, np.int32):
            with self.subTest(dtype=dtype):
                values = np.arange(1000, dtype=np.float64) / 100
                if np.issubdtype(dtype, np.integer):
                    values = np.arange(1000) % 100
                if np.issubdtype(dtype, np.complexfloating):
                    values = values + 1j*values
                values = values.astype(dtype)
                buffer = io.BytesIO(); np.save(buffer, values, allow_pickle=False)
                raw = buffer.getvalue()
                actual, metric = tx.energy_scores(raw, 1200)
                self.assertEqual(tx.scores(raw, 1200), actual)
                self.assertEqual(len(actual), (len(raw)+1199)//1200)
                self.assertTrue(all(np.isfinite(x) for x in actual))
                if np.issubdtype(dtype, np.integer):
                    self.assertEqual(metric, 'integer_mean_square_proxy_v1')
                    self.assertEqual(tx.energy_scores(raw, 1200, 'int')[0], actual)
                else:
                    self.assertEqual(metric, 'mean_log1p_abs2_proxy')
                    self.assertEqual(tx.energy_scores(raw, 1200, 'float')[0], actual)
                    legacy = np.log1p(np.abs(values).astype(np.float64)**2)
                    head = len(raw)-values.nbytes
                    expected = []
                    for q in range(len(actual)):
                        lo = max(q*1200, head)
                        hi = min((q+1)*1200, len(raw))
                        first = max(0, (lo-head)//values.itemsize)
                        last = min(values.size, (hi-head)//values.itemsize)
                        expected.append(float(legacy[first:last].mean()) if last>first else -1e30)
                    self.assertEqual(actual, expected)
        fortran = np.asfortranarray(np.arange(600, dtype=np.float32).reshape(20, 30))
        buffer = io.BytesIO(); np.save(buffer, fortran, allow_pickle=False)
        scored, _ = tx.energy_scores(buffer.getvalue(), 1200)
        self.assertGreater(scored[0], 0)
        buffer = io.BytesIO(); np.save(buffer, np.ones(8, dtype=np.bool_), allow_pickle=False)
        with self.assertRaises(ValueError):
            tx.energy_scores(buffer.getvalue(), 1200)

    def test_meta_recovers_all_lost_hello_and_late_hello(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); source = root/'source'; source.mkdir()
            (source/'example.npy').write_bytes(self.raw)
            sock = Socket()
            args = ['send', '--dir', str(source), '--host', '127.0.0.1',
                    '--mode', 'baseline', '--gap', '0', '--warmup', '0', '--once']
            with patch.object(sys, 'argv', args), patch.object(tx.socket, 'socket', return_value=sock), contextlib.redirect_stdout(io.StringIO()):
                tx.main()
            metadata = next(p[10:] for p in sock.packets if p.startswith(tx.M))
            self.assertEqual(rx.transport_info(metadata, 0, 1200, False), ('example.npy', len(self.raw)))
            self.assertIsNone(rx.transport_info(metadata, 1, 1200, False))
            original_hello = next(p for p in sock.packets if p.startswith(tx.H))
            sock.packets = [p for p in sock.packets if not p.startswith(tx.H)] + [original_hello]
            received = root/'received'
            args = ['receive', '--out-dir', str(received), '--status-interval', '0']
            with patch.object(sys, 'argv', args), patch.object(rx.socket, 'socket', return_value=sock), contextlib.redirect_stdout(io.StringIO()):
                rx.main()
            self.assertEqual((received/'example.npy').read_bytes(), self.raw)
            stats = json.loads((received/'example.recv.json').read_text())
            self.assertEqual(stats['hello_packets'], 1)
            self.assertGreater(stats['meta_packets'], 0)
            self.assertIsNotNone(stats['timing_utc']['first_meta'])
            self.assertIsNotNone(stats['timing_utc']['first_hello'])
            self.assertIsNotNone(stats['timing_utc']['first_data'])

    def test_measurement_uses_new_offsets_and_decoder_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'example.npy').write_bytes(self.raw)
            Image.fromarray(np.zeros((16, 16, 3), np.uint8)).save(root/'example.png')
            (root/'dummy.model').write_bytes(b'original')
            (root/'decoder.model').write_bytes(b'override')
            calls = []
            class Codec:
                def __init__(self, *args, **kwargs):
                    assert kwargs['decoder_checkpoint'] == str(root/'decoder.model')
                def decode(self, symbols, *args, **kwargs):
                    calls.append(symbols.copy())
                    return np.zeros((16, 16, 3), np.float64)
            argv = ['measure', '--npy-dir', str(root), '--ref-dir', str(root),
                    '--checkpoint', str(root/'dummy.model'), '--decoder-checkpoint', str(root/'decoder.model')]
            with patch.object(sys, 'argv', argv), patch.object(measure, 'SwinJSCCCodec', Codec), contextlib.redirect_stdout(io.StringIO()):
                measure.main()
            result = json.loads((root/'example.erasure.json').read_text())
            self.assertTrue(result['complete'])
            self.assertEqual(result['provenance']['bytes_per_symbol'], 2)
            self.assertEqual(result['provenance']['decoder_checkpoint_sha256'], hashlib.sha256(b'override').hexdigest())
            np.testing.assert_array_equal(calls[0], self.clean)
            expected = [self.clean]
            head = self.compact['storage']['header_bytes']
            for q in range((len(self.raw)+1199)//1200):
                if max(head, q*1200) < min(head+self.clean.size*2, (q+1)*1200):
                    expected.append(erased_symbols(self.clean, head, 1200, q, bytes_per_symbol=2))
            self.assertEqual(len(calls), len(expected))
            for actual, target in zip(calls, expected):
                np.testing.assert_array_equal(actual, target)


if __name__ == '__main__':
    unittest.main()
