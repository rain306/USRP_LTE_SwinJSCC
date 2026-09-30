"""Five-round transport integration without real sockets or LTE transmission."""
import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import send_raw as tx
import receive_raw as rx


class Socket:
    def __init__(self):
        self.packets = []
    def setsockopt(self, *args): pass
    def bind(self, *args): pass
    def settimeout(self, *args): pass
    def close(self): pass
    def sendto(self, packet, address): self.packets.append(packet)
    def recvfrom(self, size):
        if not self.packets:
            raise KeyboardInterrupt
        return self.packets.pop(0), ('127.0.0.1', 50000)


with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
    root = Path(tmp)
    source = root/'source'; source.mkdir()
    originals = {'one.npy': b'first'*700, 'two.npy': b'second'*850}
    for name, content in originals.items():
        (source/name).write_bytes(content)
    sock = Socket()
    argv = ['send', '--dir', str(source), '--host', '127.0.0.1', '--mode', 'baseline',
            '--rounds', '5', '--round-gap', '0', '--gap', '0', '--warmup', '0', '--once']
    with patch.object(sys, 'argv', argv), patch.object(tx.socket, 'socket', return_value=sock), contextlib.redirect_stdout(io.StringIO()):
        tx.main()
    hellos = [p for p in sock.packets if p.startswith(tx.H)]
    assert len({int.from_bytes(p[4:6], 'big') for p in hellos}) == 10
    assert len({p[16:] for p in hellos}) == 10
    output = root/'received'
    argv = ['receive', '--out-dir', str(output), '--grouped-runs']
    with patch.object(sys, 'argv', argv), patch.object(rx.socket, 'socket', return_value=sock), contextlib.redirect_stdout(io.StringIO()):
        rx.main()
    for index in range(1, 6):
        folder = output/('baseline_%02d' % index)
        for name, content in originals.items():
            assert (folder/name).read_bytes() == content
            stats = json.loads((folder/Path(name).with_suffix('.recv.json')).read_text())
            assert stats['missing_chunks'] == 0
    for unsafe in ('../oops', 'erasure_01__..', 'erasure_01__C:bad', 'one.npy'):
        try:
            rx.output_name(unsafe, True)
        except ValueError:
            pass
        else:
            raise AssertionError('Unsafe or ungrouped name accepted')
    with patch.object(sys, 'argv', argv), contextlib.redirect_stderr(io.StringIO()):
        try:
            rx.main()
        except SystemExit as error:
            assert error.code != 0
        else:
            raise AssertionError('Existing output root must be rejected')
print('PASS: five rounds, unique IDs/names, exact reconstruction, separate stats, overwrite protection')

with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
    root = Path(tmp)
    (root/'test.npy').write_bytes(b'mock source')
    calls = []
    def capture(s, h, p, path, args, fid, wire_name=None):
        calls.append((args.importance, fid, wire_name))
    argv = ['send', '--dir', str(root), '--host', '127.0.0.1', '--mode', 'semantic',
            '--rounds', '5', '--alternate', '--round-gap', '0', '--warmup', '0', '--once']
    with patch.object(sys, 'argv', argv), patch.object(tx.socket, 'socket', return_value=Socket()), \
         patch.object(tx, 'send_file', side_effect=capture), patch.object(tx, 'erasure_scores') as preflight, \
         patch.object(tx, 'energy_scores') as energy_preflight, \
         contextlib.redirect_stdout(io.StringIO()):
        tx.main()
    assert preflight.call_count == 1
    assert energy_preflight.call_count == 1
    assert [m for m, _, _ in calls] == ['energy', 'erasure']*5
    assert [fid for _, fid, _ in calls] == list(range(10))
    assert [name for _, _, name in calls] == [
        '%s_%02d__test.npy' % (method, i) for i in range(1, 6) for method in ('energy', 'erasure')]
    assert all(rx.output_name(name, True) == str(Path(name.split('__')[0])/'test.npy') for _, _, name in calls)
print('PASS: five alternating pairs, sidecar preflight, unique IDs and receiver routing')
