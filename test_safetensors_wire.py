"""The opt-in safetensors wire format used by workers without PyTorch (worker-rs): codec, lease and update mapping,
and a real coordinator accepting a safetensors update over loopback HTTP."""
import json
import os
from pathlib import Path
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import time
import tracemalloc
import unittest
import urllib.error
import urllib.request

import torch
import dbmoe

from test_dbmoe import tiny_lease


def forge(header, data=b''):
    """A safetensors buffer with a hand-written header (for malformed-input tests)."""
    h = header.encode() if isinstance(header, str) else header
    return struct.pack('<Q', len(h)) + h + data


class CodecTests(unittest.TestCase):
    def test_round_trip_keeps_values_dtypes_and_shapes(self):
        tensors = {
            'f32': torch.tensor([[1.0, -2.5, 3e-38], [float('inf'), -0.0, 7.0]]),
            'f64': torch.arange(5, dtype=torch.float64) / 3,
            'bf16': torch.tensor([1.5, -2.0], dtype=torch.bfloat16),
            'i64': torch.tensor([[-(2 ** 40), 3]]),
            'u8': torch.arange(256, dtype=torch.uint8).reshape(16, 16),
            'scalar': torch.tensor(4.25),
            'empty': torch.empty(0, 3),
            'strided': torch.arange(12.).reshape(3, 4).t(),
        }
        buf = dbmoe.st_dumps(tensors, {'k': 'v', 'n': 3})
        (hlen,) = struct.unpack('<Q', buf[:8])
        self.assertEqual(hlen % 8, 0)
        out, meta = dbmoe.st_loads(buf)
        self.assertEqual(meta, {'k': 'v', 'n': '3'})
        self.assertEqual(set(out), set(tensors))
        for k, t in tensors.items():
            self.assertEqual(out[k].dtype, t.dtype, k)
            self.assertEqual(out[k].shape, t.shape, k)
            self.assertTrue(torch.equal(out[k], t), k)
        self.assertEqual(str(out['f32'][1, 1].item()), '-0.0')

    def test_matches_the_reference_library_when_installed(self):
        try:
            from safetensors.torch import load, save
        except ImportError:
            self.skipTest('safetensors package not installed')
        tensors = {'a': torch.randn(3, 4), 'b': torch.arange(6, dtype=torch.uint8), 'c': torch.tensor(1.5)}
        theirs = load(dbmoe.st_dumps(tensors, {'x': 'y'}))
        mine, _ = dbmoe.st_loads(save(tensors, metadata={'x': 'y'}))
        for k, t in tensors.items():
            self.assertTrue(torch.equal(theirs[k], t), k)
            self.assertTrue(torch.equal(mine[k], t), k)

    def test_rejects_malformed_buffers(self):
        good = dbmoe.st_dumps({'x': torch.ones(2)})
        f32 = '{"x":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}'
        self.assertEqual(set(dbmoe.st_loads(forge(f32, bytes(8)))[0]), {'x'})
        bad = [
            good[:7],                                                   # shorter than the length field
            good[:-1],                                                  # truncated data
            good + b'\0',                                               # trailing byte
            struct.pack('<Q', 2 ** 63) + good[8:],                      # absurd header length
            forge(f32.replace('F32', 'F31'), bytes(8)),                 # unknown dtype
            forge(f32.replace('[2]', '[3]'), bytes(8)),                 # size does not match shape
            forge(f32.replace('[2]', '[-2]'), bytes(8)),                # negative dimension
            forge(f32.replace('[2]', '[true,2]'), bytes(8)),            # boolean dimension
            forge(f32.replace('[0,8]', '[4,12]'), bytes(12)),           # gap before the tensor
            forge('{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},'
                  '"y":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}', bytes(8)),     # overlap
            forge('{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},'
                  '"x":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}}', bytes(8)),     # duplicate name
            forge('{"x":{"dtype":"F32","shape":[2],"data_offsets":[0,8],"extra":1}}', bytes(8)),
            forge('{"__metadata__":{"k":1}}'),                          # non-string metadata
            forge('[1,2]'),                                             # header is not an object
            forge(b'{"x":\xff}'),                                       # not UTF-8
            forge('{"x":'),                                             # not JSON
        ]
        for i, buf in enumerate(bad):
            with self.subTest(i=i), self.assertRaises(ValueError):
                dbmoe.st_loads(buf)

    def test_checks_the_whole_layout_before_copying_anything(self):
        # 200 names over one 1 MiB span would be 200 MiB of copies if each were copied before the overlap check
        span = 1 << 20
        entries = ','.join(f'"t{i}":{{"dtype":"U8","shape":[{span}],"data_offsets":[0,{span}]}}' for i in range(200))
        body = forge('{' + entries + '}', bytes(span))
        tracemalloc.start()
        try:
            with self.assertRaises(ValueError):
                dbmoe.st_loads(body)
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        self.assertLess(peak, 4 * span)

    def test_caps_tensor_count_and_header_size(self):
        empties = ','.join(f'"t{i}":{{"dtype":"U8","shape":[0],"data_offsets":[0,0]}}' for i in range(4097))
        with self.assertRaises(ValueError):
            dbmoe.st_loads(forge('{' + empties + '}'))
        with self.assertRaises(ValueError):
            dbmoe.st_loads(forge('{"__metadata__":{"k":"' + 'x' * (2 << 20) + '"}}'))

    def test_refuses_unsupported_dtypes_on_write(self):
        with self.assertRaises(ValueError):
            dbmoe.st_dumps({'b': torch.tensor([True])})


class MappingTests(unittest.TestCase):
    def test_lease_carries_every_field(self):
        L = tiny_lease()
        L.update(version=dbmoe.VERSION, ttl=1800.0)
        tensors, meta = dbmoe.st_loads(dbmoe.lease_to_safetensors(L))
        fields = json.loads(meta['dbmoe.lease'])
        self.assertEqual(fields['cfg'], L['cfg'])
        self.assertEqual(fields['sigmas'], L['sigmas'])
        for k in ('lease', 'block', 'block_version', 'steps', 'batch', 'lr', 'ttl', 'version'):
            self.assertEqual(fields[k], L[k], k)
        self.assertTrue(torch.equal(tensors['data'], L['data']))
        for k, v in L['shared'].items():
            self.assertTrue(torch.equal(tensors['shared.' + k], v), k)
        for k, v in L['block_state'].items():
            self.assertTrue(torch.equal(tensors['block.' + k], v), k)
        self.assertEqual(len(tensors), 1 + len(L['shared']) + len(L['block_state']))

    def test_update_round_trip_matches_the_pickle_path(self):
        torch.manual_seed(0)
        up = dbmoe.train_lease(tiny_lease(), 'cpu')
        fields = {k: v for k, v in up.items() if k not in ('d_shared', 'd_block')}
        fields.update(name='rust-test', device='cpu')
        tensors = {f'd_shared.{k}': v for k, v in up['d_shared'].items()}
        tensors.update({f'd_block.{k}': v for k, v in up['d_block'].items()})
        got = dbmoe.update_from_safetensors(dbmoe.st_dumps(tensors, {'dbmoe.update': json.dumps(fields)}))
        ref = dbmoe.from_bytes(dbmoe.to_bytes(dict(up, name='rust-test', device='cpu')))
        self.assertEqual(set(got), set(ref))
        for k in ('lease', 'block', 'losses', 'seconds', 'tokens', 'fresh_optimizer', 'name', 'device'):
            self.assertEqual(got[k], ref[k], k)
        for group in ('d_shared', 'd_block'):
            self.assertEqual(set(got[group]), set(ref[group]))
            for k, t in ref[group].items():
                self.assertEqual(got[group][k].dtype, t.dtype)
                self.assertTrue(torch.equal(got[group][k], t), f'{group}.{k}')

    def test_update_refuses_stray_tensors_and_missing_fields(self):
        with self.assertRaises(ValueError):
            dbmoe.update_from_safetensors(dbmoe.st_dumps({'d_block.x': torch.ones(1)}))
        with self.assertRaises(ValueError):
            dbmoe.update_from_safetensors(dbmoe.st_dumps({'other.x': torch.ones(1)}, {'dbmoe.update': '{}'}))
        with self.assertRaises(ValueError):
            dbmoe.update_from_safetensors(dbmoe.st_dumps({'d_block': torch.ones(1)}, {'dbmoe.update': '{}'}))


def start_coordinator(test, extra=()):
    """A tiny coordinator bound to loopback with one token, in a temporary directory removed after the test (once the
    server has stopped, since Windows cannot delete its open log); returns (url, opener, directory)."""
    root = Path(tempfile.mkdtemp(prefix='dbmoe-st-test-'))
    test.addCleanup(shutil.rmtree, root, ignore_errors=True)
    script = str(Path(dbmoe.__file__).resolve())
    (root / 'training.txt').write_text('This is local synthetic training data only.\n' * 100)
    (root / 'tokens.json').write_text(json.dumps({'local-test-only': 'local'}))
    cfg = dict(d=16, heads=2, layers_per_block=1, blocks=2, experts=2, ffn=32, ctx=8)
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        port = s.getsockname()[1]
    url = f'http://127.0.0.1:{port}'
    log = (root / 'server.log').open('w+')
    server = subprocess.Popen([sys.executable, script, 'coord', '--dir', str(root / 'run'), '--host', '127.0.0.1',
                               '--port', str(port), '--tokens', str(root / 'tokens.json'), '--text-file',
                               str(root / 'training.txt'), '--cfg', json.dumps(cfg), '--steps', '2', '--batch', '1',
                               '--heldout', '2', '--eval-every', '0', '--snap-every', '0', '--eval-threads', '1',
                               '--no-gate-all', '--min-steps', '1', *extra], stdout=log, stderr=subprocess.STDOUT)

    def stop():
        server.terminate()
        try:
            server.wait(timeout=20)
        except subprocess.TimeoutExpired:
            server.kill()
        log.close()
    test.addCleanup(stop)                               # cleanups run last-in first-out: the server stops before rmtree
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + float(os.environ.get('DBMOE_TEST_START_TIMEOUT', 120))
    while True:
        try:
            with opener.open(url + '/status', timeout=1) as r:
                json.load(r)
            return url, opener, root
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if server.poll() is not None or time.monotonic() > deadline:
                log.seek(0)
                test.fail('local coordinator did not start: ' + log.read())
            time.sleep(.1)


class LoopbackTests(unittest.TestCase):
    def test_coordinator_serves_and_merges_safetensors(self):
        url, opener, _ = start_coordinator(self)
        auth = {'Authorization': 'Bearer local-test-only'}
        req = urllib.request.Request(url + '/lease?name=st-test&steps=2&batch=1&session=abc&fmt=safetensors', headers=auth)
        with opener.open(req, timeout=60) as r:
            self.assertEqual(r.headers['Content-Type'], dbmoe.ST_LEASE_TYPE)
            tensors, meta = dbmoe.st_loads(r.read())
        fields = json.loads(meta['dbmoe.lease'])
        L = dict(fields, data=tensors.pop('data'),
                 shared={k[7:]: v for k, v in tensors.items() if k.startswith('shared.')},
                 block_state={k[6:]: v for k, v in tensors.items() if k.startswith('block.')})
        self.assertEqual(L['data'].dtype, torch.uint8)
        self.assertEqual(tuple(L['data'].shape), (2, L['cfg']['ctx'] + 1))
        up = dbmoe.train_lease(L, 'cpu')
        meta = {k: v for k, v in up.items() if k not in ('d_shared', 'd_block')}
        meta.update(name='st-test', device='cpu')
        body = dbmoe.st_dumps({**{f'd_shared.{k}': v for k, v in up['d_shared'].items()},
                               **{f'd_block.{k}': v for k, v in up['d_block'].items()}},
                              {'dbmoe.update': json.dumps(meta)})
        # a malformed upload is refused and does not consume the lease
        bad = urllib.request.Request(url + '/update', data=body[:-4], method='POST',
                                     headers=dict(auth, **{'Content-Type': dbmoe.ST_UPDATE_TYPE}))
        with self.assertRaises(urllib.error.HTTPError) as e:
            opener.open(bad, timeout=60)
        self.assertEqual(e.exception.code, 400)
        good = urllib.request.Request(url + '/update', data=body, method='POST',
                                      headers=dict(auth, **{'Content-Type': dbmoe.ST_UPDATE_TYPE}))
        with opener.open(good, timeout=60) as r:
            reply = json.load(r)
        self.assertTrue(reply.get('ok'), reply)
        with opener.open(url + '/status', timeout=5) as r:
            status = json.load(r)
        self.assertEqual(status['updates'], 1)
        # the pickle format is still the default
        req = urllib.request.Request(url + '/lease?name=st-test&steps=2&batch=1&session=abc', headers=auth)
        with opener.open(req, timeout=60) as r:
            self.assertEqual(r.headers['Content-Type'], 'application/octet-stream')
            self.assertIn('block_state', dbmoe.from_bytes(r.read()))


if __name__ == '__main__':
    unittest.main()
