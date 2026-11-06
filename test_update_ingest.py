"""Untrusted POST /update ingest after PR #12: forged torch.save storage metadata, Python workers' safetensors updates,
and the --pickle-updates switch (guarded by default; reject once no worker still sends pickle)."""
import http.client
import inspect
import io
import json
import pickletools
import struct
import unittest
import urllib.parse
import urllib.request
import zipfile
from unittest.mock import MagicMock, patch

import torch

import dbmoe
from test_dbmoe import tiny_lease
from test_safetensors_wire import start_coordinator

AUTH = {'Authorization': 'Bearer local-test-only'}
INT_OPS = {'BININT', 'BININT1', 'BININT2', 'LONG1'}


def tsave(obj):
    b = io.BytesIO()
    torch.save(obj, b)
    return b.getvalue()


def rezip(src, patch_pkl):
    """Rebuild a torch.save archive (stored entries, same names) with its data.pkl rewritten."""
    zin, out = zipfile.ZipFile(io.BytesIO(src)), io.BytesIO()
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_STORED) as z:
        for info in zin.infolist():
            data = zin.read(info.filename)
            z.writestr(zipfile.ZipInfo(info.filename), patch_pkl(data) if info.filename.endswith('/data.pkl') else data)
    return out.getvalue()


def forge(pkl, numel=None, size0=None):
    """Rewrite the first storage's declared element count and/or the first dimension of the tensor built on it.  The
    persistent-id tuple is followed by a memo op (BINPUT) before BINPERSID, so locate the tuple itself."""
    ops = list(pickletools.genops(pkl))
    k = next(i for i, op in enumerate(ops) if op[0].name == 'BINPERSID')
    t = max(i for i in range(k) if ops[i][0].name.startswith('TUPLE'))
    m = next(i for i in range(k + 1, len(ops)) if ops[i][0].name.startswith('TUPLE'))
    assert ops[t - 1][0].name in INT_OPS and ops[m - 1][0].name in INT_OPS, 'unexpected torch.save pickle layout'
    edits = []
    if numel is not None:
        edits.append((ops[t - 1][2], ops[t][2], numel))
    if size0 is not None:
        edits.append((ops[m - 1][2], ops[m][2], size0))
    for start, end, value in sorted(edits, reverse=True):
        pkl = pkl[:start] + b'J' + struct.pack('<i', value) + pkl[end:]
    return pkl


class ForgedStorageTests(unittest.TestCase):
    """These archives pass load_update_pickle's ZIP checks, so torch must refuse the forged metadata before any tensor
    data is read: no read past a record and no allocation beyond the upload."""
    CAP = 64 << 20

    def test_rebuilt_genuine_archive_still_loads(self):     # the rewriting harness itself is sound
        body = rezip(tsave({'x': torch.arange(4.)}), lambda p: p)
        self.assertTrue(torch.equal(dbmoe.load_update_pickle(body, self.CAP)['x'], torch.arange(4.)))

    def test_storage_declared_larger_than_its_record_is_refused(self):
        for size0 in (None, 50_000_000):
            with self.subTest(size0=size0):
                body = rezip(tsave({'x': torch.arange(4.)}), lambda p: forge(p, numel=50_000_000, size0=size0))
                with self.assertRaises(RuntimeError) as caught:
                    dbmoe.load_update_pickle(body, self.CAP)
                self.assertIn('size', str(caught.exception))

    def test_tensor_larger_than_its_storage_is_refused(self):
        body = rezip(tsave({'x': torch.zeros(1)}), lambda p: forge(p, size0=1_000_000))
        with self.assertRaises(RuntimeError) as caught:
            dbmoe.load_update_pickle(body, self.CAP)
        self.assertIn('size', str(caught.exception))


class PythonWorkerSafetensorsTests(unittest.TestCase):
    def test_python_update_roundtrip(self):
        up = dbmoe.train_lease(tiny_lease(), 'cpu')
        up.update(name='python-test', device='cpu', hardware={'backend': 'cpu'})
        got = dbmoe.update_from_safetensors(dbmoe.update_to_safetensors(up))
        for key in set(up) - {'d_shared', 'd_block'}:
            self.assertEqual(got[key], up[key])
        for group in ('d_shared', 'd_block'):
            self.assertEqual(set(got[group]), set(up[group]))
            for key in up[group]:
                self.assertTrue(torch.equal(got[group][key], up[group][key]))

    def test_update_http_header(self):
        response = MagicMock()
        response.__enter__.return_value.read.return_value = b'{"ok":true}'
        with patch.object(dbmoe.urllib.request, 'urlopen', return_value=response) as opened:
            self.assertEqual(dbmoe.post_update('http://example.invalid', 'test', b'normal-test-bytes'), {'ok': True})
        self.assertEqual(opened.call_args.args[0].get_header('Content-type'), dbmoe.ST_UPDATE_TYPE)

    def test_handler_decides_the_format_before_reading_the_body(self):
        source = inspect.getsource(dbmoe.serve)
        update = source[source.index("if p != 'update':"):]
        self.assertNotIn('from_bytes(body)', update)
        self.assertNotIn('torch.load', update)
        self.assertLess(update.index("C.pickle_policy == 'reject'"), update.index('self.rfile.read'))


def lease_for(url, opener, session):
    req = urllib.request.Request(url + f'/lease?name=python-test&steps=2&batch=1&session={session}', headers=AUTH)
    with opener.open(req, timeout=60) as response:
        return dbmoe.from_bytes(response.read())


def post_update(url, opener, body, content_type):
    req = urllib.request.Request(url + '/update', data=body, headers=dict(AUTH, **{'Content-Type': content_type}))
    with opener.open(req, timeout=60) as response:
        return json.load(response)


def update_formats(url, opener):
    with opener.open(url + '/status', timeout=5) as response:
        return json.load(response)['update_formats']


def trained(lease):
    up = dbmoe.train_lease(lease, 'cpu')
    up.update(name='python-test', device='cpu')
    return up


class PickleSwitchTests(unittest.TestCase):
    def test_guarded_default_accepts_both_formats_and_counts_them(self):
        url, opener, _ = start_coordinator(self)
        self.assertEqual(update_formats(url, opener),
                         dict(safetensors=0, pickle=0, pickle_last_age_s=None, pickle_policy='guarded'))
        old = trained(lease_for(url, opener, 'old-worker'))
        self.assertTrue(post_update(url, opener, dbmoe.to_bytes(old), 'application/octet-stream')['ok'])
        new = trained(lease_for(url, opener, 'new-worker'))
        self.assertTrue(post_update(url, opener, dbmoe.update_to_safetensors(new), dbmoe.ST_UPDATE_TYPE)['ok'])
        formats = update_formats(url, opener)
        self.assertEqual((formats['pickle'], formats['safetensors']), (1, 1))
        self.assertIsNotNone(formats['pickle_last_age_s'])

    def test_reject_refuses_pickle_before_reading_the_body(self):
        url, opener, _ = start_coordinator(self, extra=['--pickle-updates', 'reject'])
        lease = lease_for(url, opener, 'format-test')
        parsed = urllib.parse.urlsplit(url)
        for content_type in (None, 'application/octet-stream', 'application/json'):
            with self.subTest(content_type=content_type):
                conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=5)
                try:
                    conn.putrequest('POST', '/update')
                    conn.putheader('Authorization', AUTH['Authorization'])
                    conn.putheader('Content-Length', '32')
                    if content_type:
                        conn.putheader('Content-Type', content_type)
                    conn.endheaders()   # deliberately no body: the refusal must not wait for one
                    response = conn.getresponse()
                    self.assertEqual(response.status, 415)
                    self.assertIn('upgrade', json.loads(response.read())['error'])
                finally:
                    conn.close()
        body = dbmoe.update_to_safetensors(trained(lease))     # the lease survived; a safetensors update still merges
        self.assertTrue(post_update(url, opener, body, dbmoe.ST_UPDATE_TYPE + '; charset=binary')['ok'])
        formats = update_formats(url, opener)
        self.assertEqual((formats['safetensors'], formats['pickle'], formats['pickle_policy']), (1, 0, 'reject'))


if __name__ == '__main__':
    unittest.main()
