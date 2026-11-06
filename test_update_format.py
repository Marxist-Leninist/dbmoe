"""Benign protocol compatibility tests for safetensors-only updates."""
import http.client
import inspect
import io
import json
import unittest
import urllib.error
import urllib.request
from unittest.mock import patch
import torch
import dbmoe
from test_dbmoe import tiny_lease
from test_safetensors_wire import start_coordinator


class UpdateFormatTests(unittest.TestCase):
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
        response = unittest.mock.MagicMock()
        response.__enter__.return_value.read.return_value = b'{"ok":true}'
        with patch.object(dbmoe.urllib.request, 'urlopen', return_value=response) as opened:
            self.assertEqual(dbmoe.post_update('http://example.invalid', 'test', b'normal-test-bytes'), {'ok': True})
        request = opened.call_args.args[0]
        self.assertEqual(request.get_header('Content-type'), dbmoe.ST_UPDATE_TYPE)

    def test_update_handler_has_no_pickle_dispatch(self):
        source = inspect.getsource(dbmoe.serve)
        update = source[source.index("if p != 'update':"):]
        self.assertNotIn('from_bytes(body)', update)
        self.assertNotIn('torch.load', update)
        self.assertLess(update.index('ctype != ST_UPDATE_TYPE'), update.index('self.rfile.read'))

    def test_unsupported_format_rejected_before_body_and_lease_survives(self):
        url, opener, _ = start_coordinator(self)
        auth = {'Authorization': 'Bearer local-test-only'}
        req = urllib.request.Request(url + '/lease?name=python-test&steps=2&batch=1&session=format-test', headers=auth)
        with opener.open(req, timeout=60) as response:
            lease = dbmoe.from_bytes(response.read())
        # A normal old-client format is refused without waiting for an upload.
        parsed = urllib.parse.urlsplit(url)
        for content_type in (None, 'application/octet-stream', 'application/json'):
            with self.subTest(content_type=content_type):
                conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=5)
                try:
                    conn.putrequest('POST', '/update')
                    conn.putheader('Authorization', 'Bearer local-test-only')
                    conn.putheader('Content-Length', '32')
                    if content_type:
                        conn.putheader('Content-Type', content_type)
                    conn.endheaders()  # Deliberately no body: MIME rejection must precede reading.
                    response = conn.getresponse()
                    self.assertEqual(response.status, 415)
                    self.assertIn('upgrade', json.loads(response.read())['error'])
                finally:
                    conn.close()
        up = dbmoe.train_lease(lease, 'cpu')
        up.update(name='python-test', device='cpu')
        req = urllib.request.Request(url + '/update', data=dbmoe.update_to_safetensors(up),
                                     headers=dict(auth, **{'Content-Type': dbmoe.ST_UPDATE_TYPE + '; charset=binary'}))
        with opener.open(req, timeout=60) as response:
            self.assertTrue(json.load(response)['ok'])
        with opener.open(url + '/status', timeout=5) as response:
            self.assertEqual(json.load(response)['updates'], 1)


if __name__ == '__main__':
    unittest.main()
