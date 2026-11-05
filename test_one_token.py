"""One minted token runs a whole network of machines, each under its own name."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request


class OneTokenManyMachines(unittest.TestCase):
    def test_mint_once_lease_from_many(self):
        script = str(Path(__file__).with_name('dbmoe.py').resolve())
        with tempfile.TemporaryDirectory(prefix='dbmoe-test-') as tmp:
            root = Path(tmp)
            (root / 'training.txt').write_text('This is local synthetic training data only.\n' * 100)
            (root / 'tokens.json').write_text('{}')
            cfg = dict(d=16, heads=2, layers_per_block=1, blocks=2, experts=2, ffn=32, ctx=8)
            with socket.socket() as s:
                s.bind(('127.0.0.1', 0))
                port = s.getsockname()[1]
            url = f'http://127.0.0.1:{port}'
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

            def get(path, token=None):
                req = urllib.request.Request(url + path, headers={'Authorization': f'Bearer {token}'} if token else {})
                try:
                    with opener.open(req, timeout=10) as r:
                        return r.status, r.read()
                except urllib.error.HTTPError as e:
                    return e.code, e.read()

            with (root / 'server.log').open('w+') as log:
                server = subprocess.Popen([sys.executable, script, 'coord', '--dir', str(root / 'run'),
                    '--host', '127.0.0.1', '--port', str(port), '--tokens', str(root / 'tokens.json'),
                    '--text-file', str(root / 'training.txt'), '--cfg', json.dumps(cfg), '--steps', '2',
                    '--batch', '1', '--heldout', '2', '--eval-every', '0', '--snap-every', '0',
                    '--eval-threads', '1', '--join-bits', '4', '--min-steps', '1'], stdout=log, stderr=subprocess.STDOUT)
                try:
                    # a cold Python + PyTorch start on a fresh CI runner (notably macOS) can exceed 20 s
                    deadline = time.monotonic() + float(os.environ.get('DBMOE_TEST_START_TIMEOUT', 120))
                    while True:
                        try:
                            if get('/status')[0] == 200:
                                break
                        except urllib.error.URLError:
                            pass
                        if server.poll() is not None or time.monotonic() > deadline:
                            log.seek(0)
                            self.fail('Local coordinator did not start: ' + log.read())
                        time.sleep(.1)
                    path = root / 'worker.token'
                    r = subprocess.run([sys.executable, script, 'enrol', '--url', url, '--token-file', str(path)],
                                       capture_output=True, text=True, timeout=60)
                    self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
                    if sys.platform != 'win32':     # Windows has no POSIX permission bits; chmod only toggles read-only
                        self.assertEqual(path.stat().st_mode & 0o777, 0o600)
                    token = path.read_text().strip()
                    self.assertNotIn(token, r.stdout + r.stderr)
                    # machines of one network start at the same moment
                    codes = [get(f'/lease?name=machine-{i}&session=s', token)[0] for i in range(6)]
                    self.assertEqual(codes, [200] * 6)
                    self.assertEqual(get('/lease?name=machine-0&session=s', token)[0], 429)     # one machine asking too fast
                    status = json.loads(get('/status')[1])
                    self.assertEqual(status['open_join']['participants'], 1)
                    self.assertEqual(len(status['active_leases']), 6)
                finally:
                    server.terminate()
                    try:
                        server.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        server.kill()
                        server.wait(timeout=5)


if __name__ == '__main__':
    unittest.main()
