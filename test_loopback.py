"""End-to-end test against an isolated, tiny coordinator bound only to loopback."""
import json
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request


class LoopbackIntegrationTest(unittest.TestCase):
    def test_cpu_leases_opt_in_and_withdrawal(self):
        script = str(Path(__file__).with_name('dbmoe.py').resolve())
        with tempfile.TemporaryDirectory(prefix='dbmoe-test-') as tmp:
            root = Path(tmp)
            (root / 'training.txt').write_text('This is local synthetic training data only.\n' * 100)
            (root / 'tokens.json').write_text(json.dumps({'local-test-only': 'local'}))
            (root / 'worker.token').write_text('local-test-only')
            cfg = dict(d=16, heads=2, layers_per_block=1, blocks=2, experts=2, ffn=32, ctx=8)
            with socket.socket() as s:
                s.bind(('127.0.0.1', 0))
                port = s.getsockname()[1]
            url = f'http://127.0.0.1:{port}'
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with (root / 'server.log').open('w+') as log:
                server = subprocess.Popen([sys.executable, script, 'coord', '--dir', str(root / 'run'),
                    '--host', '127.0.0.1', '--port', str(port), '--tokens', str(root / 'tokens.json'),
                    '--text-file', str(root / 'training.txt'), '--cfg', json.dumps(cfg), '--steps', '2',
                    '--batch', '1', '--heldout', '2', '--eval-every', '0', '--snap-every', '0',
                    '--eval-threads', '1'], stdout=log, stderr=subprocess.STDOUT)
                try:
                    deadline = time.monotonic() + 20
                    while True:
                        try:
                            with opener.open(url + '/status', timeout=1) as response:
                                json.load(response)
                            break
                        except (urllib.error.URLError, TimeoutError):
                            if server.poll() is not None or time.monotonic() > deadline:
                                log.seek(0)
                                self.fail('Local coordinator did not start: ' + log.read())
                            time.sleep(.1)
                    for opt_in in (True, False):
                        cmd = [sys.executable, script, 'worker', '--url', url,
                               '--token', '@' + str(root / 'worker.token'), '--name', 'public-test',
                               '--device', 'cpu', '--threads', '1', '--steps', '2', '--batch', '1', '--once', '1']
                        if opt_in:
                            cmd.append('--share-hardware')
                        worker = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
                        self.assertEqual(worker.returncode, 0, worker.stdout + worker.stderr)
                        with opener.open(url + '/status', timeout=2) as response:
                            status = json.load(response)
                        stats = status['workers']['local/public-test']
                        self.assertEqual('hardware' in stats, opt_in)
                        self.assertNotIn('host', stats)
                        self.assertNotIn('ip', stats)
                        if opt_in:
                            self.assertTrue(stats['hardware']['self_reported'])
                            self.assertEqual(stats['hardware']['backend'], 'cpu')
                    self.assertEqual(status['updates'], 2)
                    self.assertEqual(status['rejected'], 0)
                    self.assertEqual(status['tokens'], 32)
                finally:
                    server.terminate()
                    try:
                        server.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        server.kill()
                        server.wait(timeout=5)


if __name__ == '__main__':
    unittest.main()
