"""End to end: the Rust worker (worker-rs) trains leases from a real local coordinator over the safetensors wire format.

Skipped unless DBMOE_RUST_WORKER names a built dbmoe-worker binary (the worker-rs CI job sets it):

    cargo build --release --manifest-path worker-rs/Cargo.toml
    DBMOE_RUST_WORKER=worker-rs/target/release/dbmoe-worker python -m unittest test_rust_worker_loopback -v
"""
import json
import os
import subprocess
import unittest

from test_safetensors_wire import start_coordinator


class RustWorkerLoopbackTest(unittest.TestCase):
    def test_rust_worker_completes_leases(self):
        exe = os.environ.get('DBMOE_RUST_WORKER')
        if not exe:
            self.skipTest('set DBMOE_RUST_WORKER to a built dbmoe-worker binary to run this test')
        url, opener, root = start_coordinator(self)
        home = str(root / 'home')                                   # keep ~/.dbmoe out of the real home directory
        env = dict(os.environ, HOME=home, USERPROFILE=home)
        cmd = [exe, 'worker', '--url', url, '--token', 'local-test-only', '--name', 'rust-test', '--device', 'cpu',
               '--steps', '2', '--batch', '1', '--once', '2', '--threads', '1', '--seed', '1']
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        with opener.open(url + '/status', timeout=5) as resp:
            status = json.load(resp)
        self.assertEqual(status['updates'], 2, r.stdout + r.stderr)
        self.assertEqual(r.stdout.count('"ok":true'), 2, r.stdout)
        worker = next(iter(status['workers'].values()))
        self.assertEqual(worker['device'], 'cpu')
        self.assertNotIn('hardware', worker)


if __name__ == '__main__':
    unittest.main()
