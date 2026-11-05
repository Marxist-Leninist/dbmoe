"""Apple routing and transport regression tests. Mocks are not physical Mac validation."""
import argparse
import contextlib
import io
import unittest
from unittest import mock

import torch
import dbmoe
from test_dbmoe import tiny_lease


class AppleDeviceTests(unittest.TestCase):
    def setUp(self):
        # Keep selection tests deterministic even when this suite runs on an Intel host.
        patcher = mock.patch.object(dbmoe, 'xpu_available', return_value=False)
        patcher.start()
        self.addCleanup(patcher.stop)

    def fake_mps(self, built=True, available=True):
        backend = mock.Mock()
        backend.is_built.return_value = built
        backend.is_available.return_value = available
        patcher = mock.patch.object(torch.backends, 'mps', backend, create=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        return backend

    def test_cpu_never_probes_mps(self):
        backend = self.fake_mps()
        with mock.patch.object(torch.cuda, 'is_available', side_effect=AssertionError('CUDA touched')):
            self.assertEqual(dbmoe.resolve_worker_device('cpu')['backend'], 'cpu')
        backend.is_available.assert_not_called()
        backend.is_built.assert_not_called()

    def test_explicit_mps_selects_only_mps(self):
        self.fake_mps()
        with mock.patch.object(torch.cuda, 'is_available', side_effect=AssertionError('CUDA touched')):
            for name in ('mps', 'mps:0', ' MPS '):
                self.assertEqual(dbmoe.resolve_worker_device(name),
                                 dict(torch_device='mps', device='mps', backend='mps'))

    def test_auto_prefers_cuda_then_mps_then_cpu(self):
        backend = self.fake_mps()
        with mock.patch.object(torch.cuda, 'is_available', return_value=False):
            self.assertEqual(dbmoe.resolve_worker_device('auto')['backend'], 'mps')
            backend.is_available.return_value = False
            self.assertEqual(dbmoe.resolve_worker_device('auto')['backend'], 'cpu')
        backend.is_available.reset_mock()
        with mock.patch.object(torch.cuda, 'is_available', return_value=True), \
             mock.patch.object(torch.cuda, 'device_count', return_value=1), \
             mock.patch.object(torch.cuda, 'current_device', return_value=0), \
             mock.patch.object(torch.version, 'hip', None):
            self.assertEqual(dbmoe.resolve_worker_device('auto')['backend'], 'cuda')
        backend.is_available.assert_not_called()

    def test_missing_backend_does_not_break_cpu_or_auto(self):
        with mock.patch.object(torch.backends, 'mps', None), \
             mock.patch.object(torch.cuda, 'is_available', return_value=False):
            self.assertEqual(dbmoe.resolve_worker_device('auto')['backend'], 'cpu')
            with self.assertRaisesRegex(RuntimeError, 'no MPS support'):
                dbmoe.resolve_worker_device('mps')

    def test_explicit_unavailable_mps_never_falls_back(self):
        backend = self.fake_mps(built=False, available=False)
        with self.assertRaisesRegex(RuntimeError, 'no MPS support'):
            dbmoe.resolve_worker_device('mps')
        backend.is_built.return_value = True
        with self.assertRaisesRegex(RuntimeError, 'cannot access an MPS GPU'):
            dbmoe.resolve_worker_device('mps')

    def test_invalid_mps_indexes_are_rejected(self):
        self.fake_mps()
        for name in ('mps:', 'mps:1', 'mps:-1', 'mps:00', 'mps:abc', 'metal'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                dbmoe.resolve_worker_device(name)

    def test_mps_preflight_uses_no_cuda_apis(self):
        self.fake_mps()
        with mock.patch.object(torch.cuda, 'set_device', side_effect=AssertionError('CUDA touched')), \
             mock.patch.object(torch.cuda, 'get_device_name', side_effect=AssertionError('CUDA touched')), \
             mock.patch.object(dbmoe, 'device_smoke_test', return_value=[1., .9]) as probe:
            info = dbmoe.prepare_worker_device('mps')
        probe.assert_called_once_with('mps')
        self.assertEqual(info['gpu_name'], 'Metal (MPS)')

    def test_mps_probe_failure_precedes_network_and_token_read(self):
        self.fake_mps()
        a = argparse.Namespace(threads=1, device='mps', token='@missing-token-file')
        with mock.patch.object(dbmoe, 'device_smoke_test', side_effect=RuntimeError('unsupported operation')), \
             mock.patch.object(dbmoe, 'http') as http, \
             mock.patch('builtins.open', side_effect=AssertionError('token file touched')):
            with self.assertRaisesRegex(SystemExit, 'failed the offline training check'):
                dbmoe.worker(a)
        http.assert_not_called()

    def test_hardware_opt_in_does_not_claim_vram_or_query_cuda(self):
        with mock.patch.object(torch.cuda, 'get_device_properties', side_effect=AssertionError('CUDA touched')), \
             mock.patch.object(dbmoe.platform, 'mac_ver', return_value=('14.0', (), 'arm64')), \
             mock.patch.object(dbmoe.socket, 'gethostname', side_effect=AssertionError('hostname touched')):
            report = dbmoe.worker_hardware(dict(backend='mps', torch_device='mps'), 2)
        self.assertEqual(report['backend'], 'mps')
        self.assertEqual(report['runtime_version'], '14.0')
        self.assertEqual(report['gpu_name'], 'Metal (MPS)')
        self.assertNotIn('gpu_memory_gib', report)

    def test_mps_payload_retains_backend_and_requires_metadata_opt_in(self):
        torch.set_num_threads(1)
        lease = tiny_lease()
        update = dbmoe.train_lease(lease, 'cpu')  # transport only, not simulated Metal execution
        for opt_in in (False, True):
            sent = []
            def http(method, url, token, body=None):
                if method == 'GET':
                    return dbmoe.to_bytes(lease)
                sent.append(dbmoe.from_bytes(body))
                return b'{"ok":true}'
            args = argparse.Namespace(threads=1, device='mps', url='https://example.invalid', token='test-only',
                                      name='test', hours=0, steps=2, batch=1, once=1, share_hardware=opt_in)
            info = dict(torch_device='mps', device='mps', backend='mps', gpu_name='Metal (MPS)')
            with mock.patch.object(dbmoe, 'prepare_worker_device', return_value=info), \
                 mock.patch.object(dbmoe, 'train_lease', return_value=update) as train, \
                 mock.patch.object(dbmoe, 'http', side_effect=http), \
                 contextlib.redirect_stdout(io.StringIO()):
                dbmoe.worker(args)
            self.assertEqual(train.call_args.args[1], 'mps')
            self.assertEqual(sent[0]['device'], 'mps')
            self.assertEqual('hardware' in sent[0], opt_in)
            self.assertNotIn('host', sent[0])
            for value in sent[0]['d_block'].values():
                self.assertEqual(value.device.type, 'cpu')

    def test_portable_expert_counts_match_bincount(self):
        for experts in (1, 2, 4, 16):
            for count in (0, 1, 257):
                ids = torch.arange(count, dtype=torch.long) % experts
                torch.testing.assert_close(dbmoe.portable_expert_counts(ids, experts),
                                           torch.bincount(ids, minlength=experts), rtol=0, atol=0)

    @unittest.skipUnless(dbmoe.mps_available(), 'No MPS GPU available; physical Mac test not run')
    def test_real_mps_training_and_cpu_serialization(self):
        torch.set_num_threads(1)
        info = dbmoe.prepare_worker_device('mps')
        self.assertEqual(info['backend'], 'mps')
        self.assertEqual(len(info['check_losses']), 2)


if __name__ == '__main__':
    unittest.main()

