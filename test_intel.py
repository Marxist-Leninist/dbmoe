"""Intel XPU routing/transport regressions. Mocks are not physical Intel GPU validation."""
import argparse
import contextlib
import io
import json
import unittest
from unittest import mock

import torch
import dbmoe
from test_dbmoe import tiny_lease


class IntelDeviceTests(unittest.TestCase):
    def fake_xpu(self, available=True, count=2, current=0):
        backend = mock.Mock()
        backend.is_available.return_value = available
        backend.device_count.return_value = count
        backend.current_device.return_value = current
        backend.get_device_name.return_value = 'Mock Intel GPU'
        backend.get_device_properties.return_value = argparse.Namespace(total_memory=16 * 2**30)
        patcher = mock.patch.object(torch, 'xpu', backend, create=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        return backend

    def test_cpu_never_probes_xpu(self):
        backend = self.fake_xpu()
        backend.is_available.side_effect = AssertionError('XPU touched')
        self.assertEqual(dbmoe.resolve_worker_device('cpu')['backend'], 'cpu')
        backend.is_available.assert_not_called()

    def test_explicit_xpu_and_intel_alias_use_only_xpu(self):
        backend = self.fake_xpu(current=1)
        with mock.patch.object(torch.cuda, 'is_available', side_effect=AssertionError('CUDA touched')), \
             mock.patch.object(dbmoe, 'mps_available', side_effect=AssertionError('MPS touched')):
            for name in ('xpu', 'intel', ' XPU:1 ', 'intel:1'):
                self.assertEqual(dbmoe.resolve_worker_device(name),
                                 dict(torch_device='xpu:1', device='xpu:1', backend='xpu'))
            self.assertEqual(dbmoe.resolve_worker_device('xpu:0')['torch_device'], 'xpu:0')
        self.assertTrue(backend.is_available.called)

    def test_auto_order_cuda_xpu_mps_cpu(self):
        backend = self.fake_xpu()
        with mock.patch.object(torch.cuda, 'is_available', return_value=True), \
             mock.patch.object(torch.cuda, 'device_count', return_value=1), \
             mock.patch.object(torch.cuda, 'current_device', return_value=0), \
             mock.patch.object(torch.version, 'hip', None), \
             mock.patch.object(dbmoe, 'mps_available', return_value=True) as mps:
            self.assertEqual(dbmoe.resolve_worker_device('auto')['backend'], 'cuda')
            backend.is_available.assert_not_called()
            mps.assert_not_called()
        with mock.patch.object(torch.cuda, 'is_available', return_value=False), \
             mock.patch.object(torch.backends.mps, 'is_built', return_value=True), \
             mock.patch.object(torch.backends.mps, 'is_available', return_value=True), \
             mock.patch.object(dbmoe, 'mps_available', return_value=True) as mps:
            self.assertEqual(dbmoe.resolve_worker_device('auto')['backend'], 'xpu')
            mps.assert_not_called()
            backend.is_available.return_value = False
            self.assertEqual(dbmoe.resolve_worker_device('auto')['backend'], 'mps')
            mps.return_value = False
            self.assertEqual(dbmoe.resolve_worker_device('auto')['backend'], 'cpu')

    def test_missing_xpu_backend_supports_cpu_and_auto(self):
        with mock.patch.object(torch, 'xpu', None, create=True), \
             mock.patch.object(torch.cuda, 'is_available', return_value=False), \
             mock.patch.object(dbmoe, 'mps_available', return_value=False):
            self.assertFalse(dbmoe.xpu_available())
            self.assertEqual(dbmoe.resolve_worker_device('cpu')['backend'], 'cpu')
            self.assertEqual(dbmoe.resolve_worker_device('auto')['backend'], 'cpu')
            with self.assertRaisesRegex(RuntimeError, 'XPU-enabled PyTorch'):
                dbmoe.resolve_worker_device('xpu')

    def test_unavailable_or_broken_explicit_xpu_never_falls_back(self):
        backend = self.fake_xpu(available=False)
        with mock.patch.object(torch.cuda, 'is_available', return_value=True) as cuda:
            for name in ('xpu', 'xpu:0', 'intel', 'intel:1'):
                with self.subTest(name=name), self.assertRaisesRegex(RuntimeError, 'cannot access an Intel GPU'):
                    dbmoe.resolve_worker_device(name)
            backend.is_available.side_effect = RuntimeError('driver initialization failed')
            with self.assertRaisesRegex(RuntimeError, 'cannot access an Intel GPU'):
                dbmoe.resolve_worker_device('xpu')
            cuda.assert_not_called()

    def test_invalid_indexes_rejected_without_probing(self):
        backend = self.fake_xpu()
        for name in ('xpu:', 'xpu:-1', 'xpu:1:2', 'xpu:abc', 'xpu:²', 'intel:1.0', 'xpu:+1'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                dbmoe.resolve_worker_device(name)
        backend.is_available.assert_not_called()

    def test_out_of_range_and_empty_device_list_rejected(self):
        backend = self.fake_xpu()
        with self.assertRaisesRegex(ValueError, 'out of range'):
            dbmoe.resolve_worker_device('xpu:2')
        backend.device_count.return_value = 0
        with self.assertRaisesRegex(ValueError, 'out of range'):
            dbmoe.resolve_worker_device('xpu')

    def test_selected_xpu_is_used_for_preflight(self):
        backend = self.fake_xpu()
        with mock.patch.object(torch.cuda, 'set_device', side_effect=AssertionError('CUDA touched')), \
             mock.patch.object(torch.cuda, 'get_device_name', side_effect=AssertionError('CUDA touched')), \
             mock.patch.object(dbmoe, 'device_smoke_test', return_value=[1., .9]) as probe:
            info = dbmoe.prepare_worker_device('xpu:1')
        backend.set_device.assert_called_once_with('xpu:1')
        backend.get_device_name.assert_called_once_with('xpu:1')
        probe.assert_called_once_with('xpu:1')
        self.assertEqual(info['gpu_name'], 'Mock Intel GPU')

    def test_probe_failure_precedes_network_enrolment_and_token_read(self):
        self.fake_xpu()
        a = argparse.Namespace(threads=1, device='xpu', token='@missing-token-file')
        with mock.patch.object(dbmoe, 'device_smoke_test', side_effect=RuntimeError('unsupported operation')), \
             mock.patch.object(dbmoe, 'http') as http, \
             mock.patch.object(dbmoe, 'load_or_join') as join, \
             mock.patch('builtins.open', side_effect=AssertionError('token file touched')):
            with self.assertRaisesRegex(SystemExit, 'failed the offline training check'):
                dbmoe.worker(a)
        http.assert_not_called()
        join.assert_not_called()

    def test_hardware_report_uses_selected_xpu_and_allowlist(self):
        backend = self.fake_xpu()
        with mock.patch.object(torch.cuda, 'get_device_properties', side_effect=AssertionError('CUDA touched')), \
             mock.patch.object(torch.version, 'xpu', 'test-xpu', create=True), \
             mock.patch.object(dbmoe.socket, 'gethostname', side_effect=AssertionError('hostname touched')):
            report = dbmoe.worker_hardware(dict(backend='xpu', torch_device='xpu:1', gpu_name='Mock Intel GPU'), 2)
        backend.get_device_properties.assert_called_once_with('xpu:1')
        self.assertEqual(report['backend'], 'xpu')
        self.assertEqual(report['runtime_version'], 'test-xpu')
        self.assertEqual(report['gpu_memory_gib'], 16.)
        self.assertEqual(report['gpu_name'], 'Mock Intel GPU')
        self.assertNotIn('host', report)

    def test_unknown_xpu_runtime_version_is_empty_not_invented(self):
        self.fake_xpu()
        with mock.patch.object(torch.version, 'xpu', None, create=True):
            report = dbmoe.worker_hardware(dict(backend='xpu', torch_device='xpu:0', gpu_name='Mock Intel GPU'), 2)
        self.assertEqual(report['runtime_version'], '')

    def test_xpu_payload_retains_backend_and_hardware_opt_in(self):
        torch.set_num_threads(1)
        lease = tiny_lease()
        update = dbmoe.train_lease(lease, 'cpu')  # Transport only; this does not simulate Intel execution.
        self.fake_xpu()
        for opt_in in (False, True):
            sent = []
            def http(method, url, token, body=None):
                if method == 'GET':
                    return dbmoe.to_bytes(lease)
                sent.append(dbmoe.from_bytes(body))
                return b'{"ok":true}'
            args = argparse.Namespace(threads=1, device='xpu:1', url='https://example.invalid', token='test-only',
                                      name='test', hours=0, steps=2, batch=1, once=1, share_hardware=opt_in)
            info = dict(torch_device='xpu:1', device='xpu:1', backend='xpu', gpu_name='Mock Intel GPU')
            with mock.patch.object(dbmoe, 'prepare_worker_device', return_value=info), \
                 mock.patch.object(dbmoe, 'train_lease', return_value=dict(update)) as train, \
                 mock.patch.object(dbmoe, 'http', side_effect=http), \
                 contextlib.redirect_stdout(io.StringIO()):
                dbmoe.worker(args)
            self.assertEqual(train.call_args.args[1], 'xpu:1')
            self.assertEqual(sent[0]['device'], 'xpu:1')
            self.assertEqual('hardware' in sent[0], opt_in)
            if opt_in:
                self.assertEqual(sent[0]['hardware']['backend'], 'xpu')
            self.assertNotIn('host', sent[0])
            for section in ('d_shared', 'd_block'):
                for value in sent[0][section].values():
                    self.assertEqual(value.device.type, 'cpu')
                    self.assertEqual(value.dtype, torch.float32)

    def test_cli_accepts_xpu_without_network(self):
        with mock.patch.object(dbmoe.sys, 'argv', ['dbmoe.py', 'worker', '--device', 'xpu:1']), \
             mock.patch.object(dbmoe, 'worker') as worker:
            dbmoe.main()
        self.assertEqual(worker.call_args.args[0].device, 'xpu:1')

    def test_offline_cli_reports_xpu_version_and_device(self):
        self.fake_xpu()
        output = io.StringIO()
        with mock.patch.object(dbmoe, 'device_smoke_test', return_value=[1., .9]), \
             mock.patch.object(torch.version, 'xpu', 'test-xpu', create=True), \
             mock.patch.object(dbmoe, 'http', side_effect=AssertionError('network touched')), \
             contextlib.redirect_stdout(output):
            dbmoe.check_device_cmd(argparse.Namespace(device='xpu:1', threads=1))
        report = json.loads(output.getvalue())
        self.assertTrue(report['ok'])
        self.assertEqual(report['torch_device'], 'xpu:1')
        self.assertEqual(report['xpu_version'], 'test-xpu')

    @unittest.skipUnless(dbmoe.xpu_available(), 'No Intel XPU GPU available; physical Intel test not run')
    def test_real_xpu_training_and_cpu_serialization(self):
        torch.set_num_threads(1)
        info = dbmoe.prepare_worker_device('xpu')
        self.assertEqual(info['backend'], 'xpu')
        self.assertEqual(len(info['check_losses']), 2)


if __name__ == '__main__':
    unittest.main()
