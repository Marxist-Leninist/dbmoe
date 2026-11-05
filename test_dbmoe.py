"""CPU regression tests and mocked backend selection; actual GPU test skips without a GPU.

Run: python -m unittest -v
Mocks test routing only and are not evidence of AMD or NVIDIA hardware validation.
"""
import argparse
import contextlib
import io
import json
import math
import threading
import unittest
from unittest import mock

import torch
import dbmoe


def tiny_lease():
    cfg = dbmoe.make_cfg(dict(d=16, heads=2, layers_per_block=1, blocks=2, experts=2, ffn=32, ctx=8))
    return dict(cfg=cfg, block=0, block_version=0, sigmas=dbmoe.cfg_sigmas(cfg), lease='test-only',
                shared=dbmoe.Shared(cfg).state_dict(), block_state=dbmoe.Block(cfg).state_dict(),
                steps=2, batch=1, lr=1e-3, data=torch.arange(18, dtype=torch.uint8).reshape(2, 9))


class DeviceSelectionTests(unittest.TestCase):
    def fake_gpu(self, hip=None, available=True, count=2, current=0):
        stack = contextlib.ExitStack()
        stack.enter_context(mock.patch.object(torch.version, 'hip', hip))
        stack.enter_context(mock.patch.object(torch.cuda, 'is_available', return_value=available))
        stack.enter_context(mock.patch.object(torch.cuda, 'device_count', return_value=count))
        stack.enter_context(mock.patch.object(torch.cuda, 'current_device', return_value=current))
        self.addCleanup(stack.close)

    def test_cpu_does_not_probe_cuda(self):
        with mock.patch.object(torch.cuda, 'is_available', side_effect=AssertionError('GPU touched')):
            self.assertEqual(dbmoe.resolve_worker_device('cpu'),
                             dict(torch_device='cpu', device='cpu', backend='cpu'))

    def test_cpu_default_does_not_run_training_probe(self):
        with mock.patch.object(dbmoe, 'device_smoke_test') as probe:
            dbmoe.prepare_worker_device('cpu')
            probe.assert_not_called()

    def test_auto_cpu(self):
        self.fake_gpu(available=False)
        with mock.patch.object(dbmoe, 'mps_available', return_value=False):
            self.assertEqual(dbmoe.resolve_worker_device('auto')['backend'], 'cpu')

    def test_nvidia_cuda_and_index(self):
        self.fake_gpu()
        self.assertEqual(dbmoe.resolve_worker_device('cuda:1'),
                         dict(torch_device='cuda:1', device='cuda:1', backend='cuda'))
        self.assertEqual(dbmoe.resolve_worker_device('cuda')['torch_device'], 'cuda:0')
        self.assertEqual(dbmoe.resolve_worker_device('auto')['backend'], 'cuda')

    def test_rocm_aliases_and_legacy_cuda(self):
        self.fake_gpu(hip='test-hip')
        for spec in ('rocm:1', 'hip:1', 'cuda:1'):
            with self.subTest(spec=spec):
                self.assertEqual(dbmoe.resolve_worker_device(spec),
                                 dict(torch_device='cuda:1', device='rocm:1', backend='rocm'))
        for spec in ('rocm', 'hip', 'cuda', 'auto'):
            self.assertEqual(dbmoe.resolve_worker_device(spec)['backend'], 'rocm')

    def test_unavailable_explicit_gpu_never_falls_back(self):
        self.fake_gpu(hip='test-hip', available=False)
        for spec in ('cuda', 'cuda:0', 'rocm', 'rocm:0', 'hip'):
            with self.subTest(spec=spec), self.assertRaisesRegex(RuntimeError, 'cannot access a GPU'):
                dbmoe.resolve_worker_device(spec)

    def test_rocm_rejects_non_hip_build_even_with_nvidia_gpu(self):
        self.fake_gpu()
        with self.assertRaisesRegex(RuntimeError, 'no HIP support'):
            dbmoe.resolve_worker_device('rocm')

    def test_invalid_and_out_of_range_devices(self):
        self.fake_gpu(hip='test-hip')
        for spec in ('', 'cpu:0', 'cuda:', 'rocm:-1', 'cuda:abc', 'rocm:1:2', 'cuda:²', 'metal', 'auto:1'):
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                dbmoe.resolve_worker_device(spec)
        with self.assertRaisesRegex(ValueError, 'out of range'):
            dbmoe.resolve_worker_device('rocm:2')

    def test_selected_device_is_used_for_gpu_probe(self):
        self.fake_gpu(hip='test-hip')
        with mock.patch.object(torch.cuda, 'set_device') as select, \
             mock.patch.object(torch.cuda, 'get_device_name', return_value='Mock AMD GPU'), \
             mock.patch.object(dbmoe, 'device_smoke_test', return_value=[1., 1.]) as probe:
            info = dbmoe.prepare_worker_device('rocm:1')
        select.assert_called_once_with('cuda:1')
        probe.assert_called_once_with('cuda:1')
        self.assertEqual(info['device'], 'rocm:1')

    def test_probe_failure_is_fatal_before_any_lease_or_token_read(self):
        a = argparse.Namespace(threads=1, device='rocm', token='@missing-token-file')
        with mock.patch.object(dbmoe, 'prepare_worker_device', side_effect=RuntimeError('bad driver')), \
             mock.patch.object(dbmoe, 'http') as http:
            with self.assertRaisesRegex(SystemExit, 'bad driver'):
                dbmoe.worker(a)
            http.assert_not_called()


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_offline_cpu_probe_without_network(self):
        with mock.patch.object(dbmoe, 'http', side_effect=AssertionError('network touched')):
            losses = dbmoe.device_smoke_test('cpu')
        self.assertEqual(len(losses), 2)
        self.assertTrue(all(math.isfinite(v) for v in losses))

    def test_cpu_train_delta_format_cache_and_coordinator_validation(self):
        torch.manual_seed(123)
        L = tiny_lease()
        original = {section: {k: v.clone() for k, v in L[section].items()}
                    for section in ('shared', 'block_state')}
        cache = {}
        up = dbmoe.from_bytes(dbmoe.to_bytes(dbmoe.train_lease(L, 'cpu', cache)))
        self.assertEqual(up['tokens'], 16)
        self.assertEqual(len(up['losses']), 2)
        self.assertTrue(up['fresh_optimizer'])
        C = dbmoe.Coordinator.__new__(dbmoe.Coordinator)
        C.lr, C.model = L['lr'], dbmoe.Model(L['cfg'])
        self.assertIsNone(C._check(up, L))
        for section in original:
            for key in original[section]:
                torch.testing.assert_close(L[section][key], original[section][key], rtol=0, atol=0)
        self.assertTrue(any(bool(t.abs().max() > 0) for t in up['d_block'].values()))
        for section in ('d_block', 'd_shared'):
            for t in up[section].values():
                self.assertEqual(t.device.type, 'cpu')
                self.assertEqual(t.dtype, torch.float32)
                self.assertTrue(bool(torch.isfinite(t).all()))
        L['block_version'] = 1
        second = dbmoe.train_lease(L, 'cpu', cache)
        self.assertFalse(second['fresh_optimizer'])
        self.assertIsNone(C._check(second, L))

    def test_worker_one_cpu_lease_round_trip(self):
        L = tiny_lease()
        calls = []
        def http(method, url, token, body=None):
            calls.append((method, url, token, body))
            return dbmoe.to_bytes(L) if method == 'GET' else json.dumps(dict(ok=True)).encode()
        a = argparse.Namespace(threads=1, device='cpu', url='https://example.invalid', token='test-only',
                               name='test', hours=0, steps=2, batch=1, once=1)
        with mock.patch.object(dbmoe, 'http', side_effect=http), contextlib.redirect_stdout(io.StringIO()):
            dbmoe.worker(a)
        self.assertEqual([c[0] for c in calls], ['GET', 'POST'])
        self.assertEqual(dbmoe.from_bytes(calls[1][3])['device'], 'cpu')

    def test_amd_payload_uses_rocm_label_not_pytorch_cuda_label(self):
        L = tiny_lease()
        up = dbmoe.train_lease(L, 'cpu')
        sent = []
        def http(method, url, token, body=None):
            if body:
                sent.append(dbmoe.from_bytes(body))
            return dbmoe.to_bytes(L) if method == 'GET' else b'{"ok":true}'
        a = argparse.Namespace(threads=1, device='rocm', url='https://example.invalid', token='test-only',
                               name='test', hours=0, steps=2, batch=1, once=1)
        info = dict(torch_device='cuda:0', device='rocm:0', backend='rocm', gpu_name='Mock GPU')
        with mock.patch.object(dbmoe, 'prepare_worker_device', return_value=info), \
             mock.patch.object(dbmoe, 'train_lease', return_value=up) as train, \
             mock.patch.object(dbmoe, 'http', side_effect=http), contextlib.redirect_stdout(io.StringIO()):
            dbmoe.worker(a)
        self.assertEqual(train.call_args.args[1], 'cuda:0')
        self.assertEqual(sent[0]['device'], 'rocm:0')

    @unittest.skipUnless(torch.cuda.is_available(), 'No CUDA/ROCm GPU available; hardware test not run')
    def test_real_gpu_forward_backward_and_serialization(self):
        kind = 'rocm' if torch.version.hip else 'cuda'
        info = dbmoe.prepare_worker_device(kind)
        self.assertEqual(info['backend'], kind)
        self.assertEqual(len(info['check_losses']), 2)


class HardwarePrivacyTests(unittest.TestCase):
    def test_cli_default_name_does_not_read_hostname(self):
        with mock.patch.object(dbmoe.sys, 'argv', ['dbmoe.py', 'worker', '--url', 'https://example.invalid',
                                                 '--token', 'test-only']), \
             mock.patch.object(dbmoe.socket, 'gethostname', side_effect=AssertionError('hostname touched')), \
             mock.patch.object(dbmoe, 'worker') as worker:
            dbmoe.main()
        self.assertTrue(worker.call_args.args[0].name.startswith('worker-'))

    def test_status_filters_legacy_host_and_ip(self):
        C = dbmoe.Coordinator.__new__(dbmoe.Coordinator)
        C.lock = threading.RLock()
        C.model = dbmoe.Model(tiny_lease()['cfg'])
        C.meta = dict(updates=0, tokens=0, rejected=0, block_updates=[0, 0], block_ce=[None, None],
                      evals=[], workers={'pseudonym': dict(host='private-host', ip='192.0.2.1', device='cpu')})
        C.leases = {}
        C.data = argparse.Namespace(docs=0, buf=b'', err=None)
        C.steps, C.batch, C.lr, C.max_steps, C.max_batch, C.ttl = 2, 1, .001, 10, 2, 60
        self.assertEqual(C.status()['workers']['pseudonym'], dict(device='cpu'))
        self.assertIn('host', C.meta['workers']['pseudonym'])  # read-only rendering, not destructive migration

    def test_allowlist_discards_private_and_unknown_fields(self):
        report = dbmoe.normalize_hardware(dict(schema=1, self_reported=True, backend='rocm',
            cpu_arch='x86_64', cpu_threads=2, torch_version='test', runtime_version='test',
            gpu_name='AMD\n Test', gpu_memory_gib=15.96, host='secret-host', ip='192.0.2.1',
            serial='secret-serial', token='private-token', path='/private/path'))
        self.assertEqual(set(report), {'schema', 'self_reported', 'backend', 'cpu_arch', 'cpu_threads',
                                      'torch_version', 'runtime_version', 'gpu_name', 'gpu_memory_gib'})
        self.assertEqual(report['gpu_name'], 'AMD Test')
        self.assertEqual(report['gpu_memory_gib'], 16.0)

    def test_malformed_metadata_is_ignored_without_raising(self):
        for report in (None, [], 'text', {'schema': torch.tensor([1, 2])},
                       {'schema': True}, {'schema': 1, 'self_reported': False},
                       {'schema': 1, 'self_reported': True, 'backend': torch.tensor([1, 2])}):
            with self.subTest(report=report):
                self.assertIsNone(dbmoe.normalize_hardware(report))
        for memory in (float('nan'), float('inf'), -1, True, [16], torch.tensor([16, 16])):
            report = dbmoe.normalize_hardware(dict(schema=1, self_reported=True, backend='rocm',
                                                   gpu_memory_gib=memory, cpu_threads=True))
            self.assertNotIn('gpu_memory_gib', report)
            self.assertNotIn('cpu_threads', report)

    def test_cpu_hardware_never_queries_gpu_or_hostname(self):
        with mock.patch.object(torch.cuda, 'get_device_properties', side_effect=AssertionError('GPU touched')), \
             mock.patch.object(dbmoe.socket, 'gethostname', side_effect=AssertionError('hostname touched')):
            report = dbmoe.worker_hardware(dict(backend='cpu', torch_device='cpu'), 2)
        self.assertEqual(report['cpu_threads'], 2)
        self.assertEqual(report['backend'], 'cpu')
        self.assertNotIn('gpu_name', report)

    def test_gpu_hardware_queries_selected_device_only(self):
        props = argparse.Namespace(total_memory=24 * 2**30)
        with mock.patch.object(torch.cuda, 'get_device_properties', return_value=props) as get_props:
            report = dbmoe.worker_hardware(dict(backend='rocm', torch_device='cuda:1', gpu_name='Mock AMD'), 3)
        get_props.assert_called_once_with('cuda:1')
        self.assertEqual(report['gpu_memory_gib'], 24.)
        self.assertEqual(report['gpu_name'], 'Mock AMD')

    def test_worker_metadata_requires_opt_in_and_never_sends_host(self):
        L = tiny_lease()
        up = dbmoe.train_lease(L, 'cpu')
        for opted_in in (False, True):
            calls = []
            def http(method, url, token, body=None):
                if body:
                    calls.append(dbmoe.from_bytes(body))
                return dbmoe.to_bytes(L) if method == 'GET' else b'{"ok":true}'
            a = argparse.Namespace(threads=1, device='cpu', url='https://example.invalid', token='test-only',
                                   name='test', hours=0, steps=2, batch=1, once=1, share_hardware=opted_in)
            with self.subTest(opted_in=opted_in), \
                 mock.patch.object(dbmoe, 'train_lease', return_value=dict(up)), \
                 mock.patch.object(dbmoe, 'http', side_effect=http), contextlib.redirect_stdout(io.StringIO()):
                dbmoe.worker(a)
            self.assertNotIn('host', calls[0])
            self.assertNotIn('ip', calls[0])
            self.assertEqual('hardware' in calls[0], opted_in)


if __name__ == '__main__':
    unittest.main()

