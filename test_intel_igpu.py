"""Integrated-GPU metadata and allocator-policy simulations, not Intel hardware emulation."""
import argparse
import contextlib
import io
import json
import unittest
from unittest import mock

import torch
import dbmoe


class IntelSharedMemoryTests(unittest.TestCase):
    def fake_xpu(self, integrated=True, total=8 * 2**30):
        backend = mock.Mock()
        backend.is_available.return_value = True
        backend.device_count.return_value = 2
        backend.current_device.return_value = 0
        backend.get_device_name.return_value = 'Mock Intel Arc Graphics'
        backend.get_device_properties.return_value = argparse.Namespace(
            total_memory=total, is_integrated_gpu=integrated)
        # Some supported iGPU/driver combinations cannot report free memory.
        backend.mem_get_info.side_effect = RuntimeError('free-memory query unsupported')
        patcher = mock.patch.object(torch, 'xpu', backend, create=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        return backend

    def test_integrated_property_not_name_controls_memory_label(self):
        for integrated, expected in ((True, 'shared'), (False, 'dedicated'), (None, 'unknown'), (1, 'unknown')):
            with self.subTest(integrated=integrated):
                props = argparse.Namespace(name='Intel Arc Graphics', is_integrated_gpu=integrated)
                self.assertEqual(dbmoe.xpu_memory_kind(props), expected)
        self.assertEqual(dbmoe.xpu_memory_kind(argparse.Namespace()), 'unknown')

    def test_cap_is_applied_to_selected_device_before_training(self):
        backend = self.fake_xpu()
        events = []
        backend.set_per_process_memory_fraction.side_effect = lambda *args, **kwargs: events.append('cap')
        with mock.patch.object(dbmoe, 'device_smoke_test', side_effect=lambda device: events.append('train') or [1., .9]):
            info = dbmoe.prepare_worker_device('xpu:1', xpu_memory_fraction=.5)
        self.assertEqual(events, ['cap', 'train'])
        backend.set_per_process_memory_fraction.assert_called_once_with(.5, device='xpu:1')
        self.assertEqual(info['gpu_memory_kind'], 'shared')
        self.assertEqual(info['xpu_memory_fraction'], .5)
        self.assertEqual(info['xpu_allocator_limit_gib'], 4.)
        backend.mem_get_info.assert_not_called()

    def test_no_cap_requested_keeps_existing_behavior(self):
        backend = self.fake_xpu()
        with mock.patch.object(dbmoe, 'device_smoke_test', return_value=[1., .9]):
            info = dbmoe.prepare_worker_device('xpu')
        backend.set_per_process_memory_fraction.assert_not_called()
        backend.mem_get_info.assert_not_called()
        self.assertNotIn('xpu_allocator_limit_gib', info)

    def test_unknown_integrated_property_does_not_block_training(self):
        backend = self.fake_xpu()
        backend.get_device_properties.return_value = argparse.Namespace(total_memory=8 * 2**30)
        with mock.patch.object(dbmoe, 'device_smoke_test', return_value=[1., .9]):
            info = dbmoe.prepare_worker_device('xpu', xpu_memory_fraction=.5)
        self.assertEqual(info['gpu_memory_kind'], 'unknown')

    def test_invalid_fractions_are_rejected_before_setting_or_training(self):
        backend = self.fake_xpu()
        for value in (0., -.1, 1.01, float('nan'), float('inf'), True, '.5'):
            with self.subTest(value=value), mock.patch.object(dbmoe, 'device_smoke_test') as train:
                with self.assertRaisesRegex(ValueError, '--xpu-memory-fraction'):
                    dbmoe.prepare_worker_device('xpu', xpu_memory_fraction=value)
                train.assert_not_called()
        backend.set_per_process_memory_fraction.assert_not_called()

    def test_cap_rejected_on_non_xpu_without_touching_xpu(self):
        with mock.patch.object(torch, 'xpu', None, create=True), \
             mock.patch.object(dbmoe, 'device_smoke_test') as train:
            with self.assertRaisesRegex(ValueError, 'requires an Intel XPU device'):
                dbmoe.prepare_worker_device('cpu', xpu_memory_fraction=.5)
        train.assert_not_called()

    def test_missing_allocator_api_is_not_silently_ignored(self):
        backend = self.fake_xpu()
        backend.set_per_process_memory_fraction = None
        with mock.patch.object(dbmoe, 'device_smoke_test') as train:
            with self.assertRaisesRegex(RuntimeError, 'cannot enforce'):
                dbmoe.prepare_worker_device('xpu', xpu_memory_fraction=.5)
        train.assert_not_called()

    def test_invalid_total_memory_does_not_guess_a_budget(self):
        backend = self.fake_xpu()
        for total in (0, -1, float('nan'), float('inf'), True, '8 GiB'):
            backend.get_device_properties.return_value.total_memory = total
            with self.subTest(total=total), self.assertRaisesRegex(RuntimeError, 'valid total device memory'):
                dbmoe.configure_xpu_memory(dict(backend='xpu', torch_device='xpu:0'), .5)
        backend.set_per_process_memory_fraction.assert_not_called()

    def test_cap_failure_precedes_credentials_and_network(self):
        backend = self.fake_xpu()
        backend.set_per_process_memory_fraction.side_effect = RuntimeError('allocator unavailable')
        args = argparse.Namespace(threads=1, device='xpu', xpu_memory_fraction=.5, token='@missing-token')
        with mock.patch.object(dbmoe, 'device_smoke_test') as train, \
             mock.patch.object(dbmoe, 'http') as http, \
             mock.patch.object(dbmoe, 'load_or_join') as join, \
             mock.patch('builtins.open', side_effect=AssertionError('credential touched')):
            with self.assertRaisesRegex(SystemExit, 'Cannot apply the requested XPU memory cap'):
                dbmoe.worker(args)
        train.assert_not_called()
        http.assert_not_called()
        join.assert_not_called()

    def test_shared_hardware_metadata_uses_allowlisted_label(self):
        self.fake_xpu()
        report = dbmoe.worker_hardware(dict(backend='xpu', torch_device='xpu:0', gpu_name='Mock Arc'), 2)
        self.assertEqual(report['gpu_memory_kind'], 'shared')
        self.assertEqual(report['gpu_memory_gib'], 8.)
        for invalid in ('unbounded private text', None, ['shared'], torch.tensor([1, 2])):
            with self.subTest(value=invalid):
                sanitized = dbmoe.normalize_hardware(dict(report, gpu_memory_kind=invalid))
                self.assertNotIn('gpu_memory_kind', sanitized)

    def test_worker_and_check_cli_accept_fraction(self):
        for command, target in (('worker', 'worker'), ('check-device', 'check_device_cmd')):
            with mock.patch.object(dbmoe.sys, 'argv', ['dbmoe.py', command, '--device', 'xpu', '--xpu-memory-fraction', '.5']), \
                 mock.patch.object(dbmoe, target) as execute:
                dbmoe.main()
            self.assertEqual(execute.call_args.args[0].xpu_memory_fraction, .5)

    def test_check_device_reports_local_budget(self):
        self.fake_xpu()
        output = io.StringIO()
        with mock.patch.object(dbmoe, 'device_smoke_test', return_value=[1., .9]), \
             contextlib.redirect_stdout(output):
            dbmoe.check_device_cmd(argparse.Namespace(device='xpu', threads=1, xpu_memory_fraction=.25))
        report = json.loads(output.getvalue())
        self.assertEqual(report['gpu_memory_kind'], 'shared')
        self.assertEqual(report['xpu_allocator_limit_gib'], 2.)
        self.assertEqual(report['xpu_memory_fraction'], .25)


if __name__ == '__main__':
    unittest.main()
