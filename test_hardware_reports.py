"""Privacy and offline report regressions. No community/hardware validation is claimed."""
import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch
import dbmoe


class HardwareReportTests(unittest.TestCase):
    def test_real_cpu_check_is_local_and_not_verified_hardware(self):
        with mock.patch.object(dbmoe, 'http', side_effect=AssertionError('network')), \
             mock.patch.object(dbmoe.urllib.request, 'urlopen', side_effect=AssertionError('network')), \
             mock.patch.object(dbmoe, 'load_or_join', side_effect=AssertionError('credential')), \
             mock.patch.object(dbmoe.socket, 'gethostname', side_effect=AssertionError('hostname')):
            report = dbmoe.hardware_report('cpu', threads=1)
        self.assertEqual(report['result'], 'passed')
        self.assertTrue(report['self_reported'])
        self.assertFalse(report['independently_verified'])
        self.assertFalse(report['network_training_tested'])
        self.assertFalse(report['full_model_capacity_tested'])
        self.assertFalse(report['hardware_opt_in'])
        self.assertNotIn('hardware', report)
        self.assertEqual(len(report['source_sha256']), 64)
        self.assertEqual(set(report['checks']), {'forward_backward', 'adamw', 'bounded_cpu_deltas', 'safe_serialization'})

    def test_hardware_query_requires_explicit_opt_in(self):
        with mock.patch.object(dbmoe, 'prepare_worker_device', return_value={}), \
             mock.patch.object(dbmoe, 'report_hardware', return_value={'gpu': 'none'}) as details:
            report = dbmoe.hardware_report('cpu')
            details.assert_not_called()
            report = dbmoe.hardware_report('cpu', include_hardware=True)
        details.assert_called_once()
        self.assertTrue(report['hardware_opt_in'])
        self.assertEqual(report['hardware'], {'gpu': 'none'})

    def test_raw_errors_and_private_request_values_never_enter_report(self):
        secret = '/home/private-user/token.txt api_key=SECRET 192.0.2.3 user@example.test'
        for error, category in ((ValueError(secret), 'invalid_device_or_options'),
                                (RuntimeError(secret), 'runtime_or_training_failure'),
                                (OSError(secret), 'unexpected_error')):
            with self.subTest(error=type(error).__name__), \
                 mock.patch.object(dbmoe, 'prepare_worker_device', side_effect=error):
                report = dbmoe.hardware_report('cpu')
            self.assertEqual(report['result'], 'failed')
            self.assertEqual(report['failure_category'], category)
            self.assertNotIn(secret, json.dumps(report))
            self.assertNotIn('checks', report)
        report = dbmoe.hardware_report(secret)
        self.assertEqual(report['requested_device'], 'invalid')
        self.assertNotIn(secret, json.dumps(report))

    def test_unavailable_backend_report_is_useful_without_credentials(self):
        with mock.patch.object(dbmoe, 'xpu_available', return_value=False), \
             mock.patch.object(dbmoe, 'prepare_worker_device') as prepare, \
             mock.patch.object(dbmoe, 'load_or_join', side_effect=AssertionError('credential')):
            report = dbmoe.hardware_report('xpu', include_hardware=True)
        self.assertEqual(report['stage'], 'device_selection')
        self.assertEqual(report['failure_category'], 'backend_unavailable')
        self.assertNotIn('hardware', report)
        prepare.assert_not_called()

    def test_versions_and_unknown_platform_strings_do_not_leak_suffixes(self):
        with mock.patch.object(torch, '__version__', '2.15.0+private_username_build'), \
             mock.patch.object(dbmoe.platform, 'system', return_value='private-host'), \
             mock.patch.object(dbmoe.platform, 'machine', return_value='/private/path'), \
             mock.patch.object(dbmoe, 'prepare_worker_device', return_value={}):
            report = dbmoe.hardware_report('cpu')
        self.assertEqual(report['environment'], {'os_family': 'other', 'architecture': 'other', 'torch_version': '2.15.0'})
        self.assertNotIn('private', json.dumps(report))
        for value, expected in [('2.15.0a0+local-build', '2.15.0a0'), ('12.6', '12.6'), ('/private/token', 'unknown')]:
            self.assertEqual(dbmoe.report_version(value), expected)

    def test_report_ids_are_per_run_not_stable_device_identifiers(self):
        with mock.patch.object(dbmoe, 'prepare_worker_device', return_value={}):
            a, b = dbmoe.hardware_report('cpu'), dbmoe.hardware_report('cpu')
        self.assertNotEqual(a['report_id'], b['report_id'])
        self.assertEqual(a['source_sha256'], b['source_sha256'])
        for key in ('host', 'hostname', 'ip', 'serial', 'uuid', 'machine_id', 'username', 'email', 'token', 'url'):
            self.assertNotIn(key, a)

    def test_hardware_name_rejects_path_email_and_multiline_values(self):
        props = argparse.Namespace(total_memory=8 * 2**30, is_integrated_gpu=True)
        for name in ('/home/alice/GPU', 'GPU\\private', 'user@example.test', 'GPU\nsecret'):
            with self.subTest(name=name), \
                 mock.patch.object(torch.xpu, 'get_device_name', return_value=name), \
                 mock.patch.object(torch.xpu, 'get_device_properties', return_value=props):
                report = dbmoe.report_hardware(dict(backend='xpu', torch_device='xpu:0'))
            self.assertEqual(report['gpu_model'], 'withheld')
            self.assertEqual(report['memory_kind'], 'shared')
            self.assertNotIn(name, json.dumps(report))

    def test_optional_gpu_fields_are_allowlisted(self):
        props = argparse.Namespace(total_memory=8 * 2**30, is_integrated_gpu=True, serial='SECRET', pci_id='PRIVATE')
        with mock.patch.object(torch.xpu, 'get_device_name', return_value='Intel Arc Graphics'), \
             mock.patch.object(torch.xpu, 'get_device_properties', return_value=props):
            report = dbmoe.report_hardware(dict(backend='xpu', torch_device='xpu:1'))
        self.assertEqual(report, dict(gpu_model='Intel Arc Graphics', memory_kind='shared', total_device_memory_gib=8.))

    def test_hardware_query_errors_do_not_leak(self):
        with mock.patch.object(torch.xpu, 'get_device_name', side_effect=RuntimeError('/private/token SECRET')):
            self.assertEqual(dbmoe.report_hardware(dict(backend='xpu', torch_device='xpu:0')), {'available': False})

    def test_invalid_thread_count_is_safe_and_finite_json(self):
        for count in (0, -1, 4097, True):
            report = dbmoe.hardware_report('cpu', threads=count)
            self.assertEqual(report['result'], 'failed')
            json.dumps(report, allow_nan=False)

    def test_successful_xpu_cap_is_reported_without_runtime_suffix(self):
        selection = dict(torch_device='xpu:1', device='xpu:1', backend='xpu')
        with mock.patch.object(dbmoe, 'resolve_worker_device', return_value=selection), \
             mock.patch.object(dbmoe, 'prepare_worker_device', return_value={'xpu_memory_fraction': .5, 'xpu_allocator_limit_gib': 4.}) as prepare, \
             mock.patch.object(torch.version, 'xpu', '2026.1.3+private', create=True):
            report = dbmoe.hardware_report('xpu:1', xpu_memory_fraction=.5)
        prepare.assert_called_once_with('xpu:1', check_cpu=True, xpu_memory_fraction=.5)
        self.assertEqual(report['environment']['runtime_version'], '2026.1.3')
        self.assertEqual(report['xpu_memory_fraction'], .5)
        self.assertNotIn('xpu_allocator_limit_gib', report)  # Otherwise total GPU memory can be inferred.
        self.assertNotIn('private', json.dumps(report))

    def test_absolute_allocator_budget_requires_hardware_opt_in(self):
        selection = dict(torch_device='xpu:1', device='xpu:1', backend='xpu')
        with mock.patch.object(dbmoe, 'resolve_worker_device', return_value=selection), \
             mock.patch.object(dbmoe, 'prepare_worker_device', return_value={'xpu_memory_fraction': .5, 'xpu_allocator_limit_gib': 4.}), \
             mock.patch.object(dbmoe, 'report_hardware', return_value={'memory_kind': 'shared'}):
            report = dbmoe.hardware_report('xpu:1', xpu_memory_fraction=.5, include_hardware=True)
        self.assertEqual(report['xpu_allocator_limit_gib'], 4.)

    def test_cli_file_is_local_and_never_overwritten(self):
        report = {'result': 'passed'}
        with tempfile.TemporaryDirectory() as directory, \
             mock.patch.object(dbmoe, 'hardware_report', return_value=report), \
             contextlib.redirect_stdout(io.StringIO()):
            path = Path(directory) / 'report.json'
            args = argparse.Namespace(device='cpu', threads=1, include_hardware=False, xpu_memory_fraction=None, output=str(path))
            dbmoe.hardware_report_cmd(args)
            self.assertEqual(json.loads(path.read_text()), report)
            path.write_text('PRESERVE')
            with self.assertRaisesRegex(SystemExit, 'unused writable filename'):
                dbmoe.hardware_report_cmd(args)
            self.assertEqual(path.read_text(), 'PRESERVE')

    def test_existing_symlink_is_not_followed_for_write(self):
        with tempfile.TemporaryDirectory() as directory, \
             mock.patch.object(dbmoe, 'hardware_report', return_value={'result': 'passed'}):
            target, link = Path(directory) / 'target', Path(directory) / 'link'
            target.write_text('PRESERVE')
            link.symlink_to(target)
            args = argparse.Namespace(device='cpu', threads=1, include_hardware=False, xpu_memory_fraction=None, output=str(link))
            with self.assertRaises(SystemExit):
                dbmoe.hardware_report_cmd(args)
            self.assertEqual(target.read_text(), 'PRESERVE')

    def test_failed_test_still_emits_json_and_nonzero_exit(self):
        output = io.StringIO()
        args = argparse.Namespace(device='xpu', threads=1, include_hardware=False, xpu_memory_fraction=None, output='-')
        with mock.patch.object(dbmoe, 'xpu_available', return_value=False), contextlib.redirect_stdout(output):
            with self.assertRaises(SystemExit) as raised:
                dbmoe.hardware_report_cmd(args)
        self.assertEqual(raised.exception.code, 1)
        self.assertEqual(json.loads(output.getvalue())['failure_category'], 'backend_unavailable')

    def test_cli_defaults_do_not_collect_optional_hardware(self):
        with mock.patch.object(dbmoe.sys, 'argv', ['dbmoe.py', 'hardware-report']), \
             mock.patch.object(dbmoe, 'hardware_report_cmd') as run:
            dbmoe.main()
        args = run.call_args.args[0]
        self.assertFalse(args.include_hardware)
        self.assertEqual(args.output, '-')
        self.assertFalse(hasattr(args, 'token'))
        self.assertFalse(hasattr(args, 'url'))


if __name__ == '__main__':
    unittest.main()
