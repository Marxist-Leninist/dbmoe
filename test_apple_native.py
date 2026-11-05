"""Numerical contract only. These tests do not run Swift, Metal or an Apple app."""
import importlib.util
import json
import math
from pathlib import Path
import unittest

import torch

ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('native_reference', ROOT / 'apple/generate_parity_fixture.py')
GENERATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GENERATOR)


class NativeNumericalReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = json.loads((ROOT / 'apple/parity.json').read_text())
        cls.actual = GENERATOR.generate(ROOT / 'dbmoe.py')

    def test_fixture_contains_only_finite_json(self):
        json.dumps(self.fixture, allow_nan=False)
        self.assertEqual(self.fixture['format'], 'dbmoe.native.parity.v1')

    def test_fixture_has_explicit_noise_and_both_experts(self):
        self.assertEqual(self.fixture['batch'], self.actual['batch'])
        self.assertEqual(self.fixture['sigma'], self.actual['sigma'])
        self.assertEqual(self.fixture['epsilon'], self.actual['epsilon'])
        self.assertEqual(self.fixture['cfg'], self.actual['cfg'])
        self.assertEqual(self.fixture['expected']['expert_fractions'][0]['data'], [.5, .5])

    def test_losses_gradients_and_adamw_step_replay(self):
        expected, actual = self.fixture['expected'], self.actual['expected']
        for name in ('loss', 'ce', 'auxiliary', 'gradient_norm_before_clip'):
            self.assertTrue(math.isclose(expected[name], actual[name], rel_tol=1e-5, abs_tol=1e-6), name)
        for section in ('initial',):
            self.assertEqual(self.fixture[section], self.actual[section])
        for section in ('gradients', 'clipped_gradients', 'after_step', 'deltas'):
            self.assertEqual(set(expected[section]), set(actual[section]))
            for name, ref in expected[section].items():
                got = actual[section][name]
                if ref is None:
                    self.assertIsNone(got, name)
                    continue
                self.assertEqual(ref['shape'], got['shape'], name)
                self.assertEqual(ref['dtype'], got['dtype'], name)
                torch.testing.assert_close(torch.tensor(ref['data']), torch.tensor(got['data']),
                                           rtol=1e-5, atol=1e-6, msg=section + ': ' + name)
        for name in ('denoised', 'logits'):
            torch.testing.assert_close(torch.tensor(expected[name]['data']), torch.tensor(actual[name]['data']),
                                       rtol=1e-5, atol=1e-6, msg=name)

    def test_attention_and_clipping_are_exercised(self):
        expected = self.fixture['expected']
        self.assertGreater(expected['gradient_norm_before_clip'], 1.)
        qkv = expected['gradients']['block.layers.0.qkv.weight']['data']
        self.assertGreater(max(abs(x) for x in qkv), 1e-8)
        deltas = expected['deltas']['block.layers.0.qkv.weight']['data']
        self.assertGreater(max(abs(x) for x in deltas), 1e-8)

    def test_optional_skewed_fixture_exercises_unequal_usage(self):
        result = GENERATOR.generate(ROOT / 'dbmoe.py', skewed=True)
        expected = result['expected']
        self.assertEqual(sorted(expected['expert_fractions'][0]['data']), [.25, .75])
        self.assertGreater(abs(expected['auxiliary'] - 1.), 1e-3)
        self.assertAlmostEqual(expected['loss'], 3.6444308757781982, places=5)
        json.dumps(result, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
