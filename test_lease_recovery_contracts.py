"""Focused bookkeeping contracts against actual coordinator source.

Run DBMOE_SOURCE=/path/to/dbmoe.py python3 test_contracts.py, or select a saved
source extraction with DBMOE_SNAPSHOT=snapshot-pending-candidate.json.
Only tensor/data storage is stubbed; no model, PyTorch, or network computation runs.
"""
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import random
import secrets
import threading
import time
from types import SimpleNamespace
import unittest

def load_actual_source():
    source_path = os.environ.get('DBMOE_SOURCE')
    if not source_path:
        return json.loads(Path(__file__).with_name(os.environ.get('DBMOE_SNAPSHOT', 'snapshot.json')).read_text())
    path = Path(source_path)
    raw = path.read_bytes()
    source = raw.decode('utf-8')
    lines = source.splitlines(keepends=True)
    coordinator = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'Coordinator')
    methods = {n.name: n for n in coordinator.body if isinstance(n, ast.FunctionDef)}
    segment = lambda node: ''.join(lines[node.lineno - 1:node.end_lineno]) + '\n'
    fields = []
    for node in methods['__init__'].body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if (isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name)
                    and target.value.id == 'self' and target.attr in ('pending_leases', 'pending_workers')):
                fields.append(segment(node))
    if not fields:
        raise ValueError('actual Coordinator initializer has no pending reservation fields')
    return dict(revision=hashlib.sha256(raw).hexdigest(), path=str(path.resolve()),
                lease_source=''.join(segment(methods[name]) for name in ('expire', 'lease', '_lease')),
                check_source=segment(methods['_check']), init_fields=''.join(fields))


SNAPSHOT = load_actual_source()

class LeaseCapacityError(RuntimeError):
    pass

class Data:
    def __init__(self, rows, width):
        self.rows, self.width = rows, width
    def size(self, axis):
        return self.rows if axis == 0 else self.width
    def __getitem__(self, idx):
        return Data(len(idx), self.width)
    def clone(self):
        return Data(self.rows, self.width)

class Source:
    def windows(self, rows, context):
        return Data(rows, context + 1)

class BlockedSource(Source):
    def __init__(self):
        self.entered, self.release = threading.Event(), threading.Event()
    def windows(self, rows, context):
        self.entered.set()
        if not self.release.wait(2):
            raise RuntimeError('test release timeout')
        return super().windows(rows, context)

class BrokenSource(Source):
    def windows(self, rows, context):
        raise RuntimeError('test data source failure')

class FirstBlockedSource(Source):
    def __init__(self):
        self.entered, self.release = threading.Event(), threading.Event()
        self.calls = 0
    def windows(self, rows, context):
        self.calls += 1
        if self.calls == 1:
            self.entered.set()
            if not self.release.wait(2):
                raise RuntimeError('test release timeout')
        return super().windows(rows, context)

class EmptyState:
    def state_dict(self):
        return {}

ns = dict(time=time, math=math, random=random, secrets=secrets, threading=threading,
          torch=SimpleNamespace(randperm=lambda n: list(range(n))),
          VERSION='bookkeeping-test', LeaseCapacityError=LeaseCapacityError)
actual_source = 'class Coordinator:\n' + SNAPSHOT['lease_source'] + SNAPSHOT['check_source']
exec(compile(actual_source, SNAPSHOT['path'] + '@' + SNAPSHOT['revision'], 'exec'), ns)
Coordinator = ns['Coordinator']

def fixture():
    C = Coordinator()
    C.lock = threading.RLock()
    C.pending_leases = {}
    C.pending_workers = set()
    if 'init_fields' in SNAPSHOT:
        del C.pending_leases
        del C.pending_workers
        exec(compile('def init_fields(self):\n' + SNAPSHOT['init_fields'], '<actual initializer fields>', 'exec'), ns)
        ns['init_fields'](C)
    C.max_identity_leases = C.max_open_leases = 256
    C.max_active_leases, C.reserved_leases = 512, 8
    C.open_lease_interval, C.last_open_lease = 0.0, {}  # Rate limiting is outside these capacity contracts.
    C.min_steps, C.steps, C.max_steps, C.batch, C.max_batch = 10, 10, 400, 2, 64
    C.lr, C.ttl, C.check_windows = 0.001, 1800, 6
    C.leases = {}
    C.meta = dict(block_updates=[0, 0], workers={})
    C.model = SimpleNamespace(cfg=dict(blocks=2, ctx=8), sigmas=[0.01, 0.1, 1.0],
                              shared=EmptyState(), blocks=[EmptyState(), EmptyState()])
    C.data = Source()
    C.is_open = lambda owner: owner.startswith('open')
    C.block_weights = lambda: [1.0, 1.0]
    C.events = []
    C.log = lambda **kw: C.events.append(kw)
    return C

def fill(C, owner, count):
    ids = []
    for i in range(count):
        lid = f'{owner}-{i}'
        C.leases[lid] = dict(worker=f'{owner}/node-{i}', owner=owner, session='old-session', block=i % 2,
                            t0=time.time(), ttl=1800, same=1, all=1, steps=10, batch=2)
        ids.append(lid)
    return ids

def request(C, owner='open-a', name='node-0', session='new-session'):
    return C.lease(f'{owner}/{name}', 10, 2, session, owner=owner)

def full_network():
    C = fixture()
    own = fill(C, 'open-a', 256)
    other = fill(C, 'open-b', 248) + fill(C, 'trusted', 8)
    return C, own, other

def count_payload(C, steps=10):
    L = dict(steps=steps, batch=2, block=0)
    up = dict(losses=[1.0] * steps, seconds=1.0, d_shared={}, d_block={},
              tokens=steps * L['batch'] * C.model.cfg['ctx'])
    return L, up


class FollowupContracts(unittest.TestCase):
    def test_exact_loss_count_required(self):
        C = fixture()
        for n in (0, 1, 9, 11):
            with self.subTest(loss_count=n):
                L, up = count_payload(C)
                up['losses'] = [1.0] * n
                self.assertIsNotNone(C._check(up, L))
        L, up = count_payload(C)
        self.assertIsNone(C._check(up, L))

    def test_optional_tokens_are_exact_integers(self):
        C = fixture()
        L, up = count_payload(C)
        expected = up['tokens']
        for value in (expected + 1, float(expected), str(expected), None, False):
            with self.subTest(tokens=value):
                up['tokens'] = value
                self.assertIsNotNone(C._check(up, L))
        up['tokens'] = expected
        self.assertIsNone(C._check(up, L))
        del up['tokens']
        self.assertIsNone(C._check(up, L))

    def test_bool_not_accepted_even_when_numerically_equal(self):
        C = fixture()
        C.model.cfg['ctx'] = 1
        L, up = count_payload(C, steps=1)
        L['batch'] = 1
        up['tokens'] = True  # True == 1, but is not an integer report.
        self.assertIsNotNone(C._check(up, L))

    def test_changed_session_restart_at_contributor_cap_reclaims_only_own_name(self):
        for owner in ('open-a', 'trusted'):
            with self.subTest(owner=owner):
                C = fixture()
                old = fill(C, owner, 256)
                other_owner = fill(C, 'open-other', 1)
                result = request(C, owner, session='restarted-process')
                self.assertNotIn(old[0], C.leases)
                self.assertTrue(set(old[1:] + other_owner) <= C.leases.keys())
                self.assertIn(result['lease'], C.leases)
                self.assertEqual(len(C.leases), 257)
                self.assertFalse(C.pending_leases)

    def test_changed_session_also_reclaims_old_job_below_cap(self):
        C = fixture()
        old = fill(C, 'open-a', 2)
        result = request(C, session='restarted-process')
        self.assertNotIn(old[0], C.leases)
        self.assertIn(old[1], C.leases)
        self.assertEqual(len(C.leases), 2)
        self.assertEqual(C.leases[result['lease']]['session'], 'restarted-process')

    def test_same_session_net_zero_replacement_can_use_own_reserved_capacity(self):
        C, own, other = full_network()
        result = request(C, session='old-session')
        self.assertEqual(len(C.leases), 512)
        self.assertNotIn(own[0], C.leases)
        self.assertTrue(set(own[1:] + other) <= C.leases.keys())
        self.assertIn(result['lease'], C.leases)

    def test_changed_session_net_zero_replacement_at_full_network(self):
        C, own, other = full_network()
        result = request(C, session='restarted-process')
        self.assertEqual(len(C.leases), 512)
        self.assertNotIn(own[0], C.leases)
        self.assertTrue(set(own[1:] + other) <= C.leases.keys())
        self.assertIn(result['lease'], C.leases)

    def test_new_machine_at_capacity_does_not_evict_any_existing_machine(self):
        C, own, other = full_network()
        before = set(C.leases)
        with self.assertRaises(LeaseCapacityError):
            request(C, name='new-machine')
        self.assertEqual(set(C.leases), before)
        self.assertFalse(C.pending_leases)

    def test_reserved_slots_remain_reserved_for_new_public_allocations(self):
        C = fixture()
        fill(C, 'open-a', 256)
        fill(C, 'open-b', 248)
        before = set(C.leases)
        with self.assertRaises(LeaseCapacityError):
            request(C, owner='open-new', name='new-machine')
        self.assertEqual(set(C.leases), before)
        result = request(C, owner='trusted-new', name='new-machine')
        self.assertIn(result['lease'], C.leases)
        self.assertEqual(len(C.leases), 505)

    def test_pending_duplicate_cannot_overcommit_contributor_or_global_capacity(self):
        C, own, other = full_network()
        blocked = C.data = BlockedSource()
        outcomes, errors = [], []
        def first():
            try:
                outcomes.append(request(C, session='old-session'))
            except Exception as exc:
                errors.append(repr(exc))
        thread = threading.Thread(target=first)
        thread.start()
        try:
            self.assertTrue(blocked.entered.wait(1))
            self.assertEqual(C.pending_leases.get('open-a'), 1)
            before = set(C.leases)
            with self.assertRaises(LeaseCapacityError):
                request(C, session='old-session')
            self.assertEqual(set(C.leases), before)
            with self.assertRaises(LeaseCapacityError):
                request(C, owner='trusted-new', name='new-machine')
            self.assertEqual(set(C.leases), before)
        finally:
            blocked.release.set()
            thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(len(outcomes), 1)
        self.assertEqual(len(C.leases), 512)
        self.assertTrue(set(own[1:] + other) <= C.leases.keys())
        self.assertFalse(C.pending_leases)

    def test_replacement_source_failure_releases_reservation_and_keeps_old_work(self):
        C = fixture()
        old = fill(C, 'open-a', 256)
        before = set(C.leases)
        C.data = BrokenSource()
        with self.assertRaisesRegex(RuntimeError, 'test data source failure'):
            request(C, session='restarted-process')
        self.assertFalse(C.pending_leases)
        self.assertFalse(getattr(C, 'pending_workers', set()))
        self.assertEqual(set(C.leases), before)
        C.data = Source()
        result = request(C, session='restarted-process')
        self.assertNotIn(old[0], C.leases)
        self.assertIn(result['lease'], C.leases)
        self.assertEqual(len(C.leases), 256)

    def test_overlapping_restart_below_capacity_reserves_only_that_worker(self):
        C = fixture()
        old = fill(C, 'open-a', 2)
        blocked = C.data = FirstBlockedSource()
        outcomes, errors = [], []
        def first():
            try:
                outcomes.append(request(C, session='delayed-old-process'))
            except Exception as exc:
                errors.append(repr(exc))
        thread = threading.Thread(target=first)
        thread.start()
        try:
            self.assertTrue(blocked.entered.wait(1))
            before = set(C.leases)
            with self.assertRaises(LeaseCapacityError):
                request(C, session='restarted-process')
            self.assertEqual(set(C.leases), before)
            self.assertEqual(blocked.calls, 1)
            independent = request(C, name='another-machine', session='independent')
            self.assertIn(independent['lease'], C.leases)
            self.assertEqual(C.pending_leases, {'open-a': 1})
        finally:
            blocked.release.set()
            thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(len(outcomes), 1)
        self.assertIn(outcomes[0]['lease'], C.leases)
        self.assertIn(independent['lease'], C.leases)
        self.assertIn(old[1], C.leases)
        self.assertFalse(C.pending_leases)
        self.assertFalse(getattr(C, 'pending_workers', set()))
        restarted = request(C, session='restarted-process')
        self.assertIn(restarted['lease'], C.leases)
        self.assertNotIn(outcomes[0]['lease'], C.leases)
        self.assertIn(independent['lease'], C.leases)
        self.assertFalse(C.pending_leases)
        self.assertFalse(getattr(C, 'pending_workers', set()))


if __name__ == '__main__':
    print('SOURCE_REVISION ' + SNAPSHOT['revision'], flush=True)
    unittest.main(verbosity=2)
