"""CPU-only placement/failover proofs; no live services, weights or credentials."""
import copy
import socket
import threading
import time
import unittest
from unittest import mock

import torch
from test_routing_v4 import db, fixture, tokens


class ReplicaWorkers:
    """Real poll/serve protocol with a corrupt first choice and honest replicas."""
    def __init__(self, net, honest=True, corrupt_block=1):
        self.net, self.stop = net, threading.Event()
        self.threads, self.sockets, self.errors, self.jobs = [], [], [], []
        specs = [('bad', corrupt_block, True), ('other', 1 - corrupt_block, False)]
        if honest:
            specs.append(('replica', corrupt_block, False))
        for name, block, corrupt in specs:
            key = name + '#test'
            net.peers[key] = dict(owner=name, worker=name, open=True, target=block,
                                  ewma=0 if corrupt else 10000., hops=0, strikes=0,
                                  banned=False, snap=net.current, block=block, seen=time.time())
            a, b = socket.socketpair()
            self.sockets.extend([a, b])
            thread = threading.Thread(target=self.run, args=(name, block, corrupt, a), daemon=True)
            self.threads.append(thread)
            thread.start()
        deadline = time.monotonic() + 5
        while len(net.idle) < len(specs):
            if self.errors or time.monotonic() > deadline:
                self.close()
                raise AssertionError(self.errors or 'workers failed to park')
            time.sleep(.005)

    def run(self, name, block, corrupt, sock):
        try:
            assignment = dict(snap=self.net.current, block=block)
            blob, err = self.net.block_blob(assignment['snap'], block, name)
            assert not err, err
            held = db.load_inference_block(db.from_bytes(blob), assignment)
            result = None
            while not self.stop.is_set():
                reply = self.net.poll(name, name, dict(session='test', snap=held['snap'],
                                                       block=held['block'], result=result), sock)
                result = None
                if 'assign' in reply:
                    assignment = reply['assign']
                    blob, err = self.net.block_blob(assignment['snap'], assignment['block'], name)
                    assert not err, err
                    held = db.load_inference_block(db.from_bytes(blob), assignment)
                if 'job' in reply:
                    job = reply['job']
                    self.jobs.append((name, dict(job)))
                    result = db.serve_job(held, job)
                    if corrupt:
                        result['z'] = db.encode_floats(torch.ones(1, 1, held['d']) * 100)
                elif reply.get('idle'):
                    # A banned peer receives an immediate backoff response. Respect
                    # it rather than creating a CPU-bound fake polling client.
                    self.stop.wait(min(float(reply.get('retry_s', 0)), .05))
        except Exception as exc:
            self.errors.append(repr(exc))

    def close(self):
        self.stop.set()
        with self.net.cv:
            for slot in self.net.idle.values():
                slot.event.set()
        for thread in self.threads:
            thread.join(3)
        for sock in self.sockets:
            sock.close()
        assert not self.errors, self.errors
        assert not any(thread.is_alive() for thread in self.threads)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class PlacementTests(unittest.TestCase):
    def setUp(self):
        self.net = fixture()
        self.assignment = dict(snap=self.net.current, block=0)
        blob, err = self.net.block_blob(self.net.current, 0, 'test')
        self.assertIsNone(err)
        self.payload = db.from_bytes(blob)

    def test_matching_assignment_loads_exact_weights(self):
        held = db.load_inference_block(self.payload, self.assignment)
        self.assertEqual(held['snap'], self.assignment['snap'])
        self.assertEqual(held['block'], 0)
        for key, value in held['module'].state_dict().items():
            self.assertTrue(torch.equal(value, self.payload['block_state'][key]))
        for key, value in held['shared'].state_dict().items():
            self.assertTrue(torch.equal(value, self.payload['shared'][key]))
        self.assertFalse(held['module'].training)
        self.assertFalse(held['shared'].training)

    def test_wrong_snapshot_or_block_rejected_before_construction(self):
        for key in ('snap', 'block'):
            payload = dict(self.payload, **{key: self.payload[key] + 1})
            with self.subTest(key=key), mock.patch.object(db, 'Shared') as shared:
                with self.assertRaisesRegex(ValueError, 'does not match'):
                    db.load_inference_block(payload, self.assignment)
                shared.assert_not_called()

    def test_malformed_placement_ids_are_not_coerced(self):
        for key in ('snap', 'block'):
            for value in (None, True, False, '0', 0.0):
                with self.subTest(key=key, value=value):
                    with self.assertRaises(ValueError):
                        db.load_inference_block(dict(self.payload, **{key: value}), self.assignment)
                    with self.assertRaises(ValueError):
                        db.load_inference_block(self.payload, dict(self.assignment, **{key: value}))

    def test_invalid_container_and_block_range(self):
        for payload, assignment in ((None, self.assignment), (self.payload, [])):
            with self.assertRaises(ValueError):
                db.load_inference_block(payload, assignment)
        for block in (-1, self.payload['cfg']['blocks']):
            with self.assertRaisesRegex(ValueError, 'outside'):
                db.load_inference_block(dict(self.payload, block=block), dict(self.assignment, block=block))

    def test_failed_load_preserves_previous_held_block(self):
        held = db.load_inference_block(self.payload, self.assignment)
        previous = held
        broken = copy.deepcopy(self.payload)
        broken['block_state'].pop(next(iter(broken['block_state'])))
        with self.assertRaises(RuntimeError):
            held = db.load_inference_block(broken, self.assignment)
        self.assertIs(held, previous)


class FailoverTests(unittest.TestCase):
    def test_final_logits_and_top_sampler_replica_parity(self):
        for sampler, block in [('chain', 0), ('top', 1)]:
            with self.subTest(sampler=sampler):
                net = fixture()
                reference = list(net.generate('A', 2, seed=9, temp=.75, route='single', sampler=sampler))
                with ReplicaWorkers(net, corrupt_block=block):
                    actual = list(net.generate('A', 2, seed=9, temp=.75, route='distributed', sampler=sampler))
                self.assertEqual(tokens(actual), tokens(reference))
                self.assertEqual(actual[-1]['hops_coordinator'], 0)
                self.assertGreater(net.stats['audit_failures'], 0)

    def test_legacy_network_fallback_is_unchanged(self):
        net = fixture()
        reference = list(net.generate('A', 1, seed=9, temp=.75, route='single'))
        with ReplicaWorkers(net, honest=False):
            actual = list(net.generate('A', 1, seed=9, temp=.75, route='network'))
        self.assertEqual(tokens(actual), tokens(reference))
        self.assertGreater(actual[-1]['hops_coordinator'], 0)

    def test_corrupt_first_owner_retries_same_job_and_preserves_parity(self):
        net = fixture()
        reference = list(net.generate('A', 3, seed=17, temp=.75, route='single'))
        coordinator_before = net.stats['hops_coordinator']
        with ReplicaWorkers(net) as workers:
            actual = list(net.generate('A', 3, seed=17, temp=.75, route='distributed'))
            first = workers.jobs[0]
            self.assertEqual(first[0], 'bad')
            replacement = next(job for name, job in workers.jobs if name == 'replica')
            for key in ('snap', 'block', 'seed', 'ids', 'z', 'grid', 'final'):
                self.assertEqual(first[1][key], replacement[key], key)
            self.assertNotEqual(first[1]['id'], replacement['id'])
            self.assertGreater(net.peers['bad#test']['strikes'], 0)
        self.assertEqual(tokens(actual), tokens(reference))
        self.assertEqual(actual[-1]['state_hash'], reference[-1]['state_hash'])
        self.assertEqual(actual[-1]['hops_coordinator'], 0)
        self.assertEqual(net.stats['hops_coordinator'], coordinator_before)
        self.assertGreater(net.stats['retries'], 0)
        self.assertFalse(net.pending)
        self.assertFalse(net.active_snaps)

    def test_owner_loss_after_rejection_fails_closed(self):
        net = fixture()
        with ReplicaWorkers(net, honest=False):
            with self.assertRaisesRegex(RuntimeError, 'verification failed'):
                list(net.generate('A', 1, route='distributed'))
        self.assertEqual(net.stats['hops_coordinator'], 0)
        self.assertFalse(net.pending)
        self.assertFalse(net.active_snaps)

    def test_zero_retry_budget_does_not_use_replica(self):
        net = fixture()
        net.retries = 0
        with ReplicaWorkers(net) as workers:
            with self.assertRaisesRegex(RuntimeError, 'verification failed'):
                list(net.generate('A', 1, route='distributed'))
            self.assertEqual([name for name, _ in workers.jobs], ['bad'])
        self.assertEqual(net.stats['hops_coordinator'], 0)


if __name__ == '__main__':
    unittest.main()
