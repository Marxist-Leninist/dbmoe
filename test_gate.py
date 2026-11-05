"""The update test: honest training passes, empty uploads and noise don't, and what is tested is what goes in."""
import random
import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
import dbmoe


CFG = dbmoe.make_cfg(dict(d=32, heads=2, layers_per_block=1, blocks=2, experts=2, ffn=64, ctx=32))
TEXT = torch.tensor(list(b'the quick brown fox jumps over the lazy dog. ' * 400), dtype=torch.uint8)


def windows(n, seed):
    start = torch.randint(0, TEXT.numel() - CFG['ctx'] - 1, (n,), generator=torch.Generator().manual_seed(seed))
    return torch.stack([TEXT[i:i + CFG['ctx'] + 1] for i in start.tolist()])


def lease(model, block=1, steps=40, batch=4):
    return dict(version=dbmoe.VERSION, lease='t', block=block, block_version=0, cfg=model.cfg, sigmas=model.sigmas,
                shared={k: v.clone() for k, v in model.shared.state_dict().items()},
                block_state={k: v.clone() for k, v in model.blocks[block].state_dict().items()},
                steps=steps, batch=batch, lr=3e-3, ttl=60, data=windows(steps * batch, 1))


def measure(model, L, d_shared, d_block, margin=0.002):
    b = L['block']
    return dbmoe.gate_measure(model.cfg, model.sigmas, windows(8, 2), b, 0, L['shared'], L['block_state'], L['shared'] and model.blocks[0].state_dict(),
                              d_shared, d_block, 1.0, 1.0, L['data'][:6], margin)


class UpdateTestTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = dbmoe.Model(CFG)
        self.L = lease(self.model)

    def test_honest_training_learns_from_its_own_text(self):
        up = dbmoe.train_lease(self.L, 'cpu')
        m = measure(self.model, self.L, up['d_shared'], up['d_block'])
        self.assertLess(m['trained_after'], m['trained_before'])
        self.assertIsNotNone(m['before'])       # went on to the held-out stage
        self.assertIsNotNone(m['other_after'])

    def test_empty_upload_stops_at_the_first_stage(self):
        zero = lambda sd: {k: torch.zeros_like(v) for k, v in sd.items()}
        m = measure(self.model, self.L, zero(self.L['shared']), zero(self.L['block_state']))
        self.assertEqual(m['trained_after'], m['trained_before'])
        self.assertIsNone(m['before'])

    def test_noise_stops_at_the_first_stage(self):
        noise = lambda sd: {k: torch.randint(0, 2, v.shape).float() * 0.1 - 0.05 for k, v in sd.items()}
        m = measure(self.model, self.L, noise(self.L['shared']), noise(self.L['block_state']))
        self.assertGreater(m['trained_after'], m['trained_before'])
        self.assertIsNone(m['before'])

    def test_not_a_number_stops_at_the_first_stage(self):
        nan = lambda sd: {k: torch.full_like(v, float('nan')) for k, v in sd.items()}
        m = measure(self.model, self.L, nan(self.L['shared']), nan(self.L['block_state']))
        self.assertIsNone(m['before'])

    def test_a_harmful_round_measures_worse_and_a_real_one_does_not(self):
        noise = lambda sd: {k: torch.randint(0, 2, v.shape).float() * 0.1 - 0.05 for k, v in sd.items()}
        zero = lambda sd: {k: torch.zeros_like(v) for k, v in sd.items()}
        x, blocks = windows(8, 2), {1: self.L['block_state']}
        bad = dbmoe.round_measure(self.model.cfg, self.model.sigmas, x, self.L['shared'], blocks, noise(self.L['shared']))
        self.assertGreater(bad[1][1], bad[1][0])
        same = dbmoe.round_measure(self.model.cfg, self.model.sigmas, x, self.L['shared'], blocks, zero(self.L['shared']))
        self.assertEqual(same[1][0], same[1][1])


class Source:
    def __init__(self):
        self.seed = 100

    def windows(self, n, T):
        self.seed += 1
        return windows(n, self.seed)


def coordinator(seed=0):
    random.seed(seed)
    torch.manual_seed(seed)
    C = dbmoe.Coordinator.__new__(dbmoe.Coordinator)
    C.model = dbmoe.Model(CFG)
    C.lr, C.steps, C.batch, C.ttl = 3e-3, 40, 4, 1800
    C.max_steps, C.max_batch, C.min_steps = 400, 64, 1
    C.lock = threading.RLock()
    C.leases, C.data = {}, Source()
    C.gate_all, C.gate_tol, C.learn_margin, C.gate_strict = True, 0.03, 0.002, False
    C.gate_x, C.gate_slots, C.gate_recent, C.gate_drift, C.check_windows = windows(12, 999), threading.BoundedSemaphore(2), [], {}, 6
    C.shared_bank, C.shared_round, C.round_busy = {}, 12, False
    C.block_gates = dbmoe.collections.defaultdict(threading.Lock)
    C.meta = dict(updates=0, tokens=0, block_updates=[0] * CFG['blocks'], block_ce=[None] * CFG['blocks'], workers={},
                  evals=[], rejected=0, gate=dict(passed=0, refused=0), worker_alias_salt='test')
    C.events, C.strikes = [], []
    C.log = lambda **kw: C.events.append(kw)
    C.is_open = lambda owner: owner.startswith('open')
    C.enrol = SimpleNamespace(strike=lambda owner, weight=1: C.strikes.append((owner, weight)), passed=lambda owner: None)
    C.save = lambda *a, **kw: None
    C.snap_every, C.eval_every, C.last_save, C.eval_busy, C.eval_started = 0, 0, time.time(), False, 0
    return C


def grant(C, worker='trusted/node'):
    return C._lease(worker, 40, 4, 's' + str(random.random()), owner=worker.split('/')[0])


def state(C):
    return {k: v.clone() for k, v in C.model.state_dict().items()}


def zero(L):
    return dict(lease=L['lease'], block=L['block'], losses=[1.0] * L['steps'], seconds=1.0, d_shared={k: torch.zeros_like(v) for k, v in L['shared'].items()},
                d_block={k: torch.zeros_like(v) for k, v in L['block_state'].items()})


class CoordinatorTests(unittest.TestCase):
    def assertSame(self, a, b):
        for k in a:
            self.assertTrue(torch.equal(a[k], b[k]), k)

    def test_nothing_is_taken_on_trust_whoever_sends_it(self):
        for worker in ('trusted/node', 'open-1/node'):
            C = coordinator()
            before = state(C)
            code, body = C.update(worker, zero(grant(C, worker)))
            self.assertEqual(code, 422, body)
            self.assertEqual(C.meta['updates'], 0)
            self.assertSame(before, state(C))
            code, body = C.update(worker, dbmoe.train_lease(grant(C, worker), 'cpu'))
            self.assertEqual(code, 200, body)
            self.assertEqual(C.meta['updates'], 1)

    def test_a_block_takes_its_updates_one_at_a_time(self):
        C = coordinator()
        C.block_weights = lambda: [0.0001, 1.0]         # both go to block 1
        ups = [(w, dbmoe.train_lease(grant(C, w), 'cpu')) for w in ('trusted/a', 'trusted/b')]
        self.assertEqual({u['block'] for _, u in ups}, {1})
        seen, real, inside = [], dbmoe.gate_measure, threading.Semaphore(1)

        def measure(*a, **kw):
            self.assertTrue(inside.acquire(blocking=False), 'two tests at once on one block')
            seen.append(C.meta['updates'])
            time.sleep(0.2)
            inside.release()
            return real(*a, **kw)

        with mock.patch.object(dbmoe, 'gate_measure', side_effect=measure):
            ts = [threading.Thread(target=C.update, args=u) for u in ups]
            [t.start() for t in ts]
            [t.join() for t in ts]
        self.assertEqual(seen, [0, 1])                  # the second was tested on a model that already held the first

    def test_an_update_merges_at_the_weight_it_was_tested_at(self):
        C = coordinator()
        L = grant(C)
        up = dbmoe.train_lease(L, 'cpu')
        before, real = state(C), dbmoe.gate_measure

        def measure(*a, **kw):
            grant(C, 'trusted/other') if L['block'] == 0 else None
            C.leases[L['lease']]['same'] = 5            # more leases on this block arrive while it is under test
            return real(*a, **kw)

        with mock.patch.object(dbmoe, 'gate_measure', side_effect=measure):
            self.assertEqual(C.update('trusted/node', up)[0], 200)
        k = next(iter(up['d_block']))
        got = C.model.blocks[L['block']].state_dict()[k] - before[f"blocks.{L['block']}.{k}"]
        self.assertTrue(torch.allclose(got, up['d_block'][k], atol=1e-6))

    def test_duplicate_upload_counts_once(self):
        C = coordinator()
        up = dbmoe.train_lease(grant(C), 'cpu')
        self.assertEqual(C.update('trusted/node', up)[0], 200)
        self.assertEqual(C.update('trusted/node', up)[0], 409)
        self.assertEqual(C.meta['updates'], 1)

    def test_strikes_are_for_the_worker_not_the_network(self):
        C = coordinator()
        C.gate_strict = True
        self.assertEqual(C.update('open-1/node', zero(grant(C, 'open-1/node')))[0], 422)
        self.assertEqual(C.strikes, [])
        C.gate_strict = False
        self.assertEqual(C.update('open-1/node', zero(grant(C, 'open-1/node')))[0], 422)
        self.assertEqual(C.strikes, [('open-1', 0.5)])  # a near miss
        L = grant(C, 'open-1/node')
        with mock.patch.object(dbmoe, 'gate_measure', return_value=dict(trained_before=1.0, trained_after=0.9995, before=None, after=None)):
            self.assertEqual(C.update('open-1/node', zero(L))[0], 422)
        self.assertEqual(len(C.strikes), 1)             # it learned, just below the margin: refused, no strike
        L = grant(C, 'open-1/node')
        junk = {k: torch.randint(0, 2, v.shape).float() * 0.8 - 0.4 for k, v in L['block_state'].items()}
        self.assertEqual(C.update('open-1/node', dict(zero(L), d_block=junk))[0], 422)
        self.assertEqual(C.strikes[-1], ('open-1', 1))

    def test_a_round_goes_in_only_if_it_passes_its_own_test(self):
        C = coordinator()
        L = grant(C)
        noise = {k: torch.randint(0, 2, v.shape).float() * 8 - 4 for k, v in L['shared'].items()}
        before = state(C)
        C._commit_round({L['block']: dict(n=1, sum=noise)})
        self.assertEqual(C.meta['gate']['rounds_refused'], 1)
        self.assertEqual(C.meta.get('shared_rounds', 0), 0)
        self.assertSame(before, state(C))
        self.assertFalse(C.round_busy)
        up = dbmoe.train_lease(L, 'cpu')
        C._commit_round({L['block']: dict(n=1, sum=up['d_shared'])})
        self.assertEqual(C.meta['shared_rounds'], 1)
        k = next(iter(up['d_shared']))
        got = C.model.shared.state_dict()[k] - before['shared.' + k]
        self.assertTrue(torch.allclose(got, up['d_shared'][k] * 0.5, atol=1e-6))       # equal block weights, two blocks
        self.assertFalse(any(g.locked() for g in C.block_gates.values()))

    def test_helpful_shared_part_cannot_hide_a_harmful_block_part(self):
        C = coordinator()
        real = dbmoe.train_lease(grant(C), 'cpu')
        L = C.leases[real['lease']]
        x = C.gate_x
        before = {k: v.clone() for k, v in C.model.blocks[L['block']].state_dict().items()}
        bad = {k: torch.randint(0, 2, v.shape).float() * 0.2 - 0.1 for k, v in before.items()}
        sh = {k: v.clone() for k, v in C.model.shared.state_dict().items()}
        zero_shared = {k: torch.zeros_like(v) for k, v in sh.items()}
        m = dbmoe.gate_measure(C.model.cfg, C.model.sigmas, x, L['block'], None, sh, before, None, real['d_shared'], real['d_block'], 0.5, 1.0, L['check'], -1.0)
        joint = dbmoe.gate_measure(C.model.cfg, C.model.sigmas, x, L['block'], None, sh, before, None, zero_shared, real['d_block'], 0.0, 1.0, L['check'], -1.0)
        self.assertEqual(m['after'], joint['after'])    # 'after' is the block part alone, whatever the shared part is
        harm = dbmoe.gate_measure(C.model.cfg, C.model.sigmas, x, L['block'], None, sh, before, None, real['d_shared'], bad, 0.5, 1.0, L['check'], -1.0)
        self.assertGreater(harm['after'], harm['before'])
        self.assertLessEqual(harm['shared_after'], harm['after'])

    def test_a_full_identity_gets_told_to_retry_and_nobody_loses_a_job(self):
        C = coordinator()
        C.is_open = lambda owner: True
        C.pending_leases, C.max_open_leases, C.max_identity_leases, C.max_active_leases, C.reserved_leases = {}, 2, 2, 64, 8
        C.open_lease_interval, C.last_open_lease = 0.0, {}
        a = C.lease('open-1/a', 2, 1, 's', owner='open-1')['lease']
        b = C.lease('open-1/b', 2, 1, 's', owner='open-1')['lease']
        with self.assertRaises(dbmoe.LeaseCapacityError):
            C.lease('open-1/c', 2, 1, 's', owner='open-1')
        self.assertEqual(set(C.leases), {a, b})
        again = C.lease('open-1/a', 2, 1, 's', owner='open-1')['lease']     # replacing its own job is always allowed
        self.assertEqual(set(C.leases), {again, b})

    def test_a_restarted_machine_can_reclaim_its_old_job_at_the_cap(self):
        C = coordinator()
        C.is_open = lambda owner: True
        C.pending_leases, C.max_open_leases, C.max_identity_leases, C.max_active_leases, C.reserved_leases = {}, 2, 2, 64, 8
        C.open_lease_interval, C.last_open_lease = 0.0, {}
        a = C.lease('open-1/a', 2, 1, 'before', owner='open-1')['lease']
        b = C.lease('open-1/b', 2, 1, 'before', owner='open-1')['lease']
        again = C.lease('open-1/a', 2, 1, 'after', owner='open-1')['lease']
        self.assertEqual(set(C.leases), {again, b})
        self.assertNotIn(a, C.leases)

    def test_replacing_a_job_does_not_need_room_for_a_new_one(self):
        C = coordinator()
        C.is_open = lambda owner: owner.startswith('open')
        C.pending_leases, C.max_open_leases, C.max_identity_leases, C.max_active_leases, C.reserved_leases = {}, 8, 8, 4, 2
        C.open_lease_interval, C.last_open_lease = 0.0, {}
        C.lease('open-1/a', 2, 1, 's', owner='open-1')
        C.lease('open-1/b', 2, 1, 's', owner='open-1')
        with self.assertRaises(dbmoe.LeaseCapacityError):       # the last two slots are kept for tokens.json identities
            C.lease('open-1/c', 2, 1, 's', owner='open-1')
        C.lease('trusted/x', 2, 1, 's', owner='trusted')
        C.lease('trusted/y', 2, 1, 's', owner='trusted')
        self.assertEqual(len(C.leases), 4)
        C.lease('open-1/a', 2, 1, 's', owner='open-1')          # full, but this takes no extra slot
        self.assertEqual(len(C.leases), 4)

    def test_a_restart_at_the_global_cap_keeps_its_old_job_unless_it_gets_a_new_one(self):
        C = coordinator()
        C.is_open = lambda owner: owner.startswith('open')
        C.pending_leases, C.max_open_leases, C.max_identity_leases, C.max_active_leases, C.reserved_leases = {}, 8, 8, 3, 1
        C.open_lease_interval, C.last_open_lease = 0.0, {}
        a = C.lease('open-1/a', 2, 1, 'before', owner='open-1')['lease']
        C.lease('trusted/x', 2, 1, 's', owner='trusted')
        C.lease('trusted/y', 2, 1, 's', owner='trusted')
        with self.assertRaises(dbmoe.LeaseCapacityError):       # a stranger gets no room
            C.lease('open-2/z', 2, 1, 's', owner='open-2')
        self.assertIn(a, C.leases)
        again = C.lease('open-1/a', 2, 1, 'after', owner='open-1')['lease']     # takes no extra room, so the reserve doesn't apply
        self.assertNotIn(a, C.leases)
        self.assertIn(again, C.leases)
        self.assertEqual(len(C.leases), 3)

    def test_an_upload_must_account_for_every_step_of_its_job(self):
        C = coordinator()
        for change in (dict(losses=[1.0]), dict(tokens=1), dict(tokens=True)):
            L = grant(C)
            up = dbmoe.train_lease(L, 'cpu')
            before = state(C)
            code, body = C.update('trusted/node', dict(up, **change))
            self.assertEqual(code, 422, body)
            self.assertEqual(C.meta['updates'], 0)
            self.assertSame(before, state(C))

    def test_compute_follows_what_is_left_to_learn(self):
        C = coordinator()
        C.meta['evals'] = [dict(per_block_ce=[0.02, 2.0])]
        got = [0, 0]
        for i in range(120):
            L = C._lease(f'trusted/w{i}', 2, 1, 's', owner='trusted')
            got[L['block']] += 1
            C.leases.pop(L['lease'])
        self.assertGreater(got[1], got[0] * 3)
        self.assertGreater(got[0], 0)                   # nothing starves


if __name__ == '__main__':
    unittest.main()
