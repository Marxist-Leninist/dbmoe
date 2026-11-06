"""How much an update counts when jobs overlap on a block: in proportion to their work."""
import unittest
import unittest.mock
from test_gate import coordinator, state

import torch
import dbmoe


class Lease(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.C = coordinator()

    def lease(self, name, steps, batch, block):
        with unittest.mock.patch.object(dbmoe, 'choose_training_block', return_value=block):
            L = self.C._lease(f'trusted/{name}', steps, batch, 's' + name, owner='trusted')
        return self.C.leases[L['lease']], L

    def test_equal_jobs_split_evenly(self):
        leases = [self.lease(f'w{i}', 20, 4, 0)[0] for i in range(4)]
        for v in leases:
            self.assertEqual(v['same'], 4)

    def test_equal_jobs_overlap_in_order(self):
        a, _ = self.lease('a', 20, 4, 0)
        self.assertEqual(a['same'], 1)
        b, _ = self.lease('b', 20, 4, 0)
        self.assertEqual((a['same'], b['same']), (2, 2))
        c, _ = self.lease('c', 20, 4, 0)
        self.assertEqual((a['same'], b['same'], c['same']), (3, 3, 3))

    def test_jobs_share_in_proportion_to_their_work(self):
        big, _ = self.lease('big', 400, 64, 0)
        small, _ = self.lease('small', 20, 8, 0)
        n, m = 400 * 64, 20 * 8
        self.assertAlmostEqual(1 / big['same'], n / (n + m))
        self.assertAlmostEqual(1 / small['same'], m / (n + m))
        self.assertAlmostEqual(1 / big['same'] + 1 / small['same'], 1)

    def test_shares_of_jobs_that_all_overlap_sum_to_one(self):
        leases = [self.lease(f'w{i}', s, b, 0)[0] for i, (s, b) in enumerate([(400, 64), (200, 64), (60, 8), (40, 8), (2, 1)])]
        self.assertAlmostEqual(sum(1 / v['same'] for v in leases), 1)

    def test_many_small_jobs_add_up(self):
        big, _ = self.lease('big', 400, 64, 0)
        for i in range(160):
            self.lease(f's{i}', 10, 16, 0)              # as much work again in small pieces
        self.assertAlmostEqual(big['same'], 2)

    def test_a_job_that_ended_before_this_one_began_does_not_count(self):
        a, La = self.lease('a', 8, 4, 0)
        self.C.gate_all = False
        self.assertEqual(self.C.update('trusted/a', dbmoe.train_lease(La, 'cpu'))[0], 200)
        b, _ = self.lease('b', 8, 4, 0)
        self.assertEqual(b['same'], 1)

    def test_other_blocks_do_not_count(self):
        a, _ = self.lease('a', 20, 4, 0)
        b, _ = self.lease('b', 400, 64, 1)
        self.assertEqual(a['same'], 1)
        self.assertEqual(b['same'], 1)
        self.assertEqual(a['all'], 2)

    def test_merge_uses_the_fractional_weight(self):
        big, Lb = self.lease('big', 8, 4, 0)
        self.lease('small', 2, 4, 0)
        self.C.gate_all = False
        before = state(self.C)
        up = dbmoe.train_lease(Lb, 'cpu')
        want = 8 * 4 / (8 * 4 + 2 * 4)
        self.assertEqual(self.C.update('trusted/big', up)[0], 200)
        k = next(iter(up['d_block']))
        got = self.C.model.blocks[0].state_dict()[k] - before[f'blocks.0.{k}']
        self.assertTrue(torch.allclose(got, up['d_block'][k] * want, atol=1e-6))


class Shared(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.C = coordinator()
        self.C.gate_all = False

    def round_delta(self, works):
        """Bank a delta of value w, of work w, for each w; the round's change to the first shared parameter."""
        one = {k: torch.ones_like(v) for k, v in self.C.model.shared.state_dict().items()}
        self.C.shared_round = 10 ** 6
        for w in works:
            self.C._bank_shared(0, {k: v * w for k, v in one.items()}, w)
        bank, self.C.shared_bank = self.C.shared_bank, {}
        before = {k: v.clone() for k, v in self.C.model.shared.state_dict().items()}
        with unittest.mock.patch.object(dbmoe, 'round_measure', return_value={0: (1.0, 1.0)}):
            self.C._commit_round(bank)
        k = next(iter(before))
        return float((self.C.model.shared.state_dict()[k] - before[k]).flatten()[0])

    def test_equal_deltas_are_their_mean_as_before(self):
        wt = self.C.block_weights()
        self.assertAlmostEqual(self.round_delta([5, 5, 5]), wt[0] / sum(wt) * 5, places=5)

    def test_deltas_count_in_proportion_to_their_work(self):
        wt = self.C.block_weights()
        want = wt[0] / sum(wt) * (100 * 100 + 1 * 1) / 101
        self.assertAlmostEqual(self.round_delta([100, 1]), want, places=4)

    def test_a_bank_made_without_work_is_a_plain_mean(self):
        wt = self.C.block_weights()
        one = {k: torch.ones_like(v) * 3 for k, v in self.C.model.shared.state_dict().items()}
        before = {k: v.clone() for k, v in self.C.model.shared.state_dict().items()}
        with unittest.mock.patch.object(dbmoe, 'round_measure', return_value={0: (1.0, 1.0)}):
            self.C._commit_round({0: dict(n=3, sum=one)})
        k = next(iter(before))
        self.assertAlmostEqual(float((self.C.model.shared.state_dict()[k] - before[k]).flatten()[0]), wt[0] / sum(wt), places=5)


if __name__ == '__main__':
    unittest.main()
