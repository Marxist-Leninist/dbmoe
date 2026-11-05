"""Leaving removes every observed identity without removing accepted totals."""
import threading, types, unittest
import dbmoe as db

class LeaveIdentities(unittest.TestCase):
    def setUp(self):
        self.c = c = db.Coordinator.__new__(db.Coordinator)
        c.lock = threading.RLock()
        c.meta = {'workers': {}, 'worker_alias_salt': 'test-only', 'updates': 99, 'tokens': 1234}
        c.leases = {}
        c.enrol = None
        c.save = lambda: None
        c.log = lambda **kw: None
        c.net = n = db.NetInfer.__new__(db.NetInfer)
        n.cv = threading.Condition(threading.RLock())
        n.peers = {}
        n.idle = {}

    def peer(self, owner, name):
        worker = owner + '/' + name
        key = worker + '/session'
        self.c.net.peers[key] = {'owner': owner, 'worker': worker}
        event = threading.Event()
        self.c.net.idle[key] = types.SimpleNamespace(event=event)
        return event

    def test_inference_only_is_tombstoned_and_woken(self):
        event = self.peer('alice', 'serve')
        result = self.c.leave('alice', {})
        self.assertEqual(result['removed'], 1)
        self.assertEqual(self.c.meta['left_workers'], [self.c.alias('alice/serve')])
        self.assertTrue(event.is_set())
        self.assertFalse(self.c.net.peers)
        self.assertEqual(self.c.meta['updates'], 99)
        self.assertEqual(self.c.meta['tokens'], 1234)

    def test_lease_only_identity(self):
        self.c.leases['x'] = {'owner': 'alice', 'worker': 'alice/new'}
        self.c.leave('alice', {})
        self.assertFalse(self.c.leases)
        self.assertEqual(self.c.meta['left_workers'], [self.c.alias('alice/new')])

    def test_mixed_roles_deduplicate_and_repeated_leave_is_idempotent(self):
        self.peer('alice', 'mixed')
        self.c.meta['workers']['alice/mixed'] = {'updates': 1}
        self.c.leases['x'] = {'owner': 'alice', 'worker': 'alice/mixed'}
        self.assertEqual(self.c.leave('alice', {})['removed'], 1)
        self.assertEqual(self.c.leave('alice', {})['removed'], 0)
        self.assertEqual(self.c.meta['left_workers'], [self.c.alias('alice/mixed')])

    def test_named_leave_preserves_other_name_and_owner(self):
        for owner, name in [('alice', 'one'), ('alice', 'two'), ('alice2', 'one')]:
            self.peer(owner, name)
            key = owner + '/' + name
            self.c.meta['workers'][key] = {'updates': 1}
            self.c.leases[key] = {'owner': owner, 'worker': key}
        self.assertEqual(self.c.leave('alice', {}, 'one')['removed'], 1)
        self.assertEqual(set(self.c.meta['workers']), {'alice/two', 'alice2/one'})
        self.assertEqual(set(self.c.leases), {'alice/two', 'alice2/one'})
        self.assertEqual({p['worker'] for p in self.c.net.peers.values()}, {'alice/two', 'alice2/one'})
        self.assertEqual(self.c.meta['left_workers'], [self.c.alias('alice/one')])

    def test_all_roles_and_tombstone_bound(self):
        self.c.meta['left_workers'] = ['old-' + str(i) for i in range(5000)]
        self.c.meta['workers']['alice/train'] = {}
        self.peer('alice', 'serve')
        self.c.leases['x'] = {'owner': 'alice', 'worker': 'alice/new'}
        self.assertEqual(self.c.leave('alice', {})['removed'], 3)
        self.assertEqual(len(self.c.meta['left_workers']), 5000)
        self.assertEqual(set(self.c.meta['left_workers'][-3:]), {self.c.alias('alice/' + n) for n in ['train','serve','new']})

if __name__ == '__main__':
    unittest.main()
