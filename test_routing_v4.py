"""Isolated CPU tests, no production files, endpoints or data."""
import ast, base64, collections, copy, importlib.util, io, json, os, pathlib, socket, threading, time, types, unittest
from unittest import mock
import torch
HERE = pathlib.Path(__file__).resolve().parent

def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m

db = load('route_candidate', os.getenv('DBMOE_TEST_SOURCE', 'dbmoe.py'))
BASELINE = os.getenv('DBMOE_TEST_BASE')
old = load('route_base', BASELINE) if BASELINE else None
torch.set_num_threads(1)

def fixture(module=db):
    torch.manual_seed(12)
    model = module.Model(module.make_cfg(dict(d=8, heads=2, blocks=2, layers_per_block=1, experts=2, ffn=16, ctx=4))).eval()
    c = types.SimpleNamespace(model=model, lock=threading.RLock(), meta={'updates': 0}, copy_model=lambda: copy.deepcopy(model),
                              log=lambda **kw: None, is_open=lambda owner: False, alias=lambda w: w,
                              enrol=types.SimpleNamespace(strike=lambda *a: None))
    with mock.patch.object(threading.Thread, 'start'):
        net = module.NetInfer(c, ctx_len=4, idle_wait=.2, poll_s=.02, hop_timeout=1, audit_trusted=0, audit_open=0)
    return net

def tokens(events): return [e['b'] for e in events if e['t'] == 'byte']

class Workers:
    def __init__(self, net, blocks=(0, 1), corrupt=False):
        self.net, self.stop, self.threads, self.sockets, self.jobs = net, threading.Event(), [], [], []
        self.errors = []
        for b in blocks:
            key = f'w{b}#test'
            net.peers[key] = dict(owner='test', worker=f'w{b}', open=corrupt, target=b, ewma=1., hops=0, strikes=0,
                                  banned=False, snap=net.current, block=b, seen=time.time())
            s, other = socket.socketpair(); self.sockets += [s, other]
            t = threading.Thread(target=self.run, args=(b, s, corrupt), daemon=True); self.threads.append(t); t.start()
        self.wait_idle(len(blocks))
    def wait_idle(self, count):
        until = time.monotonic() + 3
        while len(self.net.idle) < count:
            if time.monotonic() > until: raise AssertionError(('workers not idle', self.errors, self.net.peers))
            time.sleep(.005)
    def hold(self, sid, block):
        blob, err = self.net.block_blob(sid, block, 'test'); assert not err, err
        L = db.from_bytes(blob); sh, bl = db.Shared(L['cfg']), db.Block(L['cfg'])
        sh.load_state_dict(L['shared']); bl.load_state_dict(L['block_state'])
        return dict(snap=sid, block=block, shared=sh.eval(), module=bl.eval(), d=L['cfg']['d'])
    def run(self, block, sock, corrupt):
        try:
            held, result = self.hold(self.net.current, block), None
            while not self.stop.is_set():
                r = self.net.poll('test', f'w{block}', dict(session='test', snap=held['snap'], block=held['block'], result=result), sock)
                result = None
                if 'assign' in r: held = self.hold(r['assign']['snap'], r['assign']['block'])
                if 'job' in r:
                    self.jobs.append(r['job'])
                    result = db.serve_job(held, r['job'])
                    if corrupt: result['z'] = db.encode_floats(torch.ones(1, 1, held['d']) * 100)
        except Exception as e: self.errors.append(repr(e))
    def close(self):
        self.stop.set()
        with self.net.cv:
            for slot in self.net.idle.values(): slot.event.set()
        for t in self.threads: t.join(2)
        for s in self.sockets: s.close()
        assert not self.errors, self.errors
        assert not any(t.is_alive() for t in self.threads)
    def __enter__(self): return self
    def __exit__(self, *args): self.close()

class RoutingTests(unittest.TestCase):
    @unittest.skipIf(old is None, 'Set DBMOE_TEST_BASE to an earlier dbmoe.py for optional generation parity')
    def test_v4_math_unchanged_against_base(self):
        net, previous = fixture(), fixture(old)
        for sampler in ['chain', 'top']:
            for prompt in ['The ', 'A']:
                for seed in [0, 7, 19]:
                    a = list(net.generate(prompt, 4, seed=seed, temp=.75, sampler=sampler, route='single'))
                    b = list(previous.generate(prompt, 4, seed=seed, temp=.75, sampler=sampler))
                    self.assertEqual(tokens(a), tokens(b), (sampler, prompt, seed))
                    self.assertEqual(a[-1]['hops_network'], 0)
    def test_real_protocol_parity_both_samplers(self):
        net = fixture()
        with Workers(net) as workers:
            for sampler in ['chain', 'top']:
                for prompt in ['The ', 'A']:
                    for seed in [0, 7, 19]:
                        a = list(net.generate(prompt, 4, seed=seed, temp=.75, sampler=sampler, route='single'))
                        b = list(net.generate(prompt, 4, seed=seed, temp=.75, sampler=sampler, route='distributed', snapshot_id=a[0]['snapshot_id']))
                        self.assertEqual(tokens(a), tokens(b), (sampler, prompt, seed))
                        self.assertEqual(b[-1]['hops_coordinator'], 0)
                        self.assertEqual(a[-1]['state_hash'], b[-1]['state_hash'])
                        self.assertEqual(a[-1]['sampler_config'], b[-1]['sampler_config'])
                        self.assertEqual(b[-1]['sampler'], sampler)
                        self.assertGreaterEqual(b[-1]['hops_network'], 4)
            self.assertEqual(len(workers.jobs), 72)
    def test_auto_selects_measured_distributed_and_caches(self):
        net = fixture()
        real = db.run_segment
        def slow_single(*a, **kw):                      # only the single route passes prepared=
            if kw.get('prepared') is not None:
                time.sleep(.05)
            return real(*a, **kw)
        with Workers(net), mock.patch.object(db, 'run_segment', side_effect=slow_single):
            first = list(net.generate('A', 1, seed=19, sampler='top', route='auto'))
            second = list(net.generate('A', 1, seed=19, sampler='top', route='auto'))
        self.assertEqual(first[0]['requested_route'], 'auto')
        self.assertEqual(first[0]['route'], 'distributed')
        self.assertFalse(first[0]['auto_calibration']['cached'])
        self.assertTrue(first[0]['auto_calibration']['measured']['single']['ok'])
        self.assertTrue(first[0]['auto_calibration']['measured']['distributed']['ok'])
        self.assertEqual(first[-1]['hops_coordinator'], 0)
        self.assertEqual(first[-1]['route'], 'distributed')
        self.assertEqual(second[0]['route'], 'distributed')
        self.assertTrue(second[0]['auto_calibration']['cached'])
        self.assertEqual(tokens(first), tokens(second))
        self.assertEqual(len(net.recent), 4)  # two probes, then two actual generations; the second choice is cached
        self.assertEqual([r['calibration'] for r in net.recent], [True, True, False, False])
        self.assertFalse(net.active_snaps)
    def test_auto_without_workers_uses_single_after_real_probe(self):
        net = fixture()
        events = list(net.generate('A', 1, sampler='top', route='auto'))
        self.assertEqual(events[0]['route'], 'single')
        self.assertEqual(events[0]['requested_route'], 'auto')
        self.assertTrue(events[0]['auto_calibration']['measured']['single']['ok'])
        self.assertFalse(events[0]['auto_calibration']['measured']['distributed']['ok'])
        self.assertEqual(events[-1]['t'], 'done')
        self.assertFalse(net.active_snaps)
    def test_distributed_rollover_stays_pinned(self):
        net = fixture()
        with Workers(net) as workers:
            g = net.generate('The ', 3, route='distributed'); start = next(g); first = next(g)
            for _ in range(4): net.C.meta['updates'] += 1; net.snapshot()
            rest = list(g)
            self.assertEqual(rest[-1]['hops_coordinator'], 0)
            self.assertEqual(rest[-1]['hops_network'], 6)
            self.assertEqual({j['snap'] for j in workers.jobs}, {start['snapshot_id']})
            self.assertEqual(rest[-1]['state_hash'], start['state_hash'])
            self.assertFalse(net.active_snaps)
    def test_top_only_needs_top_worker(self):
        net = fixture()
        with Workers(net, blocks=(1,)):
            out = list(net.generate('A', 2, route='distributed', sampler='top'))
            self.assertEqual(out[-1]['hops_network'], 2)
            with self.assertRaises(RuntimeError): list(net.generate('A', 1, route='distributed', sampler='chain'))
    def test_pin_close_and_eviction(self):
        net = fixture(); g = net.generate('The ', 2, route='single'); sid = next(g)['snapshot_id']
        for _ in range(4): net.snapshot()
        self.assertIn(sid, net.snaps); g.close(); self.assertFalse(net.active_snaps)
        net.snapshot(); self.assertNotIn(sid, net.snaps)
    def test_fallback_uses_pinned(self):
        net = fixture(); sid = net.current; net.snapshot()
        with mock.patch.object(net, '_local', return_value=(1, 2)) as f:
            net.hop(dict(snap=sid, route='single', block=0)); self.assertIs(f.call_args.args[1], net.snaps[sid]['model'])
    def test_claim_rejects_other_snapshot(self):
        net = fixture(); sid = net.current; net.snapshot()
        net.peers = {'new': dict(snap=net.current, block=0, banned=False, seen=time.time(), ewma=1)}
        net.idle = {'new': types.SimpleNamespace(block=0, conn=None)}
        self.assertEqual(net._claim(dict(snap=sid, block=0), set()), (None, None))
    def test_no_silent_distributed_fallback(self):
        net = fixture()
        with self.assertRaises(RuntimeError): list(net.generate('A', 1, route='distributed'))
        with self.assertRaises(RuntimeError): net.hop(dict(snap=net.current, route='distributed', block=0))
        self.assertEqual(net.stats['hops_coordinator'], 0); self.assertFalse(net.active_snaps)
    def test_corrupt_worker_cannot_force_single_substitution(self):
        net = fixture()
        with Workers(net, corrupt=True):
            with self.assertRaisesRegex(RuntimeError, 'verification failed'):
                list(net.generate('A', 1, route='distributed'))
        self.assertEqual(net.stats['hops_coordinator'], 0); self.assertFalse(net.active_snaps)
    def test_explicit_snapshot_and_bad_inputs(self):
        net = fixture(); sid = net.current; net.snapshot()
        self.assertEqual(list(net.generate('A', 1, route='single', snapshot_id=sid))[-1]['snapshot_id'], sid)
        for opts in [dict(snapshot_id=-1), dict(route='fastest'), dict(route='turbo'), dict(sampler='other')]:
            with self.assertRaises(ValueError): list(net.generate('A', 1, **opts))
        self.assertFalse(net.active_snaps)
    def test_metadata_and_decode_use_snapshot_model(self):
        net = fixture(); sid = net.current
        expected = list(net.generate('A', 2, route='single'))
        net.C.model.cfg = dict(net.C.model.cfg, d=40, vocab=41, blocks=7)
        net.C.model.sigmas = [99.]
        out = list(net.generate('A', 2, route='single', snapshot_id=sid))
        self.assertEqual(tokens(out), tokens(expected)); self.assertEqual(len(out[0]['coverage']), 2)
        res = dict(z=db.encode_floats(torch.zeros(8)), logits=db.encode_floats(torch.zeros(256)))
        z, lg = net._decode(res, True, net.snaps[sid]['model'].cfg)
        self.assertEqual(lg.numel(), 256)
    def test_failure_after_byte_releases_pin(self):
        net = fixture(); g = net.generate('A', 3, route='single'); next(g); next(g)
        with mock.patch.object(db, 'run_segment', side_effect=RuntimeError('test failure')):
            with self.assertRaises(RuntimeError): next(g)
        self.assertFalse(net.active_snaps)
    def test_epoch_hash_and_sigmas(self):
        with mock.patch.object(time, 'time', return_value=1234): a, b = fixture(), fixture()
        self.assertNotEqual(a.current, b.current); self.assertLess(a.current, 2 ** 53)
        h = a.snaps[a.current]['state_hash']; a.snapshot(); self.assertEqual(a.snaps[a.current]['state_hash'], h)
        with torch.no_grad(): next(a.C.model.parameters()).add_(.1)
        a.snapshot(); self.assertNotEqual(a.snaps[a.current]['state_hash'], h)
        h = a.snaps[a.current]['state_hash']; a.C.model.sigmas[0] += .01; a.snapshot()
        self.assertNotEqual(a.snaps[a.current]['state_hash'], h)
    def test_audit_model_survives_snapshot_eviction(self):
        net = fixture(); sid = net.current; model = net.snaps[sid]['model']
        j = dict(block=0, ids=base64.b64encode(b'A').decode(), seed=0, z=db.encode_floats(torch.zeros(8)),
                 grid=db.block_grid(model.sigmas, 0, net.spb), final=True, exit_conf=db.EXIT_CONF, snap=sid)
        zl, lg = net._local(j, model)
        for _ in range(3): net.snapshot()
        self.assertNotIn(sid, net.snaps)
        net._audit_later('test', {}, j, zl, lg, model)
        self.assertEqual(net.stats['audits'], 1); self.assertEqual(net.stats['audit_failures'], 0)
    def test_start_failure_releases_pin(self):
        net = fixture()
        with mock.patch.object(net, 'coverage', side_effect=RuntimeError('bad coverage')):
            with self.assertRaises(RuntimeError): next(net.generate('A', 1, route='single'))
        self.assertFalse(net.active_snaps)

class HTTPTests(unittest.TestCase):
    def setUp(self):
        self.net = fixture()
        tree = ast.parse(HERE.joinpath(os.getenv('DBMOE_TEST_SOURCE', 'dbmoe.py')).read_text())
        serve = next(n for n in tree.body if getattr(n, 'name', None) == 'serve')
        h = next(n for n in serve.body if isinstance(n, ast.ClassDef) and n.name == 'H')
        ns = dict(vars(db), C=types.SimpleNamespace(net=self.net))
        exec(compile(ast.Module(body=[h], type_ignores=[]), '<actual-http-handler>', 'exec'), ns)
        self.h = ns['H'].__new__(ns['H']); self.h.wfile = io.BytesIO(); self.status = []
        self.h.send_response = self.status.append; self.h.send_header = lambda *a: None; self.h.end_headers = lambda: None
    def run_request(self, q):
        self.h._net_sample(dict(prompt='A', n='1', **q)); self.assertFalse(self.net.gen_lock.locked()); self.assertFalse(self.net.active_snaps)
        return [json.loads(s) for s in self.h.wfile.getvalue().splitlines()] if self.status == [200] else []
    def test_single_success(self):
        out = self.run_request(dict(route='single', sampler='top'))
        self.assertEqual(self.status, [200]); self.assertEqual(out[-1]['t'], 'done'); self.assertEqual(out[-1]['sampler'], 'top')
    def test_real_http_transport_both_routes(self):
        import urllib.request, urllib.parse
        from http.server import ThreadingHTTPServer
        server = ThreadingHTTPServer(('127.0.0.1', 0), type(self.h))
        worker = threading.Thread(target=server.serve_forever, daemon=True); worker.start()
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            with Workers(self.net):
                for sampler in ['chain', 'top']:
                    responses = []
                    sid = self.net.current
                    for route in ['single', 'distributed']:
                        q = dict(net='1', prompt='A', n='2', sampler=sampler, route=route, snapshot=str(sid), seed='19')
                        with opener.open(f'http://127.0.0.1:{server.server_port}/sample?' + urllib.parse.urlencode(q), timeout=5) as response:
                            self.assertEqual(response.status, 200)
                            responses.append([json.loads(line) for line in response.read().splitlines()])
                    self.assertEqual(tokens(responses[0]), tokens(responses[1]))
                    self.assertEqual(responses[1][-1]['hops_coordinator'], 0)
                    self.assertEqual(responses[1][-1]['t'], 'done')
        finally:
            server.shutdown(); server.server_close(); worker.join(2)
        self.assertFalse(self.net.gen_lock.locked()); self.assertFalse(self.net.active_snaps)
    def test_bad_parameters(self):
        for q in [dict(snapshot='bad'), dict(snapshot='-1'), dict(route='fastest'), dict(route='turbo'), dict(temp='nan'), dict(seed='abc'), dict(sampler='wrong')]:
            self.setUp(); self.run_request(q); self.assertEqual(self.status, [409 if q.get('snapshot') == '-1' else 400], q)
    def test_auto_http_without_workers_selects_single(self):
        out = self.run_request(dict(route='auto', sampler='top'))
        self.assertEqual(self.status, [200])
        self.assertEqual(out[0]['requested_route'], 'auto')
        self.assertEqual(out[0]['route'], 'single')
        self.assertEqual(out[-1]['route'], 'single')
    def test_auto_http_total_failure_closes_cleanly(self):
        with mock.patch.object(self.net, '_local', side_effect=RuntimeError('test')), mock.patch.object(db, 'run_segment', side_effect=RuntimeError('test')):
            out = self.run_request(dict(route='auto', sampler='top'))
        self.assertEqual(self.status, [409])
        self.assertEqual(out, [])
    def test_distributed_no_workers_before_headers(self):
        self.run_request(dict(route='distributed')); self.assertEqual(self.status, [409])
    def test_busy(self):
        self.net.gen_lock.acquire()
        try:
            self.h._net_sample(dict(route='single')); self.assertEqual(self.status, [503]); self.assertTrue(self.net.gen_lock.locked())
        finally: self.net.gen_lock.release()
    def test_disconnect(self):
        self.h.wfile = mock.Mock(); self.h.wfile.write.side_effect = BrokenPipeError()
        self.h._net_sample(dict(route='single', n='1')); self.assertFalse(self.net.gen_lock.locked()); self.assertFalse(self.net.active_snaps)
    def test_midstream_error_no_done(self):
        with mock.patch.object(db, 'run_segment', side_effect=RuntimeError('test')):
            out = self.run_request(dict(route='single'))
        self.assertEqual(self.status, [200]); self.assertEqual([e['t'] for e in out], ['start', 'error'])

class DashboardTests(unittest.TestCase):
    def test_all_route_choices_and_auto_metadata_are_wired(self):
        html = HERE.joinpath('web', 'progress', 'index.html').read_text()
        for route in ('auto', 'single', 'distributed', 'network'):
            self.assertIn(f'option value="{route}"', html)
        self.assertIn("$('inferenceRoute').value", html)
        self.assertIn('route:selectedRoute', html)
        self.assertIn('auto_calibration', html)
        self.assertIn('one checkpoint snapshot', html)
        self.assertIn('--device cuda|rocm|xpu|mps', html)

if __name__ == '__main__': unittest.main(verbosity=2)
