import unittest, types, io, contextlib
from unittest.mock import patch
import torch
import dbmoe as m

class AllocatorTests(unittest.TestCase):
    def test_rate_limit(self):
        now=[0.0];calls=[];a=m.WorkerAllocatorMaintenance(clock=lambda:now[0]);a._trim=lambda n:calls.append(n)
        self.assertTrue(a.maybe_trim());self.assertFalse(a.maybe_trim())
        now[0]=59.99;self.assertFalse(a.maybe_trim())
        now[0]=60;self.assertTrue(a.maybe_trim());self.assertEqual(calls,[0,0])
    def test_unavailable(self):
        a=m.WorkerAllocatorMaintenance();a._trim=None;self.assertFalse(a.maybe_trim())
    def test_failure_disables(self):
        a=m.WorkerAllocatorMaintenance()
        def fail(n):raise RuntimeError('fixture')
        a._trim=fail;self.assertFalse(a.maybe_trim());self.assertIsNone(a._trim)
        self.assertTrue(a._lock.acquire(False));a._lock.release()
    def test_busy_is_nonblocking(self):
        calls=[];a=m.WorkerAllocatorMaintenance();a._trim=lambda n:calls.append(n)
        a._lock.acquire()
        try:self.assertFalse(a.maybe_trim())
        finally:a._lock.release()
        self.assertEqual(calls,[])
    def test_nonlinux_is_noop(self):
        with patch.object(m.sys,'platform','darwin'):
            a=m.WorkerAllocatorMaintenance()
        self.assertIsNone(a._trim);self.assertFalse(a.maybe_trim())
    def test_live_tensors_unchanged(self):
        a=m.WorkerAllocatorMaintenance(interval=0);x=torch.arange(100000,dtype=torch.float32);expected=x.clone()
        a.maybe_trim();self.assertTrue(torch.equal(x,expected))
    def _worker(self,device):
        order=[]
        args=types.SimpleNamespace(threads=1,device=device,url='http://unused.invalid',token='isolated-test',name='isolated',
            no_share_hardware=True,serve_only=False,no_serve=True,hours=0,steps=60,batch=8,once=2)
        up=dict(fresh_optimizer=True,losses=[1.0,0.9],tokens=10,seconds=1.0)
        lease=dict(block=0)
        class FakeMaintenance:
            def __init__(self):order.append('construct')
            def maybe_trim(self):order.append('trim')
        def post(*a):order.append('post');return dict(ok=True)
        with patch.object(m,'prepare_worker_device',return_value={'torch_device':device,'device':device}), \
             patch.object(m,'http',return_value=b'never sent'),patch.object(m,'from_bytes',return_value=lease), \
             patch.object(m,'train_lease',side_effect=lambda *a:dict(up)),patch.object(m,'update_to_safetensors',return_value=b'fixture'), \
             patch.object(m,'post_update',side_effect=post),patch.object(m,'WorkerAllocatorMaintenance',FakeMaintenance), \
             contextlib.redirect_stdout(io.StringIO()):
            m.worker(args)
        return order
    def test_cpu_worker_trims_after_receipt(self):
        self.assertEqual(self._worker('cpu'),['construct','post','trim','post','trim'])
    def test_cuda_worker_does_not_trim(self):
        self.assertEqual(self._worker('cuda:0'),['post','post'])

if __name__=='__main__':unittest.main()
