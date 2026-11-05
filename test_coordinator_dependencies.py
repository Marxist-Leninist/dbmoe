"""Coordinator install contract and actual snapshot smoke, no network or live services."""
import copy,pathlib,threading,types,unittest
from unittest.mock import patch
import torch
import dbmoe as db

class CoordinatorDependencies(unittest.TestCase):
 def test_numpy_is_declared(self):
  requirements=pathlib.Path(__file__).with_name('requirements.txt').read_text().splitlines()
  self.assertTrue(any(line.strip().lower().startswith('numpy>=') for line in requirements))
 def test_snapshot_fingerprint_uses_available_numpy(self):
  import numpy
  self.assertTrue(numpy.__version__)
  torch.set_num_threads(1);torch.manual_seed(3)
  model=db.Model(db.make_cfg(dict(d=8,heads=2,blocks=2,layers_per_block=1,experts=2,ffn=16,ctx=4))).eval()
  c=types.SimpleNamespace(model=model,lock=threading.RLock(),meta={'updates':0},copy_model=lambda:copy.deepcopy(model))
  with patch.object(threading.Thread,'start'):
   net=db.NetInfer(c)
  first=net.snaps[net.current]['state_hash'];net.snapshot()
  self.assertRegex(first,r'^[0-9a-f]{64}$')
  self.assertEqual(first,net.snaps[net.current]['state_hash'])
if __name__=='__main__':unittest.main()
