"""Exercise the actual dashboard renderer with a small DOM test double."""
import pathlib,shutil,subprocess,unittest
class DashboardWorkers(unittest.TestCase):
 @unittest.skipUnless(shutil.which('node'),'Node.js required for dashboard renderer tests')
 def test_inference_only_and_mixed_contributors(self):
  subprocess.run(['node',str(pathlib.Path(__file__).with_suffix('.js'))],check=True,timeout=15)
