"""The POST /update pickle path must reject a decompression bomb by its declared uncompressed size before
torch.load allocates anything.  A genuine torch.save update is a STORED (uncompressed) ZIP, so its uncompressed
size is about its wire size and it passes; a DEFLATE-compressed ZIP that is tiny on the wire but inflates past the
cap is refused.  See load_update_pickle in dbmoe.py."""
import io
import unittest
import zipfile

import torch

import dbmoe


class UpdateZipGuardTest(unittest.TestCase):
    CAP = 1 << 20                                           # 1 MiB test cap (production uses --max-upload-mb << 20)

    def test_genuine_update_round_trips(self):
        # A realistic update payload: a handful of delta tensors plus scalar metadata, as a worker would send.
        up = dict(lease='L1', block=2, name='w', seconds=1.0,
                  d_block={'out.weight': torch.zeros(256, 256), 'ln.weight': torch.ones(256)},
                  d_shared={'emb.weight': torch.zeros(256, 256)})
        body = dbmoe.to_bytes(up)
        self.assertEqual(body[:4], b'PK\x03\x04')           # torch.save writes a ZIP
        self.assertLess(len(body), self.CAP)                # and it is comfortably under the cap on the wire
        out = dbmoe.load_update_pickle(body, self.CAP)
        self.assertEqual(out['lease'], 'L1')
        self.assertEqual(out['block'], 2)
        self.assertTrue(torch.equal(out['d_block']['ln.weight'], torch.ones(256)))

    def test_deflate_bomb_rejected_by_declared_size(self):
        # 4 MiB of zeros, DEFLATE-compressed: a few KB on the wire, but it declares 4 MiB uncompressed.
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr('archive/data/0', b'\x00' * (4 << 20))
        body = buf.getvalue()
        self.assertEqual(body[:4], b'PK\x03\x04')
        # The point of the bug: the body itself is tiny, so the wire-size cap alone never catches it.
        self.assertLess(len(body), self.CAP)
        with self.assertRaises(ValueError) as cm:
            dbmoe.load_update_pickle(body, self.CAP)
        self.assertIn('inflates', str(cm.exception))

    def test_many_entries_rejected(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, 'w', zipfile.ZIP_STORED) as zf:
            for i in range(20):
                zf.writestr(f'archive/data/{i}', b'x')
        with self.assertRaises(ValueError) as cm:
            dbmoe.load_update_pickle(buf.getvalue(), self.CAP, max_entries=8)
        self.assertIn('too many entries', str(cm.exception))

    def test_non_zip_body_skips_zip_check(self):
        # A non-ZIP body cannot inflate past its own length, so the zip guard is skipped and torch.load runs
        # (and rejects this garbage itself). The guard must NOT raise its own 'inflates'/'entries' error here.
        try:
            dbmoe.load_update_pickle(b'not-a-zip-body', self.CAP)
        except ValueError as e:
            self.assertNotIn('inflates', str(e))
            self.assertNotIn('too many entries', str(e))
        except Exception:
            pass                                            # torch.load raising on non-pickle input is fine


if __name__ == '__main__':
    unittest.main()
