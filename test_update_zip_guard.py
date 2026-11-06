"""Bounded validation for untrusted POST /update pickle uploads."""
import io
import struct
import unittest
import zipfile
from unittest.mock import patch

import torch

import dbmoe


class UpdateZipGuardTest(unittest.TestCase):
    CAP = 1 << 20

    @staticmethod
    def archive(entries, compression=zipfile.ZIP_STORED):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, 'w', compression) as zf:
            for name, data in entries:
                zf.writestr(name, data)
        return buf.getvalue()

    def reject_before_load(self, body, **kwargs):
        cap = kwargs.pop('max_bytes', self.CAP)
        with patch.object(dbmoe.torch, 'load') as load:
            with self.assertRaises(ValueError):
                dbmoe.load_update_pickle(body, cap, **kwargs)
            load.assert_not_called()

    def test_genuine_update_round_trips(self):
        up = dict(lease='L1', block=2, name='w', seconds=1.0,
                  d_block={'out.weight': torch.zeros(256, 256), 'ln.weight': torch.ones(256)},
                  d_shared={'emb.weight': torch.zeros(256, 256)})
        body = dbmoe.to_bytes(up)
        self.assertLess(len(body), self.CAP)
        out = dbmoe.load_update_pickle(body, self.CAP)
        self.assertEqual(out['lease'], 'L1')
        self.assertEqual(out['block'], 2)
        self.assertTrue(torch.equal(out['d_block']['ln.weight'], up['d_block']['ln.weight']))

    def test_legacy_allocation_bypass_rejected(self):
        buf = io.BytesIO()
        torch.save(torch.zeros(1 << 20), buf, _use_new_zipfile_serialization=False)
        self.reject_before_load(buf.getvalue()[:512])

    def test_valid_legacy_pickle_rejected(self):
        buf = io.BytesIO()
        torch.save({'x': torch.ones(1)}, buf, _use_new_zipfile_serialization=False)
        self.reject_before_load(buf.getvalue())

    def test_non_zip_rejected(self):
        self.reject_before_load(b'not-a-zip-body')

    def test_deflate_bomb_rejected(self):
        body = self.archive([('archive/data.pkl', b'x'),
                             ('archive/data/0', bytes(4 << 20))], zipfile.ZIP_DEFLATED)
        self.assertLess(len(body), self.CAP)
        self.reject_before_load(body)

    def test_small_compressed_archive_rejected(self):
        self.reject_before_load(self.archive([('archive/data.pkl', b'x')], zipfile.ZIP_DEFLATED))

    def test_many_entries_rejected(self):
        body = self.archive([('archive/data.pkl', b'x')] +
                            [('archive/data/' + str(i), b'x') for i in range(10)])
        self.reject_before_load(body, max_entries=8)

    def test_forged_eocd_count_cannot_hide_directory_entries(self):
        body = bytearray(self.archive([('archive/data.pkl', b'x')] +
                                      [('archive/data/' + str(i), b'x') for i in range(4)]))
        eocd = body.rfind(b'PK\x05\x06')
        struct.pack_into('<H', body, eocd + 8, 1)
        struct.pack_into('<H', body, eocd + 10, 1)
        self.reject_before_load(bytes(body))

    def test_zip64_end_record_rejected(self):
        body = bytearray(self.archive([('archive/data.pkl', b'x')]))
        eocd = body.rfind(b'PK\x05\x06')
        struct.pack_into('<H', body, eocd + 8, 0xffff)
        struct.pack_into('<H', body, eocd + 10, 0xffff)
        self.reject_before_load(bytes(body))

    def test_duplicate_rejected(self):
        self.reject_before_load(self.archive([('archive/data.pkl', b'x')] * 2))

    def test_unknown_record_rejected(self):
        self.reject_before_load(self.archive([('archive/data.pkl', b'x'), ('archive/code.py', b'x')]))

    def test_mixed_root_rejected(self):
        self.reject_before_load(self.archive([('archive/data.pkl', b'x'), ('other/data/0', b'x')]))

    def test_missing_metadata_rejected(self):
        self.reject_before_load(self.archive([('archive/data/0', b'x')]))

    def test_aggregate_metadata_rejected(self):
        body = self.archive([('archive/data.pkl', bytes(600000)), ('archive/byteorder', bytes(600000))])
        self.reject_before_load(body)

    def test_wire_cap_enforced(self):
        body = dbmoe.to_bytes({'x': torch.ones(16)})
        self.reject_before_load(body, max_bytes=len(body) - 1)

    def test_malformed_zip_rejected(self):
        self.reject_before_load(b'PK\x03\x04' + bytes(30))

    def test_trailing_data_rejected(self):
        self.reject_before_load(self.archive([('archive/data.pkl', b'x')]) + b'extra')

    def test_untrusted_handler_uses_guard(self):
        import inspect
        source = inspect.getsource(dbmoe.serve)
        self.assertIn('load_update_pickle(body, a.max_upload_mb << 20)', source)
        self.assertIn('update_from_safetensors(body)', source)

    def test_trusted_loader_unchanged(self):
        buf = io.BytesIO()
        torch.save({'x': torch.ones(1)}, buf, _use_new_zipfile_serialization=False)
        self.assertTrue(torch.equal(dbmoe.from_bytes(buf.getvalue())['x'], torch.ones(1)))


if __name__ == '__main__':
    unittest.main()
