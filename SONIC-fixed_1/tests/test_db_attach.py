"""SONICDatabaseManager attaches archive + delta .npz entries without a full rebuild."""
import json
import shutil
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from _synth import melody, write  # noqa: E402
import sonic_db_manager as M
from sonic.fingerprint.binary_embed import BinaryEmbedIndex
from sonic.retrieval.pipeline import query


def _wait(db, timeout=120):
    t0 = time.time()
    while db.domain_index is None:
        if time.time() - t0 > timeout:
            raise AssertionError("index did not attach")
        time.sleep(0.05)
    return db.domain_index


class DbAttachTest(unittest.TestCase):
    def setUp(self):
        import faiss
        self.tmp = Path(tempfile.mkdtemp(prefix="sonic_test_attach_"))
        self.store = self.tmp / "store"
        (self.store / "fingerprints").mkdir(parents=True)
        self.wavs = [write(self.tmp / f"track{i}.wav", melody(40 + i, 3.0 + i)) for i in range(4)]
        entries, fps = [], []
        for i, p in enumerate(self.wavs):
            fp = M.SONICDatabaseManager.extract_fingerprint(p)
            fps.append(fp)
            np.savez_compressed(self.store / "fingerprints" / f"fp_{i}.npz", hash=fp["hash"],
                                anchor_time=fp["anchor_time"], raw_vec=fp["raw_vec"], subfp=fp["subfp"],
                                subfp_fine=fp["subfp_fine"])
            entries.append({"file_id": i, "filename": p.name, "path": str(p), "category": "audio",
                            "duration_sec": fp["duration_sec"], "fp_file": f"fp_{i}.npz", "source_folder": "t"})
        with open(self.store / "manifest.json", "w") as f:
            json.dump({"count": 4, "files": entries}, f)
        # archive covering only the first 3 entries (same keys as the real unified_archive.npz)
        a = fps[:3]
        emb = BinaryEmbedIndex()
        raw = np.stack([f["raw_vec"] for f in a]).astype(np.float32)
        np.savez(self.store / "unified_archive.npz",
                 file_ids=np.arange(3, dtype=np.int64),
                 n_hashes=np.array([f["hash"].size for f in a], dtype=np.int32),
                 hashes=np.concatenate([f["hash"] for f in a]).astype(np.uint32),
                 lids=np.concatenate([np.full(f["hash"].size, i, np.int32) for i, f in enumerate(a)]),
                 times=np.concatenate([f["anchor_time"] for f in a]).astype(np.float32),
                 raw_vecs=raw,
                 subfp_fids=np.concatenate([np.full(len(f["subfp"]), i, np.int64) for i, f in enumerate(a)]),
                 subfp_codes=np.concatenate([f["subfp"] for f in a]),
                 subfp_fine_fids=np.concatenate([np.full(len(f["subfp_fine"]), i, np.int64) for i, f in enumerate(a)]),
                 subfp_fine_codes=np.concatenate([f["subfp_fine"] for f in a]))
        h = faiss.IndexBinaryHNSW(256, 32)
        h.add(np.stack([emb.binarize(v) for v in raw]))
        faiss.write_index_binary(h, str(self.store / "hnsw_index.faiss"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_delta_attach_matches_full_rebuild(self):
        before = {p.name: p.stat().st_mtime for p in self.store.rglob("*")}
        db = M.SONICDatabaseManager(store_dir=self.store)
        idx = _wait(db)
        self.assertEqual(idx.landmark_index.n_files, 4)
        self.assertEqual(sorted(idx.subfp_index.file_start), [0, 1, 2, 3])
        self.assertEqual(idx.binary_hnsw.index.ntotal, 4)
        self.assertEqual(set(db._manifest_by_id), {0, 1, 2, 3})
        delta_top = []
        for i, p in enumerate(self.wavs):
            res = query(str(p), idx)
            self.assertEqual(res.file_id, i)
            delta_top.append(res.ranked_ids[0])
        self.assertEqual(before, {p.name: p.stat().st_mtime for p in self.store.rglob("*")})  # store untouched

        # full .npz rebuild path (no archive) gives the same answers
        (self.store / "unified_archive.npz").unlink()
        db2 = M.SONICDatabaseManager(store_dir=self.store)
        idx2 = _wait(db2)
        self.assertEqual(idx2.landmark_index.n_files, 4)
        self.assertEqual([query(str(p), idx2).ranked_ids[0] for p in self.wavs], delta_top)


if __name__ == "__main__":
    unittest.main()
