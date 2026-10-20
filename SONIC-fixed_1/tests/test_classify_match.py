"""Decision layer: evidence-based, independent of the candidate's source path."""
import shutil
import tempfile
import unittest
from pathlib import Path

from _synth import melody, write  # noqa: E402
import sonic_db_manager as M
from sonic.retrieval.pipeline import MatchResult


def _res(file_id, evidence, kind="edited", confident=False):
    return MatchResult(file_id=file_id, confident=confident, stage_used="primary+secondary", primary_confidence=0.0,
                       offset_confidence=0.0, weighted_score=0.0, margin=0.0, escalated=True,
                       ranked_ids=sorted(evidence, key=lambda f: evidence[f]["score"]) + [99],
                       candidate_evidence=evidence, match_kind=kind if file_id is not None else None,
                       evidence=evidence.get(file_id, {}))


def _ev(score, passes, kind="edited", semi=0.0, tempo=1.0):
    return {"coarse_ber": score, "fine_ber": score, "score": score, "n_aligned": 10, "passes": passes,
            "kind": kind, "semitones": semi, "tempo": tempo}


class ClassifyMatchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="sonic_test_cls_"))
        cls.q = write(cls.tmp / "upload_query.wav", melody(1, 4.0))
        cls.existing = write(cls.tmp / "db_track.wav", melody(2, 4.0))
        cls.missing = r"C:\Users\someone\Echoes\db_track.wav"

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_same_verdict_whether_source_exists_or_not(self):
        res = _res(7, {7: _ev(0.15, True, semi=-2.0), 8: _ev(0.3, False)})
        a = M.SONICDatabaseManager._classify_match(str(self.q), str(self.existing), res, cand_fid=7)
        b = M.SONICDatabaseManager._classify_match(str(self.q), self.missing, res, cand_fid=7)
        self.assertEqual(a, (True, "Edited Audio"))
        self.assertEqual(a, b)

    def test_duplicate_label_and_rejection(self):
        res = _res(7, {7: _ev(0.03, True, kind="duplicate")}, kind="duplicate")
        self.assertEqual(M.SONICDatabaseManager._classify_match(str(self.q), self.missing, res), (True, "Duplicate"))
        none = _res(None, {7: _ev(0.3, False)})
        self.assertEqual(M.SONICDatabaseManager._classify_match(str(self.q), self.missing, none), (False, None))

    def test_duplicate_with_different_length_is_edited(self):
        res = _res(7, {7: _ev(0.03, True, kind="duplicate")}, kind="duplicate")
        out = M.SONICDatabaseManager._classify_match(str(self.q), self.missing, res, cand_fid=7, db_duration=9.0)
        self.assertEqual(out, (True, "Edited Audio"))
        same = M.SONICDatabaseManager._classify_match(str(self.q), self.missing, res, cand_fid=7, db_duration=4.0)
        self.assertEqual(same, (True, "Duplicate"))

    def test_candidate_without_own_evidence_rejected(self):
        res = _res(7, {7: _ev(0.05, True, kind="duplicate"), 8: _ev(0.33, False)}, kind="duplicate")
        self.assertEqual(M.SONICDatabaseManager._classify_match(str(self.q), self.missing, res, cand_fid=99), (False, None))
        self.assertEqual(M.SONICDatabaseManager._classify_match(str(self.q), self.missing, res, cand_fid=8), (False, None))

    def test_audit_style_loop_flags_only_verified_candidates(self):
        res = _res(7, {7: _ev(0.0, True, kind="duplicate"), 8: _ev(0.12, True), 9: _ev(0.31, False)}, kind="duplicate")
        flagged = [f for f in res.ranked_ids if f != 7 and
                   M.SONICDatabaseManager._classify_match(str(self.q), self.missing, res, cand_fid=f)[0]]
        self.assertEqual(flagged, [8])


if __name__ == "__main__":
    unittest.main()
