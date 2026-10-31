from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

SUITE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SUITE_DIR / "scripts"))

from benchmark_matrix import BenchmarkDataError  # noqa: E402
from results import PROVENANCE_PREFIX, collect_provenance  # noqa: E402

CORE = ('{"repositories":{"tigris_compiler":{"revision":"%s"},'
        '"tigris_runtime":{"revision":"%s"}}}')


class CoreProvenanceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.raw = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def write(self, name: str, provenance: str | None) -> None:
        text = "BENCH_RESULT:framework=x\nBENCH_DONE\n"
        if provenance is not None:
            text += PROVENANCE_PREFIX + provenance + "\n"
        (self.raw / name).write_text(text)

    def test_agreeing_captures_give_the_summary_core(self) -> None:
        self.write("tigris_a.log", CORE % ("a" * 40, "b" * 40))
        self.write("tigris_b.log", CORE % ("a" * 40, "b" * 40))
        self.write("tflm_a.log", None)
        record = collect_provenance(self.raw)
        self.assertEqual(
            record["repositories"]["tigris_runtime"]["revision"], "b" * 40)

    def test_no_provenance_lines_give_none(self) -> None:
        self.write("tigris_a.log", None)
        self.assertIsNone(collect_provenance(self.raw))

    def test_a_capture_without_provenance_fails(self) -> None:
        self.write("tigris_a.log", CORE % ("a" * 40, "b" * 40))
        self.write("tigris_b.log", None)
        with self.assertRaisesRegex(BenchmarkDataError, "tigris_b.log"):
            collect_provenance(self.raw)

    def test_captures_naming_different_cores_fail(self) -> None:
        self.write("tigris_a.log", CORE % ("a" * 40, "b" * 40))
        self.write("tigris_b.log", CORE % ("a" * 40, "c" * 40))
        with self.assertRaisesRegex(BenchmarkDataError, "different cores"):
            collect_provenance(self.raw)


if __name__ == "__main__":
    unittest.main()
