from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

SUITE_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = SUITE_DIR / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import slim_summary  # noqa: E402


class SlimSummaryTest(unittest.TestCase):
    def test_large_output_replaced_with_canary_and_digest(self) -> None:
        values = [((i % 255) - 127) for i in range(5000)]  # all in [-128, 127]
        config = {"model": "unet_matched", "latency_mean_ms": 1.0,
                  "output_values": {"i8": values}}
        slim_summary.slim_config(config)

        # The inline vector is now just the canary slice.
        self.assertEqual(config["output_values"]["i8"],
                         values[:slim_summary.CANARY_LEN])
        # The digest records the true length and the sha256 of the full int8
        # byte vector (recomputable from the reference or a fresh capture).
        digest = config["output_digest"]["i8"]
        self.assertEqual(digest["len"], 5000)
        self.assertEqual(digest["canary_len"], slim_summary.CANARY_LEN)
        raw = bytes((v & 0xFF) for v in values)
        self.assertEqual(digest["sha256"], hashlib.sha256(raw).hexdigest())

    def test_small_output_unchanged(self) -> None:
        config = {"model": "ds_cnn", "output_values": {"i8": [1, -2, 3]}}
        slim_summary.slim_config(config)
        self.assertEqual(config["output_values"]["i8"], [1, -2, 3])
        self.assertNotIn("output_digest", config)

    def test_failure_cell_without_outputs_unchanged(self) -> None:
        config = {"model": "unet", "status": "ARENA_TOO_SMALL"}
        slim_summary.slim_config(config)
        self.assertNotIn("output_digest", config)
        self.assertNotIn("output_values", config)

    def test_slimmed_output_preserves_validate_cell_contract(self) -> None:
        # validate_cell requires output_values == {"i8": <non-empty ints in
        # [-128, 127]>}; the canary must keep satisfying that after trimming.
        values = (list(range(-128, 127)) * 3000)  # > MAX_INLINE_OUTPUT, in range
        config = {"output_values": {"i8": values}}
        slim_summary.slim_config(config)
        outs = config["output_values"]
        self.assertEqual(set(outs), {"i8"})
        self.assertTrue(outs["i8"])
        self.assertTrue(all(isinstance(v, int) and -128 <= v <= 127
                            for v in outs["i8"]))

    def test_summary_roundtrip_only_trims_large_cells(self) -> None:
        big = [((i % 255) - 127) for i in range(524288)]
        summary = {"configs": [
            {"model": "ds_cnn", "output_values": {"i8": [1, -2, 3]}},
            {"model": "unet_matched", "output_values": {"i8": big}},
            {"model": "unet", "status": "ARENA_TOO_SMALL"},
        ], "count": 3}
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "summary.json"
            path.write_text(json.dumps(summary, indent=2) + "\n")
            slim_summary.slim_summary(json.loads(path.read_text()))  # sanity
            summary = json.loads(path.read_text())
            slim_summary.slim_summary(summary)
            # small classifier untouched, dense map trimmed, failure cell untouched
            self.assertEqual(summary["configs"][0]["output_values"]["i8"], [1, -2, 3])
            self.assertEqual(summary["configs"][1]["output_digest"]["i8"]["len"], 524288)
            self.assertEqual(len(summary["configs"][1]["output_values"]["i8"]),
                             slim_summary.CANARY_LEN)
            self.assertNotIn("output_digest", summary["configs"][2])


if __name__ == "__main__":
    unittest.main()
