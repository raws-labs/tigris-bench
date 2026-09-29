"""Device-to-device parity must compare real vectors, never pass on missing ones."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "validate_accuracy.py"


def _record(framework, output):
    return {"framework": framework, "kernel": "cmsis_nn", "board": "nucleo_h753zi",
            "model": "ds_cnn", "status": "ok", "output_values": {"i8": output}}


class ParityValidatorTest(unittest.TestCase):
    def _run(self, configs):
        with tempfile.TemporaryDirectory() as directory:
            summary = Path(directory) / "summary.json"
            summary.write_text(json.dumps({"configs": configs}))
            return subprocess.run([sys.executable, str(SCRIPT), str(summary)],
                                  capture_output=True, text=True)

    def test_identical_vectors_pass(self):
        result = self._run([_record("tigris", [1, -2, 3]), _record("tflm", [1, -2, 3])])
        self.assertEqual(result.returncode, 0, result.stdout)

    def test_successful_records_without_vectors_fail(self):
        result = self._run([_record("tigris", []), _record("tflm", [])])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("without OUTPUT_I8", result.stdout)

    def test_one_missing_vector_fails(self):
        result = self._run([_record("tigris", [1, -2, 3]), _record("tflm", [])])
        self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
