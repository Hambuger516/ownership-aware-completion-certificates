"""Run both CPU suites from this checkout, independently of the caller's cwd.

No model or CUDA workload is started. The protocol test uses a localhost socket.
Optional dependency skips are retained and reported separately from passes.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
SUITES = ("package/tests", "verification")

# Each suite gets its own interpreter, so their test-module names cannot collide.
# Insert absolute paths explicitly: PYTHONPATH alone is ignored by isolated
# interpreters, and an installed copy must not satisfy imports for this checkout.
CHILD = r"""
import json
from pathlib import Path
import sys
import unittest

root, relative_suite, report = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
sys.path[:0] = [str(root / "package"), str(root), str(root / "verification")]
import prc_nc
if Path(prc_nc.__file__).resolve().parent != root / "package" / "prc_nc":
    raise RuntimeError("prc_nc was not imported from the requested checkout")
suite = unittest.TestLoader().discover(
    start_dir=str(root / relative_suite), pattern="test_*.py"
)
result = unittest.TextTestRunner(verbosity=2).run(suite)
report.write_text(json.dumps({
    "suite": relative_suite,
    "tests": result.testsRun,
    "failures": len(result.failures),
    "errors": len(result.errors),
    "skipped": len(result.skipped),
    "expected_failures": len(result.expectedFailures),
    "unexpected_successes": len(result.unexpectedSuccesses),
    "successful": result.wasSuccessful(),
}) + "\n")
raise SystemExit(0 if result.wasSuccessful() else 1)
"""


def main() -> int:
    if sys.version_info < (3, 12):
        print("CPU tests require Python 3.12 or later.", file=sys.stderr)
        return 2

    environment = dict(os.environ)
    environment.update(
        PYTHONPATH=os.pathsep.join((str(ROOT / "package"), str(ROOT))),
        PYTHONDONTWRITEBYTECODE="1",
        CUDA_VISIBLE_DEVICES="",
    )
    results = []
    with tempfile.TemporaryDirectory(prefix="completion-cpu-tests-") as temp:
        for index, suite in enumerate(SUITES):
            report = Path(temp) / f"suite-{index}.json"
            process = subprocess.run(
                [sys.executable, "-B", "-c", CHILD, str(ROOT), suite, str(report)],
                cwd=ROOT,
                env=environment,
                check=False,
            )
            if report.is_file():
                result = json.loads(report.read_text())
            else:
                result = {"suite": suite, "tests": None, "successful": False}
            result["exit_code"] = process.returncode
            results.append(result)

    successful = all(result["successful"] and result["exit_code"] == 0
                     for result in results)
    summary = {
        "python": sys.version.split()[0],
        "suites": results,
        "tests": sum(result["tests"] or 0 for result in results),
        "successful": successful,
    }
    print("CPU_TEST_SUMMARY=" + json.dumps(summary, sort_keys=True), flush=True)
    return 0 if successful else 1


if __name__ == "__main__":
    raise SystemExit(main())
