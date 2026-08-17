"""Assert the harness self-tests RAN, rather than merely not failing.

    uv run python scripts/assert_selftests_ran.py

Every data-dependent test in this repo guards on `skipif(not PARQUET.exists())`
-- ten files, plus a module-level `pytestmark` in `test_harness_selftest.py`.
That is correct locally: a contributor without a 500 MB data cache should get
skips rather than errors.

On a runner it is a trap. `data/` is gitignored, so a fresh checkout has no
parquet, and every one of the checks CLAUDE.md names as *the reason any number
in this repo is trustworthy* skips silently while pytest reports success. A
green build would then mean nothing at all, and would mean it convincingly.

So this asserts presence, not absence of failure. A skip is a failure here.

The four checks, quoting CLAUDE.md:

  - A result-peeking cheater must score RPS < 0.01 and be flagged.
  - A deliberately poisoned split must raise from `assert_no_leakage`.
  - Betting the de-vigged market back into its own prices must place zero bets
    at any positive EV threshold, and betting all of them must return exactly
    minus the margin.
  - The market must land at RPS 0.19-0.21.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# One entry per claim in CLAUDE.md's "Verification that must keep passing".
REQUIRED: dict[str, str] = {
    "tests/test_harness_selftest.py::test_cheater_is_flagged_as_impossible":
        "a result-peeking cheater scores RPS < 0.01 and is flagged",
    "tests/test_harness_selftest.py::test_leakage_guard_catches_a_deliberately_leaky_split":
        "a poisoned split raises from assert_no_leakage",
    "tests/test_harness_selftest.py::test_market_bet_against_its_own_price_places_no_bets":
        "the de-vigged market bet into its own prices places zero bets",
    "tests/test_harness_selftest.py::test_forced_market_betting_loses_approximately_the_vig":
        "betting the whole market returns exactly minus the margin",
    "tests/test_harness_selftest.py::test_the_market_lands_where_the_literature_says_it_should":
        "the market lands at RPS 0.19-0.21",
}


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        report = Path(tmp) / "report.xml"
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", *REQUIRED, "-q", "--tb=short",
             f"--junitxml={report}"],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        if not report.exists():
            print(proc.stdout[-4000:])
            print(proc.stderr[-2000:], file=sys.stderr)
            print("FAIL: pytest produced no report at all", file=sys.stderr)
            return 1
        root = ET.parse(report).getroot()

    def as_key(nodeid: str) -> tuple[str, str]:
        """'tests/x.py::test_y' -> ('tests.x', 'test_y').

        The junit report identifies a case by `classname` and `name`; it carries
        no `file` attribute, so the node id has to be converted rather than
        compared directly.
        """
        path, _, name = nodeid.partition("::")
        return path.removesuffix(".py").replace("/", "."), name

    seen: dict[tuple[str, str], str] = {}
    for case in root.iter("testcase"):
        node = (str(case.get("classname")), str(case.get("name")))
        if case.find("skipped") is not None:
            seen[node] = "SKIPPED"
        elif case.find("failure") is not None or case.find("error") is not None:
            seen[node] = "FAILED"
        else:
            seen[node] = "ran"

    problems = []
    for nodeid, claim in REQUIRED.items():
        status = seen.get(as_key(nodeid), "NOT COLLECTED")
        print(f"  {status:14s} {claim}")
        if status != "ran":
            problems.append(f"{nodeid}: {status}")

    if problems:
        print()
        print("FAIL -- the checks that make this repo's numbers trustworthy did not run:",
              file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        print("\nOn a runner this almost always means data/processed/matches.parquet is\n"
              "absent, so the module-level skipif fired. Build the corpus before\n"
              "trusting the suite.", file=sys.stderr)
        return 1

    print(f"\nOK -- all {len(REQUIRED)} harness self-tests ran and passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
