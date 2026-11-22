"""Official SDK entry point: describe, train, or predict."""
import sys

from contest_harness.cli import main

if __name__ == "__main__":
    # The wire format is UTF-8 even on Windows hosts with another locale.
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv))
