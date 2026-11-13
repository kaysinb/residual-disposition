"""A clean control and "explain", read by the second interpreter
(``residual_disposition.follow_reader2_meaning``), by ``scripts/follow_reader2.py``'s pins, kit,
queue, import, send and result, on ``follow/reader2_meaning``.

    uv run python scripts/follow_reader2_meaning.py --pin
    uv run python scripts/follow_reader2_meaning.py --kit
    uv run python scripts/follow_reader2_meaning.py --queue
    uv run python scripts/follow_reader2_meaning.py --import DIR
    uv run python scripts/follow_reader2_meaning.py --send
    uv run python scripts/follow_reader2_meaning.py --result

The pins go to ``manifests/follow_reader2_meaning_inputs.json``, the kit and queue under
``<DATA_DIR>/follow_reader2_meaning``, the summary to ``runs/follow_reader2_meaning/summary.json``.
The reader, its isolation, the session check and the judges are ``follow_reader2``'s, unchanged.
"""

from __future__ import annotations

import sys

import follow_reader2 as S

from residual_disposition import follow_reader2_meaning as E


def main(argv: list[str] | None = None) -> int:
    S.R = E
    S.PINS = "manifests/follow_reader2_meaning_inputs.json"
    S.LABEL = "follow/reader2-meaning"
    S.ROOT = "follow_reader2_meaning"
    return S.main(argv)


if __name__ == "__main__":
    sys.exit(main())
