"""The "EL" fragment-family deletion read by the second interpreter
(``residual_disposition.follow_reader2_el``), by ``scripts/follow_reader2.py``'s pins, kit, queue,
import, send and result, on ``follow/reader2_el``.

    uv run python scripts/follow_reader2_el.py --pin
    uv run python scripts/follow_reader2_el.py --kit
    uv run python scripts/follow_reader2_el.py --queue
    uv run python scripts/follow_reader2_el.py --import DIR
    uv run python scripts/follow_reader2_el.py --send
    uv run python scripts/follow_reader2_el.py --result

The pins go to ``manifests/follow_reader2_el_inputs.json``, the kit and queue under
``<DATA_DIR>/follow_reader2_el``, the summary to ``runs/follow_reader2_el/summary.json``. The
reader, its isolation, the session check and the judges are ``follow_reader2``'s, unchanged.
"""

from __future__ import annotations

import sys

import follow_reader2 as S

from residual_disposition import follow_reader2_el as E


def main(argv: list[str] | None = None) -> int:
    S.R = E
    S.PINS = "manifests/follow_reader2_el_inputs.json"
    S.LABEL = "follow/reader2-el"
    S.ROOT = "follow_reader2_el"
    return S.main(argv)


if __name__ == "__main__":
    sys.exit(main())
