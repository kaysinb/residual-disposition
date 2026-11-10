"""The second interpreter of another family (``follow_reader2``): its cells, input and checks."""

from residual_disposition import follow_reader2 as R


def test_cells_are_distinct_and_sized():
    rows = R.readings()
    assert len({r["salt"] for r in rows}) == len(rows)
    fixed = [r for r in rows if not r["conditional"]]
    assert sum(r["attempts"] for r in fixed) == 560
    assert sum(r["attempts"] for r in rows if r["conditional"]) == 360
    lines = {R.line_of(r) for r in rows if r["role"] == "L"}
    assert all(R.line_of(r) in lines for r in rows if r["conditional"])


def test_input_holds_the_prompt_and_package_verbatim():
    text = R.input_text("SYS\nline", "USER\nbody")
    assert text.startswith(R.WRAPPER)
    assert "\nSYS\nline\n" in text and text.endswith("\nUSER\nbody")


def test_session_problems():
    text = "x" * 60_000
    ok = [{"type": "item.completed", "item": {"type": "agent_message"}},
          {"type": "turn.completed", "usage": {"input_tokens": R.RUNNER_OVERHEAD_TOKENS + 12_000}}]
    assert R.session_problems(ok, text) == []
    short = [ok[0], {"type": "turn.completed",
                     "usage": {"input_tokens": R.RUNNER_OVERHEAD_TOKENS + 5_000}}]
    assert R.session_problems(short, text)
    tool = [{"type": "item.completed", "item": {"type": "command_execution"}}, *ok]
    assert R.session_problems(tool, text) == ["item command_execution"]


def test_count_needs_every_attempt():
    done = {"hypothesis": {"value": "h"},
            "labels": {"strict": {"value": ("names_behavior", True)}, "lenient": {"value": True}}}
    assert R.count([done] * 20, 20) == {"counted": 20, "n": 20}
    assert "pending" in R.count([done] * 19 + [{"hypothesis": {"due": 1}}], 20)
