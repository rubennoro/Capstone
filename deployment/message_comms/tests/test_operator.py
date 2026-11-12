import io

import pytest

from fakes import FakeClock
from message_comms.episode_loop import LoopProgress, Phase
from message_comms.errors import ScriptError
from message_comms.keyboard_operator import KeyboardOperator
from message_comms.operator import Key
from message_comms.scripted_operator import ScriptedOperator, parse_script


def progress(phase, steps=0, episode=0):
    return LoopProgress(phase, episode, steps, 0)


def drive(op, states):
    """Feed LoopProgress values; collect the keys returned."""
    out = []
    for p in states:
        key = op.poll_key(p)
        if key is not None:
            out.append(key.value)
    return out


def load(text, clock=None):
    return ScriptedOperator(parse_script(text, "script.yaml"), io.StringIO(), clock or FakeClock())


def test_whole_episode():
    op = load("- {wait_phase: idle}\n- {key: c}\n- {wait_steps: 3}\n- {key: y}\n- {key: q}\n")
    keys = drive(op, [progress(Phase.STARTING), progress(Phase.IDLE), progress(Phase.RESETTING),
                      progress(Phase.RUNNING, 1), progress(Phase.RUNNING, 3), progress(Phase.IDLE)])
    assert keys == ["c", "y", "q"] and op.done()


def test_wait_steps_needs_running():
    op = load("- {key: c}\n- {wait_steps: 2}\n- {key: y}\n- {key: q}\n")
    assert drive(op, [progress(Phase.IDLE, 5), progress(Phase.RESETTING, 5)]) == ["c"]


def test_wait_s_by_the_clock():
    clock = FakeClock()
    op = load("- {wait_s: 1.0}\n- {key: q}\n", clock)
    assert op.poll_key(progress(Phase.IDLE)) is None
    clock.advance(0.9)
    assert op.poll_key(progress(Phase.IDLE)) is None
    clock.advance(0.2)
    assert op.poll_key(progress(Phase.IDLE)) is Key.QUIT


def test_two_episodes_with_stop_home_and_fail():
    text = ("- {wait_phase: idle}\n- {key: c}\n- {wait_steps: 1}\n- {key: y}\n"
            "- {wait_phase: idle}\n- {key: c}\n- {wait_steps: 1}\n- {key: s}\n"
            "- {wait_phase: idle}\n- {key: h}\n- {wait_phase: idle}\n"
            "- {key: c}\n- {wait_steps: 1}\n- {key: n}\n- {wait_phase: idle}\n- {key: q}\n")
    op = load(text)
    seq = [progress(Phase.IDLE), progress(Phase.RUNNING, 1), progress(Phase.IDLE), progress(Phase.RESETTING),
           progress(Phase.RUNNING, 1), progress(Phase.IDLE), progress(Phase.HOMING), progress(Phase.IDLE),
           progress(Phase.RUNNING, 1), progress(Phase.IDLE)]
    keys = []
    for p in seq:
        for _ in range(3):
            k = op.poll_key(p)
            if k is not None:
                keys.append(k.value)
    assert keys == ["c", "y", "c", "s", "h", "c", "n", "q"] and op.done()


@pytest.mark.parametrize("text, words", [
    ("- {key: c}\n- {press: y}\n- {key: q}\n", "script.yaml:2: unknown entry 'press'"),
    ("- {key: c}\n- {key: x}\n- {key: q}\n", "script.yaml:2: unknown key 'x'"),
    ("- {key: c}\n- {key: s}\n", "script.yaml:2: the last entry must be {key: q}"),
    ("- {wait_phase: flying}\n- {key: q}\n", "script.yaml:1: unknown phase 'flying'"),
    ("- {wait_steps: -1}\n- {key: q}\n", "script.yaml:1: wait_steps needs"),
    ("- {wait_s: fast}\n- {key: q}\n", "script.yaml:1: wait_s needs"),
    ("key: q\n", "a YAML list"),
    ("", "a YAML list"),
])
def test_bad_scripts_refused_with_the_line(text, words):
    with pytest.raises(ScriptError) as e:
        parse_script(text, "script.yaml")
    assert words in str(e.value)


def test_load_from_file(tmp_path):
    path = tmp_path / "s.yaml"
    path.write_text("- {key: q}\n")
    op = ScriptedOperator.load(path, io.StringIO(), FakeClock())
    assert op.poll_key(progress(Phase.IDLE)) is Key.QUIT
    with pytest.raises(ScriptError):
        ScriptedOperator.load(tmp_path / "missing.yaml", io.StringIO(), FakeClock())


def wait_keys(op, n):
    import time
    keys, end = [], time.monotonic() + 5
    while len(keys) < n and time.monotonic() < end:
        k = op.poll_key(progress(Phase.IDLE))
        if k is not None:
            keys.append(k)
        time.sleep(0.001)
    return keys


def test_keyboard_from_a_pipe():
    out = io.StringIO()
    op = KeyboardOperator(io.StringIO("c\nyes\n\n  n\nS\nhome\nq\n"), out)
    op.start()
    assert wait_keys(op, 6) == [Key.START, Key.SUCCESS, Key.FAIL, Key.STOP, Key.HOME, Key.QUIT]
    op.close()


def test_keyboard_unknown_key_shows_the_key_list():
    out = io.StringIO()
    op = KeyboardOperator(io.StringIO("x\nq\n"), out)
    op.start()
    assert wait_keys(op, 1) == [Key.QUIT]
    assert "not a key" in out.getvalue() and "c start" in out.getvalue()


def test_status_rewritten_in_place_and_messages_above():
    out = io.StringIO()
    op = KeyboardOperator(io.StringIO(""), out)
    op.show_status("[idle] a")
    op.show_message("hello")
    assert out.getvalue() == "\r[idle] a\x1b[K\r\x1b[Khello\n[idle] a"
