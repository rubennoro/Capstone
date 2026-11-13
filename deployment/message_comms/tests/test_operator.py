import io

import pytest

from fakes import FakeClock
from message_comms.episode_loop import LoopProgress, Phase
from message_comms.errors import ScriptError, ScriptTimeout
from message_comms.keyboard_operator import KeyboardOperator
from message_comms.operator import Key
from message_comms.scripted_operator import ScriptedOperator, parse_script


LIMIT_S = 30.0


def progress(phase, steps=0, episode=0, busy=False):
    return LoopProgress(phase, episode, steps, 0, busy)


def drive(op, states):
    """Feed LoopProgress values; collect the keys returned."""
    out = []
    for p in states:
        key = op.poll_key(p)
        if key is not None:
            out.append(key.value)
    return out


def load(text, clock=None, limit_s=LIMIT_S):
    return ScriptedOperator(parse_script(text, "script.yaml"), io.StringIO(), clock or FakeClock(), limit_s,
                            "script.yaml")


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
    op = ScriptedOperator.load(path, io.StringIO(), FakeClock(), LIMIT_S)
    assert op.poll_key(progress(Phase.IDLE)) is Key.QUIT
    with pytest.raises(ScriptError):
        ScriptedOperator.load(tmp_path / "missing.yaml", io.StringIO(), FakeClock(), LIMIT_S)


# ---- time limits and refused keys -----------------------------------------------------------------------------------

@pytest.mark.parametrize("text, seen, words", [
    ("- {wait_phase: idle}\n- {key: q}\n", progress(Phase.RUNNING, 3),
     "script.yaml:1: wait_phase idle: not reached (phase running) after 2.0 s"),
    ("- {key: c}\n- {wait_steps: 5}\n- {key: q}\n", progress(Phase.RUNNING, 3),
     "script.yaml:2: wait_steps 5: not reached (phase running, 3 sent) after 2.0 s"),
])
def test_a_wait_that_never_ends_times_out_naming_the_entry(text, seen, words):
    clock = FakeClock()
    op = load(text, clock, limit_s=2.0)
    drive(op, [seen, seen])                              # the wait begins at the first poll that reaches it
    clock.advance(1.9)
    assert op.poll_key(seen) is None
    clock.advance(0.1)
    with pytest.raises(ScriptTimeout) as e:
        op.poll_key(seen)
    assert words in str(e.value) and "operator_script_wait_s" in str(e.value)


def test_wait_s_has_no_time_limit():
    clock = FakeClock()
    op = load("- {wait_s: 5.0}\n- {key: q}\n", clock, limit_s=1.0)
    assert op.poll_key(progress(Phase.IDLE)) is None
    clock.advance(5.0)
    assert op.poll_key(progress(Phase.IDLE)) is Key.QUIT


def test_a_refused_key_is_retried_once_something_changes():
    op = load("- {key: h}\n- {key: q}\n")
    busy = progress(Phase.IDLE, 5, 1, busy=True)
    assert op.poll_key(busy) is Key.HOME
    op.key_refused(Key.HOME, "h: waiting for the last policy step to finish")
    assert not op.done()
    assert op.poll_key(busy) is None                     # nothing changed: no retry (no hint every cycle)
    assert op.poll_key(progress(Phase.IDLE, 5, 1)) is Key.HOME     # the step finished: retried
    assert op.poll_key(progress(Phase.IDLE, 5, 1)) is Key.QUIT and op.done()


def test_a_key_refused_for_too_long_times_out():
    clock = FakeClock()
    op = load("- {wait_phase: idle}\n- {key: y}\n- {key: q}\n", clock, limit_s=1.0)
    idle = progress(Phase.IDLE)
    assert op.poll_key(idle) is Key.SUCCESS
    op.key_refused(Key.SUCCESS, "y: no episode running (c start, h home, q quit)")
    clock.advance(0.5)
    assert op.poll_key(idle) is None
    clock.advance(0.5)
    with pytest.raises(ScriptTimeout) as e:
        op.poll_key(idle)
    assert "script.yaml:2: key y: still refused (y: no episode running (c start, h home, q quit)) after 1.0 s" \
        in str(e.value)


def test_the_time_limit_of_a_retried_key_counts_from_its_first_try():
    clock = FakeClock()
    op = load("- {key: c}\n- {key: q}\n", clock, limit_s=1.0)
    for steps in range(1, 20):                           # refused in every state: retried each time, then the limit
        try:
            key = op.poll_key(progress(Phase.RUNNING, steps, 1))
        except ScriptTimeout:
            break
        assert key is Key.START
        op.key_refused(key, "c: an episode is running (y success, n fail, s stop)")
        clock.advance(0.1)
    else:
        raise AssertionError("no time limit")
    assert steps == 11


def test_key_refused_for_another_key_is_ignored():
    op = load("- {key: c}\n- {key: q}\n")
    assert op.poll_key(progress(Phase.IDLE)) is Key.START
    op.key_refused(Key.HOME, "h: x")
    assert op.poll_key(progress(Phase.IDLE)) is Key.QUIT


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
