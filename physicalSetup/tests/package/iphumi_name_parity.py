#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Naming parity with iPhUMI's own parsers (T3). Stdlib only; run as `python3 -I iphumi_name_parity.py <repo_root>`
with package names on stdin, one per line. Prints one tab-separated line per name:
    <name>\t<accepted 0|1>\t<session>\t<recording_type>\t<side>\t<type_from_title>\t<session_from_title>

The rules below mirror, line for line, the cited iPhUMI code; before using them the script checks that those exact
source lines are still present in the repo (so a change in iPhUMI breaks this test instead of silently diverging).
"""
import os
import sys

REPO = sys.argv[1]
GROUP = os.path.join(REPO, "iPhUMI/python_package/iphumi/demonstration_processing/process_stages/group.py")
UTIL = os.path.join(REPO, "iPhUMI/python_package/iphumi/demonstration_processing/utils/generic_util.py")

CITED = [
    (GROUP, 36, "split = demonstration_name.split('_')"),
    (GROUP, 37, "if len(split) == 5:"),
    (GROUP, 38, "demonstration_time_str, demonstration_randomizer, demonstration_session_name, recording_type, side = split"),
    (GROUP, 39, "demonstration_title = f'{demonstration_time_str}_{demonstration_randomizer}_{demonstration_session_name}_{recording_type}'"),
    (UTIL, 308, "return demonstration_name.split('_')[-1]"),
    (UTIL, 368, "split = demonstration_title.split('_')"),
    (UTIL, 369, "if len(split) == 4:"),
    (UTIL, 370, "demonstration_time_str, demonstration_randomizer, demonstration_session_name, recording_type = split"),
]


def check_sources():
    for path, line, text in CITED:
        with open(path, encoding="utf-8") as f:
            lines = f.read().split("\n")
        got = lines[line - 1].strip() if line - 1 < len(lines) else "<eof>"
        if got != text:
            print(f"SOURCE CHANGED {os.path.basename(path)}:{line}: expected {text!r}, got {got!r}", file=sys.stderr)
            sys.exit(3)


def group_rule(name):
    # group.py:36-39
    split = name.split('_')
    if len(split) == 5:
        t, r, session, recording_type, side = split
        title = f'{t}_{r}_{session}_{recording_type}'
        return session, recording_type, side, title
    return None


def type_from_dir(title):
    # generic_util.py:306-308 get_demonstration_type(os.path.basename(dir))
    return title.split('_')[-1]


def session_from_title(title):
    # generic_util.py:368-370 (session filter)
    split = title.split('_')
    if len(split) == 4:
        return split[2]
    return None


def main():
    check_sources()
    for raw in sys.stdin.read().split("\n"):
        name = raw.strip()
        if not name:
            continue
        g = group_rule(name)
        if g is None:
            print(f"{name}\t0\t\t\t\t\t")
            continue
        session, recording_type, side, title = g
        s2 = session_from_title(title)
        print(f"{name}\t1\t{session}\t{recording_type}\t{side}\t{type_from_dir(title)}\t{s2 if s2 is not None else ''}")


if __name__ == "__main__":
    main()
