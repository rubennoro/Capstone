#!/bin/bash
# Fail if anything the iPhone path depends on changed relative to a baseline ref.
#
#   scripts/regression/check_protected.sh [REF] [--head]
#
# REF     baseline commit (default 0ddedef: the last commit before any sensor_rig work)
# --head  compare REF..HEAD (committed state only) instead of REF..working tree (default: includes uncommitted and
#         untracked, non-ignored files)
# Rules: scripts/regression/protected_paths.txt (syntax documented at its top). Exit 0 = clean, 1 = violations,
# 2 = usage / git error. Needs python3 with PyYAML (any env: system python, iphumi, bp); override with PYTHON=...
set -eo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$HERE" rev-parse --show-toplevel)"
exec "${PYTHON:-python3}" - "$REPO" "$HERE/protected_paths.txt" "$@" <<'PY'
import fnmatch, subprocess, sys

repo, rules_file, *argv = sys.argv[1:]
ref, head = "0ddedef", False
for a in argv:
    if a == "--head":
        head = True
    elif a in ("-h", "--help"):
        print("usage: check_protected.sh [REF] [--head]"); sys.exit(0)
    else:
        ref = a

def git(*args, check=True, text=True):
    p = subprocess.run(["git", "-C", repo, *args], capture_output=True, text=text)
    if check and p.returncode != 0:
        sys.exit(f"git {' '.join(args)} failed: {p.stderr.strip() if text else p.stderr}")
    return p.stdout

ref_sha = git("rev-parse", "--verify", ref + "^{commit}").strip()
target = "HEAD (" + git("rev-parse", "--short", "HEAD").strip() + ")" if head else "working tree"

protected, allow, special = [], [], {}
for line in open(rules_file):
    line = line.split("#", 1)[0].strip()
    if not line:
        continue
    if line.startswith("+"):
        allow.append(line[1:].strip())
    elif line.startswith("@"):
        kind, path = line[1:].split(None, 1)
        if kind not in ("additive", "yaml-append"):
            sys.exit(f"unknown rule @{kind}")
        special[path.strip()] = kind
    else:
        protected.append(line)

def is_protected(p):
    return any(p == r or (r.endswith("/") and p.startswith(r)) for r in protected) or p in special

# ---- changed paths: tracked (diff) + untracked non-ignored files
diff_args = ["diff", "--name-status", "--no-renames", ref_sha] + (["HEAD"] if head else [])
changes = {}
for row in git(*diff_args).splitlines():
    st, path = row.split("\t", 1)
    changes[path] = st[0]
if not head:
    for path in git("ls-files", "--others", "--exclude-standard").splitlines():
        changes[path] = "A?"  # untracked

fails, allowed, oks = [], [], []
for path, st in sorted(changes.items()):
    if not is_protected(path) or path in special:
        continue
    if st.startswith("A") and any(fnmatch.fnmatch(path, g) for g in allow):
        allowed.append(f"new file (allowed by + rule){' [untracked]' if st == 'A?' else ''}: {path}")
    else:
        what = {"M": "modified", "D": "DELETED", "A": "new file", "A?": "new untracked file", "T": "type changed"}
        fails.append(f"{what.get(st, st)}: {path}")

def new_bytes(path):
    if head:
        p = subprocess.run(["git", "-C", repo, "show", f"HEAD:{path}"], capture_output=True)
        return p.stdout if p.returncode == 0 else None
    try:
        return open(f"{repo}/{path}", "rb").read()
    except FileNotFoundError:
        return None

def old_bytes(path):
    p = subprocess.run(["git", "-C", repo, "show", f"{ref_sha}:{path}"], capture_output=True)
    return p.stdout if p.returncode == 0 else None

def yaml_compare(old, new, where=""):
    """-> (problems, appended): every old key must exist in new with an equal value."""
    probs, added = [], []
    if isinstance(old, dict) and isinstance(new, dict):
        for k in old:
            if k not in new:
                probs.append(f"key removed: {where}/{k}")
            else:
                p, a = yaml_compare(old[k], new[k], f"{where}/{k}")
                probs += p; added += a
        added += [f"{where}/{k}" for k in new if k not in old]
    elif old != new or type(old) is not type(new):
        probs.append(f"value changed at {where or '/'}: {old!r} -> {new!r}")
    return probs, added

for path, kind in sorted(special.items()):
    o, n = old_bytes(path), new_bytes(path)
    if o is None:
        fails.append(f"@{kind} {path}: not in baseline {ref}"); continue
    if n is None:
        fails.append(f"@{kind} {path}: DELETED"); continue
    if o == n:
        oks.append(f"@{kind} {path}: unchanged"); continue
    if kind == "additive":
        d = git("diff", "-U0", "--no-color", ref_sha, *(["HEAD"] if head else []), "--", path)
        removed = [l for l in d.splitlines() if l.startswith("-") and not l.startswith("---")]
        plus = [l[1:].strip() for l in d.splitlines() if l.startswith("+") and not l.startswith("+++")]
        if removed:
            fails.append(f"@additive {path}: {len(removed)} baseline line(s) removed/changed: {removed[:3]}")
        else:
            allowed.append(f"@additive {path}: only added lines: {plus}")
    else:  # yaml-append
        import yaml
        msgs = []
        if not n.startswith(o):
            k = next((i for i, (a, b) in enumerate(zip(o, n)) if a != b), min(len(o), len(n)))
            line = o[:k].count(b"\n") + 1
            msgs.append(f"baseline text is not a byte-identical prefix (first difference at line {line})")
        probs, added = yaml_compare(yaml.safe_load(o), yaml.safe_load(n))
        msgs += probs
        if msgs:
            fails.append(f"@yaml-append {path}: " + "; ".join(msgs))
        else:
            allowed.append(f"@yaml-append {path}: baseline bytes intact, appended keys: {added}")

# baseline sanity: every protected path exists at the ref
missing = [r for r in protected if not git("ls-tree", "--name-only", ref_sha, "--", r.rstrip("/"))]

print(f"iPhone-path protected files: baseline {ref} ({ref_sha[:9]}) vs {target}")
print(f"  rules: {len(protected)} protected paths, {len(allow)} allowed-addition globs, {len(special)} special")
for r in missing:
    print(f"  WARN  protected path not in baseline: {r}")
for m in oks:
    print(f"  OK    {m}")
for m in allowed:
    print(f"  INFO  {m}")
for m in fails:
    print(f"  FAIL  {m}")
print(f"RESULT: {'FAIL' if fails else 'PASS'} ({len(fails)} violation(s), {len(allowed)} allowed change(s))")
sys.exit(1 if fails else 0)
PY
