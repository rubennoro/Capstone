"""One-off: convert sim_evals/ from the record `sim_deploy.episode/v1` names to `sim_deploy.rollout/v2` (SCHEMA.md
"Versions": an eval trial is a rollout; "episode" stays for training demonstrations and policy_deploy's own files).

    python -m sim_deploy.evaltool migrate-rollouts [--dry-run] [--root sim_evals]

Per eval folder (any folder with an ``eval.yaml`` under the root, outside ``session/`` and ``_reports/``):
- ``episodes/`` -> ``rollouts/``; in each ``NNN/``: ``episode.json`` -> ``rollout.json`` (record_schema, ``gt`` path,
  real ``session.episode_index`` -> ``rollout_index``), ``session_episode.json`` -> ``session_rollout.json`` (content
  unchanged: policy_deploy's record);
- ``episodes.jsonl`` -> ``rollouts.jsonl`` (``pd_episode_id`` -> ``pd_rollout_id``, ``session_episode`` ->
  ``session_rollout``, ``episodes/NNN/...`` paths);
- ``eval.yaml``: ``schema`` eval/v2, ``record_schema`` rollout/v2, ``episodes`` / ``episode_seeds`` / ``only_episodes``
  / ``home.episode_start_O_T_tcp`` -> ``rollout*``, ``protocol.seed_rule.key`` "per episode n" -> "per rollout n" and
  ``protocol_sha`` recomputed (only when the stored sha matched the stored protocol); ``command`` is left as run;
- ``score.json`` (schema score/v2, keys and paths), ``metrics.json`` (``episodes``, ``episodes_recorded``),
  ``summary.html`` (links ``episodes/`` -> ``rollouts/``).
Also ``index.csv`` (header ``episodes`` -> ``rollouts``; the old file copied to ``_logs/`` first) and
``_batches/<id>/{plan.yaml, manifest.jsonl}`` (plan key, manifest run keys). Untouched: ``session/`` folders
(policy_deploy's ``manager/episodes.jsonl``), ``_reports/`` (old reports), ``_logs/`` run logs, folder names such as
``pick1-v1_ep002`` (a training episode index).

Idempotent: a folder or file already converted is skipped; an eval with both ``episodes/`` and ``rollouts/`` is
reported as a conflict and left alone. The log goes to ``<root>/_logs/migrate_rollouts_<time>.txt`` (not on --dry-run).
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

import yaml

from sim_deploy import REPO

OLD_RECORD, NEW_RECORD = "sim_deploy.episode/v1", "sim_deploy.rollout/v2"
OLD_EVAL, NEW_EVAL = "sim_deploy.eval/v1", "sim_deploy.eval/v2"
OLD_SCORE, NEW_SCORE = "sim_deploy.score/v1", "sim_deploy.score/v2"
OLD_KEY_TEXT, NEW_KEY_TEXT = "rng([seed, n]) per episode n", "rng([seed, n]) per rollout n"
SKIP_PARTS = {"session", "_reports"}
PATH_KEYS = ("gt", "video", "tap")


def sha16(doc) -> str:
    """gt_recorder.protocol_sha / import_real.sha16."""
    return hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]


def _ren(d: dict, pairs) -> list[str]:
    """Rename keys of d in place, keeping their order. Returns the renamed old keys."""
    if not isinstance(d, dict):
        return []
    done = [old for old, new in pairs if old in d and new not in d]
    if not done:
        return []
    m = dict(pairs)
    items = [(m[k] if k in done else k, v) for k, v in d.items()]
    d.clear()
    d.update(items)
    return done


def _path(v):
    return "rollouts/" + v[len("episodes/"):] if isinstance(v, str) and v.startswith("episodes/") else v


def _paths(d: dict, keys=PATH_KEYS) -> list[str]:
    out = []
    for k in keys:
        if isinstance(d, dict) and _path(d.get(k)) != d.get(k):
            d[k] = _path(d[k])
            out.append(k)
    return out


class Migration:
    def __init__(self, root: Path, dry: bool) -> None:
        self.root, self.dry = root, dry
        self.lines: list[str] = []
        self.n: dict[str, int] = {}

    def note(self, kind: str, text: str) -> None:
        self.n[kind] = self.n.get(kind, 0) + 1
        self.lines.append(f"{kind:<10} {text}")

    def rel(self, p: Path) -> str:
        try:
            return str(p.relative_to(self.root))
        except ValueError:
            return str(p)

    # ---------------------------------------------------------------------------------------------- writing
    def write(self, p: Path, text: str) -> None:
        if not self.dry:
            tmp = p.with_name(p.name + ".migrate_tmp")
            tmp.write_text(text)
            tmp.replace(p)

    def rename(self, a: Path, b: Path) -> None:
        if not self.dry:
            a.rename(b)

    def move_edit(self, old: Path, new: Path, text: str | None) -> None:
        """new = text (or old's bytes), then old removed."""
        if self.dry:
            return
        if text is None:
            old.rename(new)
        else:
            self.write(new, text)
            old.unlink()

    # -------------------------------------------------------------------------------------------- one record
    def rollout_line(self, rec: dict) -> list[str]:
        ch = _ren(rec, (("pd_episode_id", "pd_rollout_id"), ("session_episode", "session_rollout")))
        ch += _paths(rec)
        return ch

    def rollout_json(self, doc: dict) -> list[str]:
        ch = []
        if doc.get("record_schema") == OLD_RECORD:
            doc["record_schema"] = NEW_RECORD
            ch.append("record_schema")
        ch += _paths(doc)
        ch += [f"session.{k}" for k in _ren(doc.get("session"), (("episode_index", "rollout_index"),))]
        return ch

    # -------------------------------------------------------------------------------------------- one eval
    def eval_folder(self, ev: Path) -> None:
        r = self.rel(ev)
        old_dir, new_dir = ev / "episodes", ev / "rollouts"
        if old_dir.is_dir() and new_dir.exists():
            self.note("CONFLICT", f"{r}: both episodes/ and rollouts/ exist: left alone")
            return
        # per-rollout files, inside the (still old) folder, then the folder itself
        src_dir = old_dir if old_dir.is_dir() else new_dir
        if src_dir.is_dir():
            for d in sorted(x for x in src_dir.iterdir() if x.is_dir()):
                self.per_rollout(d, r)
        if old_dir.is_dir():
            self.rename(old_dir, new_dir)
            self.note("DIR", f"{r}/episodes -> rollouts")
        # rollouts.jsonl
        oj, nj = ev / "episodes.jsonl", ev / "rollouts.jsonl"
        if oj.is_file() and nj.exists():
            self.note("CONFLICT", f"{r}: both episodes.jsonl and rollouts.jsonl: left alone")
        elif oj.is_file():
            out, keys = [], set()
            for ln in oj.read_text().splitlines():
                if not ln.strip():
                    out.append(ln)
                    continue
                try:
                    rec = json.loads(ln)
                except ValueError:
                    out.append(ln)                       # a torn line stays as it was
                    continue
                keys.update(self.rollout_line(rec))
                out.append(json.dumps(rec))
            self.move_edit(oj, nj, "\n".join(out) + ("\n" if out else ""))
            self.note("FILE", f"{r}/episodes.jsonl -> rollouts.jsonl" + (f" (keys {sorted(keys)})" if keys else ""))
        self.eval_yaml(ev / "eval.yaml", r)
        self.score_json(ev / "score.json", r)
        self.metrics_json(ev / "metrics.json", r)
        self.summary_html(ev / "summary.html", r)

    def per_rollout(self, d: Path, r: str) -> None:
        rd = f"{r}/{d.parent.name}/{d.name}"
        for old, new in (("episode.json", "rollout.json"), ("session_episode.json", "session_rollout.json")):
            o, n = d / old, d / new
            if o.is_file() and n.exists():
                self.note("CONFLICT", f"{rd}: both {old} and {new}: left alone")
                continue
            if o.is_file():
                text = None
                ch: list[str] = []
                if old == "episode.json":
                    try:
                        doc = json.loads(o.read_text())
                        ch = self.rollout_json(doc)
                        text = json.dumps(doc, indent=1)
                    except ValueError:
                        text = None
                self.move_edit(o, n, text)
                self.note("FILE", f"{rd}/{old} -> {new}" + (f" (keys {ch})" if ch else ""))
            elif n.is_file() and new == "rollout.json":     # already moved: finish its keys (idempotent)
                try:
                    doc = json.loads(n.read_text())
                except ValueError:
                    continue
                ch = self.rollout_json(doc)
                if ch:
                    self.write(n, json.dumps(doc, indent=1))
                    self.note("EDIT", f"{rd}/{new} (keys {ch})")

    def eval_yaml(self, p: Path, r: str) -> None:
        text = p.read_text()
        doc = yaml.safe_load(text) or {}
        ch = []
        if doc.get("schema") == OLD_EVAL:
            doc["schema"] = NEW_EVAL
            ch.append("schema")
        if doc.get("record_schema") == OLD_RECORD:
            doc["record_schema"] = NEW_RECORD
            ch.append("record_schema")
        ch += _ren(doc, (("episodes", "rollouts"), ("episode_seeds", "rollout_seeds"),
                         ("only_episodes", "only_rollouts")))
        ch += [f"home.{k}" for k in _ren(doc.get("home"), (("episode_start_O_T_tcp", "rollout_start_O_T_tcp"),))]
        prot = doc.get("protocol")
        sr = prot.get("seed_rule") if isinstance(prot, dict) else None
        if isinstance(sr, dict) and sr.get("key") == OLD_KEY_TEXT:
            old_sha = doc.get("protocol_sha")
            matched = old_sha == sha16(prot)
            sr["key"] = NEW_KEY_TEXT
            ch.append("protocol.seed_rule.key")
            if matched:
                doc["protocol_sha"] = sha16(prot)
                ch.append(f"protocol_sha {old_sha} -> {doc['protocol_sha']}")
            else:
                ch.append(f"protocol_sha {old_sha} kept (it did not match the stored protocol)")
        if not ch:
            return
        self.write(p, yaml.safe_dump(doc, sort_keys=False))
        self.note("EDIT", f"{r}/eval.yaml ({'; '.join(ch)})")

    def score_json(self, p: Path, r: str) -> None:
        if not p.is_file():
            return
        try:
            doc = json.loads(p.read_text())
        except ValueError:
            self.note("SKIP", f"{r}/score.json: not JSON")
            return
        ch = []
        if doc.get("schema") == OLD_SCORE:
            doc["schema"] = NEW_SCORE
            ch.append("schema")
        ch += _ren(doc, (("invalid_episodes", "invalid_rollouts"), ("episodes", "rollouts")))
        ch += [f"summary.{k}" for k in _ren(doc.get("summary"), (
            ("episodes_recorded", "rollouts_recorded"), ("episodes_rated", "rollouts_rated"),
            ("table_contact_episodes", "table_contact_rollouts")))]
        ev = doc.get("eval") or {}
        if ev.get("record_schema") == OLD_RECORD:
            ev["record_schema"] = NEW_RECORD
            ch.append("eval.record_schema")
        n_paths = 0
        for e in doc.get("rollouts") or []:
            if not isinstance(e, dict):
                continue
            n_paths += len(_paths(e, ("video", "tap")))
            if e.get("source") == "episode.json (real)":
                e["source"] = "rollout.json (real)"
                n_paths += 1
            tc = e.get("table_contacts")
            if isinstance(tc, dict) and tc.get("source") == "episodes.jsonl (driver)":
                tc["source"] = "rollouts.jsonl (driver)"
                n_paths += 1
        if n_paths:
            ch.append(f"{n_paths} per-rollout paths / sources")
        if ch:
            self.write(p, json.dumps(doc, indent=1))
            self.note("EDIT", f"{r}/score.json ({'; '.join(ch)})")

    def metrics_json(self, p: Path, r: str) -> None:
        if not p.is_file():
            return
        try:
            doc = json.loads(p.read_text())
        except ValueError:
            self.note("SKIP", f"{r}/metrics.json: not JSON")
            return
        ch = _ren(doc, (("episodes", "rollouts"), ("episodes_recorded", "rollouts_recorded")))
        if ch:
            self.write(p, json.dumps(doc, indent=1))
            self.note("EDIT", f"{r}/metrics.json (keys {ch})")

    def summary_html(self, p: Path, r: str) -> None:
        if not p.is_file():
            return
        text = p.read_text()
        new, k = re.subn(r'href="episodes/', 'href="rollouts/', text)
        if k:
            self.write(p, new)
            self.note("EDIT", f"{r}/summary.html ({k} links)")

    # -------------------------------------------------------------------------------------- index and batches
    def index_csv(self, stamp: str) -> None:
        p = self.root / "index.csv"
        if not p.is_file():
            return
        text = p.read_text()
        head, sep, rest = text.partition("\n")
        cols = head.split(",")
        if "episodes" not in cols or "rollouts" in cols:
            return
        new_head = ",".join("rollouts" if c == "episodes" else c for c in cols)
        bak = self.root / "_logs" / f"index_before_migrate_rollouts_{stamp}.csv"
        if not self.dry:
            bak.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, bak)
        self.write(p, new_head + sep + rest)
        self.note("EDIT", f"index.csv header episodes -> rollouts ({rest.count(chr(10))} rows unchanged; "
                          f"backup {self.rel(bak)})")

    def batches(self) -> None:
        for b in sorted((self.root / "_batches").glob("*")):
            if not b.is_dir():
                continue
            plan = b / "plan.yaml"
            if plan.is_file():
                text = plan.read_text()
                new, k = re.subn(r"(?m)^episodes:", "rollouts:", text)
                if k:
                    self.write(plan, new)
                    self.note("EDIT", f"{self.rel(plan)} (key episodes)")
            man = b / "manifest.jsonl"
            if man.is_file():
                out, n = [], 0
                for ln in man.read_text().splitlines():
                    try:
                        rec = json.loads(ln)
                    except ValueError:
                        out.append(ln)
                        continue
                    k = _ren(rec.get("run"), (("episodes", "rollouts"), ("use_episodes", "use_rollouts"),
                                              ("only_episodes", "only_rollouts")))
                    k += _ren(rec, (("done_episodes", "done_rollouts"),))
                    n += bool(k)
                    out.append(json.dumps(rec))
                if n:
                    self.write(man, "\n".join(out) + "\n")
                    self.note("EDIT", f"{self.rel(man)} ({n} lines)")

    # ------------------------------------------------------------------------------------------------ all
    def evals(self) -> list[Path]:
        out = []
        for y in sorted(self.root.rglob("eval.yaml")):
            parts = set(y.relative_to(self.root).parts[:-1])
            if parts & SKIP_PARTS:
                continue
            out.append(y.parent)
        return out

    def run(self) -> Path | None:
        stamp = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        evs = self.evals()
        for ev in evs:
            self.eval_folder(ev)
        self.index_csv(stamp)
        self.batches()
        summary = (f"{len(evs)} eval folders; " + ", ".join(f"{k} {v}" for k, v in sorted(self.n.items()))
                   if self.n else f"{len(evs)} eval folders; nothing to change (already migrated)")
        head = [f"# migrate_rollouts {stamp} root {self.root}{' (dry run: nothing changed)' if self.dry else ''}",
                f"# {summary}"]
        text = "\n".join(head + self.lines) + "\n"
        if self.dry:
            print(text, end="")
            return None
        log = self.root / "_logs" / f"migrate_rollouts_{stamp}.txt"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(text)
        print("\n".join(head))
        print(f"log: {log}")
        return log


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sim_deploy.evaltool migrate-rollouts", description=__doc__.split("\n")[0])
    ap.add_argument("--dry-run", action="store_true", help="print what would change, change nothing")
    ap.add_argument("--root", default=str(REPO / "sim_evals"))
    a = ap.parse_args(argv)
    Migration(Path(a.root).expanduser().resolve(), a.dry_run).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
