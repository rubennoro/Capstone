"""A2fix: Kin collision link groups are the arm's real links on both arms (FR3 was shifted by one link in A2).

Run: cd controlSW/franka_sim && env -u PYTHONPATH $RE -m pytest -q tests/test_kin_groups.py
"""

from __future__ import annotations

import re

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from franka_deploy.station import PROFILES, load_station  # noqa: E402
from franka_sim import kinematics as K  # noqa: E402
from franka_sim import model as M  # noqa: E402

ROBOTS = ("panda", "fr3")


def _build(robot, sets=()):
    spec = {k: dict(v) for k, v in PROFILES["sim-iphone"].items()}
    spec["station"]["robot"] = robot
    return M.build_model(load_station(spec, sets=list(sets)))


@pytest.fixture(scope="module", params=ROBOTS)
def kin(request):
    return K.Kin(_build(request.param), "flange")


def _by_name(kin) -> dict[int, set[int]]:
    """Collidable geoms grouped by the link index in their geom/body name (``link<i>...`` / ``fr3_link<i>...``)."""
    m, out = kin.m, {}
    for g in range(m.ngeom):
        if not (m.geom_contype[g] or m.geom_conaffinity[g]):
            continue
        mt = re.fullmatch(r"(?:fr3_)?link(\d)(?:[:_].*)?", kin._gname(g))
        if mt:
            out.setdefault(int(mt.group(1)), set()).add(g)
    return out


def _body_name(kin, g) -> str:
    return kin.m.body(int(kin.m.geom_bodyid[g])).name


def test_link_groups_match_link_names(kin):
    byname = _by_name(kin)
    for i in range(8):
        grp = set(kin.groups[f"link{i}"])
        assert grp, f"link{i} has no collision geoms"
        assert grp == byname.get(i, set()), f"link{i}: {[kin._gname(x) for x in grp]}"
        assert {re.sub(r"^fr3_", "", _body_name(kin, x)) for x in grp} == {f"link{i}"}
    # arm = links 1-7 (link 7 included, base excluded)
    assert set(kin.groups["arm"]) == set().union(*(byname[i] for i in range(1, 8)))
    assert not set(kin.groups["arm"]) & byname[0]


def test_link7_checked_and_base_off_table(kin):
    g, P = kin.groups, kin._pairs
    l0, l7 = set(g["link0"]), set(g["link7"])
    table_flat = {x for p in P["table"] for x in p}
    self_flat = {x for p in P["self"] for x in p}
    assert l7 <= table_flat and l7 & self_flat                       # link 7 is checked
    assert not table_flat & l0                                       # base stands on the table
    # at home nothing touches the table (A2 FR3 reported fr3_link0 vs table on every configuration)
    assert kin.contacts(kin.q_home, groups=("table",)) == []
    assert kin.min_distance(kin.q_home, [("arm", "table")]) > 0


def test_tooling_pairs_use_real_links(kin):
    g = kin.groups
    tool = set(g["tooling"])
    others = {b for p in kin._pairs["tooling"] for b in p if b not in tool}
    links = {_body_name(kin, b) for b in others}
    assert {re.sub(r"^fr3_", "", n) for n in links} <= {f"link{i}" for i in range(7)}
    assert not others & set(g["link7"]) and not others & set(g["hand"])
    for i in (5, 6):
        names = {re.sub(r"^fr3_", "", _body_name(kin, b)) for b in g[f"link{i}"]}
        assert names == {f"link{i}"}
        # the tooling collision group really contains tooling vs this link (link 6 included on both arms)
        assert {b for p in kin._pairs["tooling"] for b in p} >= set(g[f"link{i}"])
        # min_distance over the group equals the direct min over the tooling x link geom pairs
        d_grp = kin.min_distance(kin.q_home, [("tooling", f"link{i}")])
        kin._set(kin.q_home)
        d_geo = min(kin._dist(a, b, 1.0) for a in g["tooling"] for b in g[f"link{i}"])
        assert d_grp == pytest.approx(min(d_geo, 1.0))


def _old_groups(kin):
    """A2's link grouping (walk link7 -> world), kept here to prove Panda is unchanged by the fix."""
    m = kin.m
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, kin.sm.names["bodies"]["link7"])
    chain = [b]
    while m.body_parentid[chain[-1]] > 0:
        chain.append(int(m.body_parentid[chain[-1]]))
    chain = chain[::-1]
    coll = lambda g: bool(m.geom_contype[g] or m.geom_conaffinity[g])  # noqa: E731
    return {f"link{i}": [g for g in range(m.ngeom) if m.geom_bodyid[g] == bb and coll(g)] for i, bb in enumerate(chain)}


def test_panda_groups_identical_to_a2():
    kin = K.Kin(_build("panda"), "flange")
    old = _old_groups(kin)
    for i in range(8):
        assert kin.groups[f"link{i}"] == old[f"link{i}"]
    assert kin.groups["arm"] == [x for i in range(1, 8) for x in old[f"link{i}"]]


def test_fr3_was_shifted_and_now_is_not():
    kin = K.Kin(_build("fr3"), "flange")
    old = _old_groups(kin)
    assert old["link1"] == kin.groups["link0"]                       # the A2 bug, documented
    assert kin.groups["link7"] and set(kin.groups["link7"]) != set(old.get("link7", []))


@pytest.mark.parametrize("robot", ROBOTS)
def test_demo_like_configs_no_base_table_hit(robot):
    """Home plus a few reach-down configurations: no table contact reported for link 0 on either arm."""
    kin = K.Kin(_build(robot), "flange")
    l0 = {kin._gname(x) for x in kin.groups["link0"]}
    qs = [kin.q_home] + [np.clip(kin.q_home + d, kin.q_lo, kin.q_hi) for d in
                         (np.array([0.5, 0.3, 0, -0.3, 0, 0.3, 0]), np.array([-0.6, 0.4, 0.2, -0.2, 0.3, 0.2, -0.5]),
                          np.array([0.0, 0.5, 0, 0.3, 0, -0.2, 0]))]
    for q in qs:
        assert not any(c["a"] in l0 or c["b"] in l0 for c in kin.contacts(q, margin=0.01, groups=("table",)))
