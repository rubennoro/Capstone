"""python -m sim_deploy.model.inspect <scene.xml>: load the file in this fresh process, print JSON (describe() at the
home key + the reload checks). Used by build.py; also handy to check what the sim PC / sim Pi will see."""

import json
import sys

import mujoco
import numpy as np

from sim_deploy.model.build import _site_T, checks_of, describe


def main(path: str) -> int:
    m = mujoco.MjModel.from_xml_path(path)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
    mujoco.mj_forward(m, d)
    out = {"describe": describe(m, d), "checks": checks_of(m), "qpos_home": d.qpos.tolist(),
           "O_T_flange": _site_T(d, m.site("flange").id).tolist(), "O_T_tcp": _site_T(d, m.site("tcp").id).tolist(),
           "ncon_home": int(d.ncon),
           "contacts_home": [[m.geom(c.geom1).name or str(c.geom1), m.geom(c.geom2).name or str(c.geom2), float(c.dist)]
                             for c in d.contact[: d.ncon]]}
    json.dump(out, sys.stdout, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else float(o))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
