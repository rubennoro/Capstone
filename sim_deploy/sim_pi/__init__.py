"""sim_deploy.sim_pi (E5): a live stand-in for the Raspberry Pi 5 camera sender (DESIGN.md section 6, "E5: as built").

It mirrors the sim PC's physics state (``sim.state/v1``), renders the IMX519 (``main``) and IMX219 (``ultrawide``)
views the way the real sensor chain delivers them (native-resolution render, lens by remap, INTER_AREA to 640x480,
JPEG q90) and publishes ``pi_stream/v1`` on ZMQ PUB :5555, so the copied ``policy_deploy`` frames process
(``launcher.fake_pi: false``) receives it exactly as it would from the real Pi.

    python -m sim_deploy.sim_pi --station FILE [--model scene.xml] [--state tcp://127.0.0.1:5650]
    python -m sim_deploy.sim_pi.state_stub --model scene.xml --bind tcp://127.0.0.1:5660   # smoke runs only
    python -m sim_deploy.sim_pi.handheld_check --out DIR                                      # check G3

Every camera constant comes from the copy's deployed rig file (``deployment/contracts/rigs/rig_pi_v3_640.yaml``),
loaded with the copy's own loader (``_deploy.py``). See README.md.
"""
