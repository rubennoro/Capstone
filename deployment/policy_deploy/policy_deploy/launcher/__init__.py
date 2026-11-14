"""policy_deploy.launcher: `policy_deploy check | run [--station FILE] [--mock]` for the laptop's processes
(STRUCTURE.md section 9).

It imports only the standard library, yaml, policy_deploy.contracts and deploy_contracts, never a process subpackage:
the processes run as subprocesses (`python -m policy_deploy.<name>`), so torch, OpenCV or PyAV never load here.
Importing this package imports nothing else.
"""
