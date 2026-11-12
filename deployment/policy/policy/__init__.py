"""The policy process: serves the trained policy over C4 (policy/STRUCTURE.md).

Importing this package imports nothing heavy (no torch, no behavior_prompting): only `policy_factory.build_policy`
imports a torch policy, and only when it is chosen.
"""
