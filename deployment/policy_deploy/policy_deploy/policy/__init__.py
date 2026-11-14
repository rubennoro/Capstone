"""The policy process: serves the trained policy (or the hold-still stand-in) over C4 (STRUCTURE.md 5).

Importing this package imports nothing heavy (no torch, no behavior_prompting): only `policy_factory.build_policy`
imports a torch policy, and only when it is chosen.
"""
