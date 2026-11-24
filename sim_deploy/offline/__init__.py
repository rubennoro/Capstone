"""sim_deploy.offline: W1 (PLAN.md §11), the deploy path vs training on real recorded images, offline (no rendering).

S1 harness modules: common (paths, GPU lock, copy version), episodes (package index <-> Pi episode <-> slot, Pi-format
staging, run folders, splits), deploy_path (light env: recording -> the copy's frames + manager -> C4 Observation),
direct_path (torch env, CPU: the training dataset at (episode, T)), build_obs (the cache obs/<dataset>/ep<ii>_r<row>),
compare (metrics, montage), policy_runner (torch env, GPU: one checkpoint built the deploy way, seeded outputs),
approach (first-chunk approach), smoke (the step 1 smoke). Other agents' modules live here too (prompting_*,
make_settings, bench_checkpoints, c4_overhead).
"""
