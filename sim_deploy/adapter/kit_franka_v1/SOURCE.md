# Finger kit iphumi_umi_soft_franka_v1

- kit sha256: 88fc545321953122d28a639a673d634dc513849b5a01dc79fc8423940d2ce188

| relative path | bytes | sha256 |
|---|---|---|
| kit.yaml | 3684 | 75487db521a8f41e5926aba2cc2f624f2c4a0a4970037f30b49cadb6b2fca4e3 |
| adapter_left.stl | 193084 | 8f4e6d832ecb2d9b20ea51402b57b465bf7cd4012ab3c8862a5c4a2206b51c04 |
| adapter_right.stl | 193084 | 5efa7792f27bde98e2d58d926ee6936dfa4c1bc533b853faed26acf4e0991a10 |
| finger_holder_left.stl | 373384 | d8793e2927e8d9a8033959468751d30cc5475be7367223d14412ab67b122b120 |
| finger_holder_right.stl | 372384 | 3b25d8a70b86cb09f888ae9b4087666fb64bb465c29913500dcbe2c37e346be6 |
| finger_left.stl | 97284 | f9453d03f6395da3da1f44402c609e813021596121925b15d72bf152de2abc40 |
| finger_right.stl | 97284 | 1636beb2d80b5ebb784ea377903508f76e2ef5eb1cd190a7452e84e3baceee3f |

## Origin

- finger_*.stl, finger_holder_*.stl: copied unchanged from controlSW/franka_sim/franka_sim/assets/finger_kits/iphumi_umi_soft_v1 (see its SOURCE.md).
- adapter_*.stl: sim_deploy/adapter/adapter.py (E3), binary STL in metres, asm coordinates of each side.
- Re-generate with the script (it re-pins this file); then update tooling kit_sha256.
