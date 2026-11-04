# Finger kit iphumi_umi_soft_v1

- kit sha256: 1e7778bcec2b1287387275c9d18839e6d7e5cab999b57432ca6442fc8e0f34d4

| relative path | bytes | sha256 |
|---|---|---|
| kit.yaml | 4940 | 0e12d1f6d7a2e9daed19a8fab2a0f33a33a904b0938b27bfffc01bbc458381f1 |
| finger_holder_left.stl | 373384 | d8793e2927e8d9a8033959468751d30cc5475be7367223d14412ab67b122b120 |
| finger_holder_right.stl | 372384 | 3b25d8a70b86cb09f888ae9b4087666fb64bb465c29913500dcbe2c37e346be6 |
| finger_left.stl | 97284 | f9453d03f6395da3da1f44402c609e813021596121925b15d72bf152de2abc40 |
| finger_right.stl | 97284 | 1636beb2d80b5ebb784ea377903508f76e2ef5eb1cd190a7452e84e3baceee3f |


## Origin

- CAD: Onshape documents a8a98201374f5cc9c6538790 ("UMI Gripper", whole assembly) and c8d06048fc2e425772f2749c ("Soft
  Finger"), exported as STL by the user on 2026-10-27/28.
- Original exports (not stored in the kit):
  - `vis_assembly.stl`: 3874984 bytes, sha256 9878ecc325e87264c844c8ef24cadf351c0b0b51acedee6c06ac1a384c087498
    (binary STL, metres, 77,498 triangles, 14 connected components, gripper shown open).
  - `soft_gripper_finger.stl`: 97284 bytes, sha256 2a019aec54578b1933d178ad6ed673ea1903557f16ea8fedf8d6e14e03968472
    (the soft finger alone in its part frame; used only to confirm the assembly copies have the same shape, < 1e-4 mm).
- Split: parts separated from `vis_assembly.stl` by vertex connectivity (vertices welded at 1e-4 mm), triangles copied
  unchanged, kept in assembly coordinates (m). finger_left / finger_holder_left = the assembly -x components (1944 /
  7466 triangles), finger_right / finger_holder_right = the +x components (1944 / 7446 triangles).
- Frames (kit_T_M, tip, tags, camera_face): derived from the geometry, see the `provenance` block in kit.yaml.
- Re-pin after any edit: `python -m franka_sim finger-kit pin DIR` rewrites this file; re-append this section.
