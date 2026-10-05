#!/usr/bin/env python3
"""Rename the side in an unzipped iPhUMI replay buffer, e.g. left -> right.

The single-arm training configs (task/umi.yaml) read `camera_right_*` and
`gripper_right_*` keys, so a dataset recorded with the gripper set to `L`
needs its keys renamed before training.
"""
import argparse

import zarr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('zarr_path', help='unzipped .zarr directory (not .zarr.zip)')
    parser.add_argument('--from-side', default='left')
    parser.add_argument('--to-side', default='right')
    args = parser.parse_args()

    root = zarr.open(args.zarr_path, mode='r+')
    for group_name in ('data', 'meta', 'labels'):
        if group_name not in root:
            continue
        group = root[group_name]
        for key in list(group.array_keys()):
            new_key = (key
                .replace(f'camera_{args.from_side}_', f'camera_{args.to_side}_')
                .replace(f'gripper_{args.from_side}_', f'gripper_{args.to_side}_'))
            if new_key == key:
                continue
            assert new_key not in group, f'{group_name}/{new_key} already exists'
            group.move(key, new_key)
            print(f'{group_name}/{key} -> {group_name}/{new_key}')


if __name__ == '__main__':
    main()
