#!/usr/bin/env python3
"""Drop optimizer state from a training checkpoint so it is small enough to share.

Deployment (behavior_prompting/deployment/detached_inference.py) already loads
checkpoints with exclude_keys=["optimizer"], so the slimmed file still deploys.
"""
import argparse

import dill
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_ckpt')
    parser.add_argument('output_ckpt')
    args = parser.parse_args()

    payload = torch.load(open(args.input_ckpt, 'rb'), map_location='cpu', pickle_module=dill)
    dropped = [k for k in ('optimizer', 'lr_scheduler') if payload['state_dicts'].pop(k, None) is not None]
    torch.save(payload, open(args.output_ckpt, 'wb'), pickle_module=dill)
    print(f'Dropped {dropped}; kept {list(payload["state_dicts"])}')


if __name__ == '__main__':
    main()
