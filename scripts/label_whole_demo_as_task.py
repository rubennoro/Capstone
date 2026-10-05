#!/usr/bin/env python3
"""Give every unlabeled demo one task spanning the whole recording.

Demos recorded with task label type "None" produce zero tasks, and behavior
prompting (`*_dunetp`) samples by task. Run this after the `group` stage of
process_demos_iphone.py; it edits the grouped copies in tmp_demonstrations,
not the raw export.
"""
import argparse
import glob
import json
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--demonstrations-dir', required=True, help='e.g. .../demonstration_processing/tmp_demonstrations')
    parser.add_argument('--session-name', required=True)
    parser.add_argument('--task-name', required=True)
    args = parser.parse_args()

    n_labeled = 0
    for json_path in sorted(glob.glob(os.path.join(args.demonstrations_dir, '*', '*', '*.json'))):
        if os.path.basename(json_path) not in ('left.json', 'right.json', 'head.json'):
            continue
        with open(json_path) as f:
            demo = json.load(f)
        if demo.get('type') != 'demonstration' or demo.get('sessionName') != args.session_name:
            continue
        if demo.get('labelType', 'None') != 'None':
            continue

        rgb_times = [t for t in demo['rgbTimes'] if t]
        demo['labelType'] = 'Predefined'
        demo['taskNames'] = [args.task_name]
        demo['taskStartTimestamps'] = [rgb_times[0]]
        demo['taskEndTimestamps'] = [rgb_times[-1]]
        with open(json_path, 'w') as f:
            json.dump(demo, f)
        n_labeled += 1

    print(f'Labeled {n_labeled} demonstration files with task "{args.task_name}"')


if __name__ == '__main__':
    main()
