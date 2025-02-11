from argparse import ArgumentParser
import json
import math
import multiprocessing
import os
import pickle
import re
import warnings

from tqdm import tqdm


def find_sub_records(directory: str):
    pattern = re.compile(r"\d+-of-\d+-record\.json(l)?")

    sub_record_files = [f for f in os.listdir(directory) if pattern.match(f)]
    sorted_files = sorted(sub_record_files, key=lambda filename: int(filename.split("-of")[0]))
    return sorted_files


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--sub_record_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
    )
    args = parser.parse_args()

    if args.save_path is None:
        args.save_path = os.path.join(args.sub_record_dir, "record.jsonl")

    l_sub_records = find_sub_records(args.sub_record_dir)

    print(f"find {len(l_sub_records)} sub-records in {args.sub_record_dir}")
    print(str(l_sub_records) + "\n\n")

    complete_record = []
    for sub_record in l_sub_records:
        with open(os.path.join(args.sub_record_dir, sub_record)) as f:
            lines = f.readlines()
            for i, l in enumerate(lines):
                try:
                    l_item = json.loads(l)
                    complete_record.append(l_item)
                except:
                    if i == len(lines) - 1:
                        print(f"{sub_record} seems still writing, skip last incomplete record")
                    else:
                        warnings.warn(f"read line {i} in {sub_record} failed: {l}")

    d_id_item = {}
    repeated_count = 0
    for item in complete_record:
        if item["ori_id"] not in d_id_item:
            d_id_item[item["ori_id"]] = item
        else:
            assert item == d_id_item[item["ori_id"]]
            repeated_count += 1

    print("ori_count", len(complete_record), "repeated_count", repeated_count, "remianing_count", len(d_id_item))
    complete_record = list(d_id_item.values())
    # complete_record = list({item['ori_id']: item for item in complete_record}.values())

    complete_record = sorted(complete_record, key=lambda x: x["ori_id"])

    if args.save_path.endswith(".json"):
        with open(args.save_path, "w") as f:
            json.dump(complete_record, f, indent=1)
    elif args.save_path.endswith(".jsonl"):
        with open(args.save_path, "w") as f:
            for line in complete_record:
                f.write(json.dumps(line) + "\n")
    else:
        raise ValueError(f"unknown save file type {args.save_path}")
