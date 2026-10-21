"""
Filter HICO anno.mat and anno_bbox.mat to only include images that
actually exist in datasets/hico/images/train2024 and datasets/hico/images/test2024.

Usage: python3 filter_anno.py [--dry-run]
"""

import argparse
import os
import shutil

import numpy as np
import scipy.io as sio

HICO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets", "hico")
TRAIN_DIR = os.path.join(HICO_DIR, "images", "train2024")
TEST_DIR = os.path.join(HICO_DIR, "images", "test2024")


def filenames_from_list(list_arr):
    """list_train/list_test are (N, 1) cell arrays, each cell a 1-element str array."""
    return [str(list_arr[i, 0][0]) for i in range(list_arr.shape[0])]


def filter_anno_mat(path, keep_train, keep_test, dry_run):
    data = sio.loadmat(path)
    train_names = filenames_from_list(data["list_train"])
    test_names = filenames_from_list(data["list_test"])

    mask_train = np.array([n in keep_train for n in train_names])
    mask_test = np.array([n in keep_test for n in test_names])

    print(f"{os.path.basename(path)}:")
    print(f"  list_train: {mask_train.sum()}/{len(train_names)} kept")
    print(f"  list_test:  {mask_test.sum()}/{len(test_names)} kept")

    if dry_run:
        return

    data["list_train"] = data["list_train"][mask_train]
    data["list_test"] = data["list_test"][mask_test]
    data["anno_train"] = data["anno_train"][:, mask_train]
    data["anno_test"] = data["anno_test"][:, mask_test]
    sio.savemat(path, data)


def filter_bbox_mat(path, keep_train, keep_test, dry_run):
    data = sio.loadmat(path)
    bbox_train = data["bbox_train"]  # (1, N) struct array
    bbox_test = data["bbox_test"]

    train_names = [str(bbox_train[0, i]["filename"][0]) for i in range(bbox_train.shape[1])]
    test_names = [str(bbox_test[0, i]["filename"][0]) for i in range(bbox_test.shape[1])]

    mask_train = np.array([n in keep_train for n in train_names])
    mask_test = np.array([n in keep_test for n in test_names])

    print(f"{os.path.basename(path)}:")
    print(f"  bbox_train: {mask_train.sum()}/{len(train_names)} kept")
    print(f"  bbox_test:  {mask_test.sum()}/{len(test_names)} kept")

    if dry_run:
        return

    data["bbox_train"] = bbox_train[:, mask_train]
    data["bbox_test"] = bbox_test[:, mask_test]
    sio.savemat(path, data)


def main():
    parser = argparse.ArgumentParser(description="Filter HICO .mat annotations to existing images.")
    parser.add_argument("--dry-run", action="store_true", help="Only report counts, do not write files.")
    args = parser.parse_args()

    try:
        keep_train = set(os.listdir(TRAIN_DIR))
    except OSError as e:
        raise SystemExit(f"Error: cannot list train images dir ({TRAIN_DIR}): {e}") from e
    try:
        keep_test = set(os.listdir(TEST_DIR))
    except OSError as e:
        raise SystemExit(f"Error: cannot list test images dir ({TEST_DIR}): {e}") from e
    print(f"Images on disk: train={len(keep_train)}, test={len(keep_test)}")

    anno_path = os.path.join(HICO_DIR, "anno.mat")
    bbox_path = os.path.join(HICO_DIR, "anno_bbox.mat")

    if not args.dry_run:
        for path in (anno_path, bbox_path):
            backup = path + ".bak"
            if not os.path.exists(backup):
                shutil.copy2(path, backup)
                print(f"Backup created: {backup}")

    filter_anno_mat(anno_path, keep_train, keep_test, args.dry_run)
    filter_bbox_mat(bbox_path, keep_train, keep_test, args.dry_run)
    print("Done." if not args.dry_run else "Dry run complete, nothing written.")


if __name__ == "__main__":
    main()
