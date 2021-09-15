# Copyright (c) Gorilla-Lab. All rights reserved.
import os
import random
import argparse

random.seed(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--class_name",
                        type=str,
                        nargs="+",
                        default=["03001627"],
                        help="Categories to process")
    parser.add_argument("--src_dataset_dir",
                        type=str,
                        help="Path to the unzipped `ShapeNetCore.v1` folder")
    parser.add_argument("--split_dir",
                        type=str,
                        help="Path to the folder to save split files")
    parser.add_argument("--train_val_test_ratio",
                        type=float,
                        nargs=3,
                        default=[0.8, 0.1, 0.1],
                        help="Ratio to split")
    args = parser.parse_args()

    if not os.path.exists(args.split_dir):
        os.makedirs(args.split_dir)

    assert sum(args.train_val_test_ratio) == 1

    train_ratio = args.train_val_test_ratio[0]
    val_ratio = args.train_val_test_ratio[1]

    for class_name in args.class_name:
        assert os.path.exists(os.path.join(args.src_dataset_dir, class_name))
        obj_list = os.listdir(os.path.join(args.src_dataset_dir, class_name))
        obj_list = [
            s for s in obj_list
            if os.path.isdir(os.path.join(args.src_dataset_dir, class_name, s))
        ]
        random.shuffle(obj_list)

        train_num = int(train_ratio * len(obj_list))
        val_num = int(val_ratio * len(obj_list))

        write_dict = dict(train=obj_list[:train_num],
                          val=obj_list[train_num:(train_num + val_num)],
                          test=obj_list[(train_num + val_num):])

        for sp in write_dict.keys():
            to_save_path = os.path.join(args.split_dir,
                                        f"{class_name}_{sp}.lst")
            with open(to_save_path, "w") as f:
                for line in write_dict[sp]:
                    f.write(f"{line}\n")
            print(f"save split one file to {to_save_path}")

    print("All done.")
