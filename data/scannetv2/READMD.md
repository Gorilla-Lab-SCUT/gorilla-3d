# Prepare ScanNet Data
- Download origin [ScanNet](https://github.com/ScanNet/ScanNet) v2 data
```sh
dataset
└── scannetv2
    ├── meta_data
    │   ├── scannetv2_train.txt
    │   ├── scannetv2_val.txt
    │   ├── scannetv2_test.txt
    │   └── scannetv2-labels.combined.tsv
    ├── scans
    │   ├── ...
    │   ├── [scene_id]
    |   |    └── [scene_id]_vh_clean_2.ply & [scene_id]_vh_clean_2.labels.ply & [scene_id]_vh_clean_2.0.010000.segs.json & [scene_id].aggregation.json
    |   └── ...
    └── scans_test
        ├── ...
        ├── [scene_id]
        |    └── [scene_id]_vh_clean_2.ply & [scene_id].txt
        └── ...
```

- We've modify the code, and it can generate input files `[scene_id]_inst_nostuff.pth` for instance segmentation directly, you don't need to split the origin data into `train/val/test`.
```sh
python prepare_data_inst.py --data_split train
python prepare_data_inst.py --data_split val
python prepare_data_inst.py --data_split test
```
- Prepare the `.txt` instance ground-truth files as the following.
```sh
python prepare_data_inst_gttxt.py
```

