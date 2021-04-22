# PointGroup in gorilla3d

# Quick Start
## Data Preparation
- Download origin [ScanNet](https://github.com/ScanNet/ScanNet) v2 data
```sh
dataset
└── scannetv2
    ├── meta_data(unnecessary, we have moved into our source code)
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

- Refer to [PointGroup](https://github.com/Jia-Research-Lab/PointGroup), we've modify the code, and it can generate input files `[scene_id]_inst_nostuff.pth` for instance segmentation directly, you don't need to split the origin data into `train/val/test`, the script refer to `gorilla3d/preprocessing/scannetv2/inst_seg`.
- And we package these command. You just running:
```sh
sh prepare_data.sh
```

## Compile

## Training
- single gpu
```sh
CUDA_VISIBLE_DEVICES=${gpu_id} python train.py --config config/default.yaml
python plain_train.py --config config/default.yaml # equal
```
- distributed
```sh
python plain_train.py --config config/default.yaml --num-gpus 2
```

## Testing
```sh
CUDA_VISIBLE_DEVICES=${gpu_id} python test.py --config config/default.yaml --pretrain ${checkpoint} (--semantic for only semantic evaluation)
```

