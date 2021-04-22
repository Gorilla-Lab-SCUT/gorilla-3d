# PointNet(classification) in gorilla3d

# Quick Start
## Data Preparation
Download alignment ModelNet [here](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip) and unzip in `data/modelnet40_normal_resampled/`.

## Training
- training using single gpu
```sh
python plain_train.py --config config/default.yaml
```
- distributed training
```sh
python plain_train.py --config config/default.yaml --num-gpus 2
```

## Testing
TODO

