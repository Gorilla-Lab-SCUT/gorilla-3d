# Prepare S3DIS Data

- Download [S3DIS Dataset](http://buildingparser.stanford.edu/dataset.html) and extract `Stanford3dDataset_v1.2_Aligned_Version.zip` or `Stanford3dDataset_v1.2.zip`.
```sh
dataset
└── S3DIS
    ├── Area_1 (area_id)
    │   ├── conferenceRoom_1 (room_name)
    │   │   ├── conferenceRoom_1.txt
    │   │   └── Annotations
    │   │       ├── chair_1.txt (instance)
    │   │       └── ...
    |   └── ...
    |       └── ...
    ├── ...
    |   └── ...
    └── Area_6
        └── ...
```

- Refer to [superponit_graph](https://github.com/loicland/superpoint_graph), we can generate input files `{area_id}_{room_name}.pth` in `inputs` dir for instance segmentation directly.

- And we package these command. You just running:
```sh
sh prepare_data.sh
```
