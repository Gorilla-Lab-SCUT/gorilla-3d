python -m gorilla3d.preprocessing.s3dis.prepare_data_inst --data-root ./data --save-dir ./inputs --patch # --align # work for aligned version dataset
python -m gorilla3d.preprocessing.s3dis.prepare_data_inst_gttxt --data-dir ./inputs --save-dir ./labels 

# random sample
python -m gorilla3d.preprocessing.s3dis.downsample --data-dir ./inputs --ratio 0.25

# # voxelize down-sample
# CUDA_VISIBLE_DEVICES=0 python -m gorilla3d.preprocessing.s3dis.downsample --data-dir ./inputs --voxel-size 0.01
# # partition down-sample
# python -m gorilla3d.preprocessing.s3dis.partition --data-dir ./inputs

