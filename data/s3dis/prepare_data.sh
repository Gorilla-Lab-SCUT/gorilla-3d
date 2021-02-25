python -m gorilla3d.preprocessing.s3dis.prepare_data_inst --data-root ./data --save-dir ./inputs
python -m gorilla3d.preprocessing.s3dis.prepare_data_inst_gttxt --data-dir ./inputs --save-dir ./labels 

# # voxelize down-sample
# CUDA_VISIBLE_DEVICES=0 python -m gorilla3d.preprocessing.s3dis.downsample --data-dir ./inputs
# # partition down-sample
# python -m gorilla3d.preprocessing.s3dis.partition --data-dir ./inputs

