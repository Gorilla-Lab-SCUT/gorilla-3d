python -m gorilla3d.preprocessing.s3dis.prepare_data_inst --data-root ./data --save-dir ./inputs
python -m gorilla3d.preprocessing.s3dis.prepare_data_inst_gttxt --data-dir ./inputs --save-dir ./labels 

# # voxelize down-sample
# python -m gorilla3d.preprocessing.s3dis.voxelize --data-dir ./inputs
