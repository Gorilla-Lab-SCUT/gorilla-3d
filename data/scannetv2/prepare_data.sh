# preprocess scannet dataset gt
python -m gorilla3d.preprocessing.scannetv2.inst_seg.prepare_data_inst --data_split train
python -m gorilla3d.preprocessing.scannetv2.inst_seg.prepare_data_inst --data_split val
python -m gorilla3d.preprocessing.scannetv2.inst_seg.prepare_data_inst --data_split test
# prepare validation dataset gt
python -m gorilla3d.preprocessing.scannetv2.inst_seg.prepare_data_inst_gttxt --data_split val
