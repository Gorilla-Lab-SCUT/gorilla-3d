# preprocess scannet dataset gt
python -m gorilla3d.preprocessing.scannetv2.inst_seg.prepare_data_inst --data-split train
python -m gorilla3d.preprocessing.scannetv2.inst_seg.prepare_data_inst --data-split val
python -m gorilla3d.preprocessing.scannetv2.inst_seg.prepare_data_inst --data-split test
# prepare validation dataset gt
python -m gorilla3d.preprocessing.scannetv2.inst_seg.prepare_data_inst_gttxt --data-split val
