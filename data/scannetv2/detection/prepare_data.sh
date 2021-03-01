# preprocess scannet dataset gt
python -m gorilla3d.preprocessing.scannetv2.segmentation.prepare_data_inst --data-split train
python -m gorilla3d.preprocessing.scannetv2.segmentation.prepare_data_inst --data-split val
python -m gorilla3d.preprocessing.scannetv2.segmentation.prepare_data_inst --data-split test
