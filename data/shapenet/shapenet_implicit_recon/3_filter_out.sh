
nohup python -u 3_filter_out.py --class_name 03001627 --split train 2>&1 > 3_03001627_train.log &
nohup python -u 3_filter_out.py --class_name 03001627 --split test 2>&1 > 3_03001627_test.log &

nohup python -u 3_filter_out.py --class_name 02691156 --split train 2>&1 > 3_02691156_train.log &
nohup python -u 3_filter_out.py --class_name 02691156 --split test 2>&1 > 3_02691156_test.log &

nohup python -u 3_filter_out.py --class_name 04090263 --split train --minMB 1.0 2>&1 > 3_04090263_train.log &
nohup python -u 3_filter_out.py --class_name 04090263 --split test --minMB 1.0 2>&1 > 3_04090263_test.log &

nohup python -u 3_filter_out.py --class_name 04256520 --split train 2>&1 > 3_04256520_train.log &
nohup python -u 3_filter_out.py --class_name 04256520 --split test 2>&1 > 3_04256520_test.log &

nohup python -u 3_filter_out.py --class_name 04379243 --split train 2>&1 > 3_04379243_train.log &
nohup python -u 3_filter_out.py --class_name 04379243 --split test 2>&1 > 3_04379243_test.log &

# --------------

# new
# 03797390  mug杯子
# 03211117  display屏幕电视
# 02933112  cabinet家具柜子橱柜
# 02876657  bottle瓶子
# 02871439  bookshelf书架
# 02818832  bed床
# 02801938  basket篮子柜子

nohup python -u 3_filter_out.py --class_name 03797390 --split train 2>&1 > 3_03797390_train.log &
nohup python -u 3_filter_out.py --class_name 03797390 --split test 2>&1 > 3_03797390_test.log &

nohup python -u 3_filter_out.py --class_name 03211117 --split train 2>&1 > 3_03211117_train.log &
nohup python -u 3_filter_out.py --class_name 03211117 --split test 2>&1 > 3_03211117_test.log &

nohup python -u 3_filter_out.py --class_name 02933112 --split train 2>&1 > 3_02933112_train.log &
nohup python -u 3_filter_out.py --class_name 02933112 --split test 2>&1 > 3_02933112_test.log &

nohup python -u 3_filter_out.py --class_name 02876657 --split train 2>&1 > 3_02876657_train.log &
nohup python -u 3_filter_out.py --class_name 02876657 --split test 2>&1 > 3_02876657_test.log &

nohup python -u 3_filter_out.py --class_name 02871439 --split train 2>&1 > 3_02871439_train.log &
nohup python -u 3_filter_out.py --class_name 02871439 --split test 2>&1 > 3_02871439_test.log &

nohup python -u 3_filter_out.py --class_name 02818832 --split train 2>&1 > 3_02818832_train.log &
nohup python -u 3_filter_out.py --class_name 02818832 --split test 2>&1 > 3_02818832_test.log &

nohup python -u 3_filter_out.py --class_name 02801938 --split train 2>&1 > 3_02801938_train.log &
nohup python -u 3_filter_out.py --class_name 02801938 --split test 2>&1 > 3_02801938_test.log &


