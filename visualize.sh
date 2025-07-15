echo ==== Show Start ====

python main.py \
    --mode vs \
    --data_path "./../Datasets/CT_SR" \
    --save_path "save" \
    --load_path "/mnt/d/Research/CT_SR/save/sr_cnn_ver1.0/latest.pt"\
    --num_imgs 1\
    --workframe edge_red_cnn \
    --version 1.0