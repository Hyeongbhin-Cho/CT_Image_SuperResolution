echo ==== Show Start ====

python main.py \
    --mode vs \
    --data_path "./../Datasets/CT_SR" \
    --save_path "save" \
    --load_path "/mnt/d/Research/CT_SR/save/red_cnn_ver1.0/latest.pt"\
    --num_imgs 1\
    --workframe red_cnn \
    --version 2.0