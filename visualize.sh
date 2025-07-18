echo ==== Show Start ====

python main.py \
    --mode vs \
    --data_path "./../Datasets/CT_SR" \
    --save_path "save" \
    --load_path "/mnt/d/Research/CT_SR/save/edge_cnn_ver2.1/best.pt"\
    --num_imgs 1\
    --workframe edge_cnn \
    --version 2.1