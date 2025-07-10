echo ==== Evaluate Start ====

python main.py \
    --mode eval \
    --data_path "./../Datasets/CT_SR" \
    --save_path "save" \
    --load_path "/mnt/d/Research/CT_SR/save/red_cnn_ver1.0/latest.pt"\
    --result_fig \
    --workframe red_cnn \
    --version 1.0