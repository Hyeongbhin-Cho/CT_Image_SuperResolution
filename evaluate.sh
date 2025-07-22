echo ==== Evaluate Start ====

python main.py \
    --mode eval \
    --data_path "./../Datasets/CT_SR" \
    --save_path "save" \
    --load_path "save/edge_cnn_ver1.5/best.pt"\
    --result_fig \
    --workframe edge_cnn \
    --version 1.5