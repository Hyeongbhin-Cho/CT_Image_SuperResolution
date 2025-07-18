echo ==== Train Start ====

python main.py \
    --mode train \
    --data_path "./../Datasets/CT_SR" \
    --save_path "save" \
    --result_fig \
    --workframe edge_cnn \
    --version 2.1