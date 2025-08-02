echo ==== Evaluate Start ====

python main.py \
    --mode eval \
    --data_path "./../Datasets/CT_SR" \
    --save_path "save" \
    --load_path "save/fft_cnn_ver1.0/best.pt"\
    --result_fig \
    --workframe fft_cnn \
    --version 1.0