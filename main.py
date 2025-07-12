# main.py
import os
import argparse
from torch.backends import cudnn

from train import train
from evaluate import evaluate
from visualize import visualize


def make_directory(path, exist_ok=True):
    if not os.path.exists(path):
        log_path = os.path.join(path, 'log')
        fig_path = os.path.join(path, 'fig')
        img_path = os.path.join(path, 'image')
    
        os.makedirs(log_path, exist_ok=exist_ok)
        os.makedirs(fig_path, exist_ok=exist_ok)
        os.makedirs(img_path, exist_ok=exist_ok)
        
        print(f'Create path : {path}')
        
        
def main(args):
    cudnn.benchmark = True
    
    # Make Directory
    make_directory(args.save_path)

    # Train or Evaluate
    if args.mode == 'train':
        train(args)
        
    elif args.mode == 'eval' or args.mode == 'evaluate':
        evaluate(args)
        
    elif args.mode == 'vs' or args.mode == 'visualize':
        visualize(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--data_path', type=str, default='./SRdata')
    parser.add_argument('--save_path', type=str, default='./save/')
    parser.add_argument('--load_path', type=str, default='./save')
    parser.add_argument('--test_patient', type=str, default='L506')
    parser.add_argument('--result_fig', action='store_true')
    parser.add_argument('--num_imgs', type=int, default=1)
    
    parser.add_argument('--workframe', type=str, default='red_cnn')
    parser.add_argument('--version', type=float, default=1.0)

    args = parser.parse_args()
    args.save_path = os.path.join(args.save_path,'{}_ver{}'.format(args.workframe, args.version))
    
    main(args)