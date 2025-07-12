# visualize.py

from networks import get_model
from solver import RedCNNSolver
from utils import get_loader, get_transforms

import yaml

def visualize(args: dict):
    with open(f'config/{args.workframe}.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # args + yaml
    args_dict = vars(args)  # Namespace → dict
    overlap = set(args_dict).intersection(config)
    if overlap:
        print(f"[Warning] These keys from args will overwrite config: {overlap}")

    vs_config = {**config, **args_dict}
    
    # Data loader
    vs_loader_config = vs_config["visualize"]["data_loader"]
    vs_loader = get_loader(mode='eval',
                           data_path=vs_config['data_path'],
                           transform=get_transforms(vs_loader_config.get('transform', {})),
                           batch_size=vs_loader_config.get('batch_size', 1),
                           num_workers=vs_loader_config.get('num_workers', 0),
                           shuffle=vs_loader_config.get('shuffle', False))

    # Build model
    ModelClass = get_model(vs_config['network']['type'])
    model = ModelClass()

    # Solver
    solver = RedCNNSolver(config=vs_config, model=model, vs_loader=vs_loader)
    solver.visualize()