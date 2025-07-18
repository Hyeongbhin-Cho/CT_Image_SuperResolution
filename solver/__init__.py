# solver/__init__.py
from .base_solver import BaseSolver
from .red_cnn_solver import RedCNNSolver 
from .sr_cnn_solver import SRCNNSolver
from .edge_cnn_solver import EdgeCNNSolver

def get_solver(name: str) -> type[BaseSolver]:
    solvers = {
        'red_cnn': RedCNNSolver,
        'sr_cnn': SRCNNSolver,
        'edge_cnn': EdgeCNNSolver
    }

    name = name.lower()
    if name not in solvers:
        raise ValueError(f"Invalid solver name '{name}'. Available: {list(solvers.keys())}")

    return solvers[name]