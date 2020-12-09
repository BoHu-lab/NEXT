import numpy as np
from environment import MazeEnv
from model import Model
from algorithm import NEXT_plan, RRTS_plan
from utils import set_random_seed, load_model, plot_tree

set_random_seed(1234)
cuda = False

dim = 3
UCB_type = 'kde'
environment = MazeEnv(dim = dim)
model = Model(cuda = cuda, dim = dim)
model_file = 'trained_models/NEXT_%dd.pt' % dim
load_model(model.net, model_file, cuda)

# Sample a problem from the environment
pb_idx = 2101 # 0 - 2999
pb = environment.init_new_problem(pb_idx)
model.set_problem(pb)
search_tree, done = NEXT_plan(
    env = environment,
    model = model,
    T = 500,
    g_explore_eps = 0.1,
    stop_when_success = True,
    UCB_type = UCB_type
)
plot_tree(
    states = search_tree.states,
    parents = search_tree.parents,
    problem = environment.get_problem()
)

