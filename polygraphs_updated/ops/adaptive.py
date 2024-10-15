import torch
import networkx as nx
import dgl
import numpy as np

from . import core
from . import math
from .. import init


class AdaptivePolyGraphOp(torch.nn.Module):
    def __init__(self, graph, params):
        super().__init__()

        # Set device for experimentation
        self._device = params.device

        # The shape of all node attributes
        size = (graph.num_nodes(),)

        # Initialize activities
        activities_params = getattr(params, 'activities', None)
        if activities_params is not None:
            self.activities = self._initialize_activities(activities_params, size)
        else:
            raise ValueError("Activities attribute not found in parameters.")

        # Node beliefs and other initializations
        if params.init.type == "uniform":
            graph.ndata["beliefs"] = torch.FloatTensor(size[0]).uniform_(params.init.lower, params.init.upper).to(self._device)
        elif params.init.type == "gaussian":
            mean = getattr(params.init, 'mean', 0.0)
            std = getattr(params.init, 'std', 1.0)
            graph.ndata["beliefs"] = torch.FloatTensor(size[0]).normal_(mean, std).to(self._device)
        else:
            graph.ndata["beliefs"] = init.init(size, params.init).to(device=self._device)


        self.graph = graph
        self.params = params

        # Access adaptation parameters if they exist
        adaptation_params = getattr(params, 'adaptation_params', None)
        gamma = getattr(adaptation_params, 'gamma', 2.0)
        m = getattr(adaptation_params, 'm', 5)

        # Choose the adaptation rule
        if hasattr(params, "adaptation_rule"):
            if params.adaptation_rule == "homophily_based_AD":
                self.adaptation_rule = lambda g: _homophily_based_AD(
                    g,
                    gamma=gamma,
                    m=m,
                    activities=self.activities
                )
            elif params.adaptation_rule == "custom":
                self.adaptation_rule = custom_adaptation_rule_homophily
            else:
                self.adaptation_rule = self._default_adaptation_rule
        else:
            self.adaptation_rule = self._default_adaptation_rule

        # Choose belief update rule based on params
        if params.belief_update_rule == "default":
            self.belief_update_rule = self._default_belief_update_rule
        elif params.belief_update_rule == "sigma_AD":
            self.belief_update_rule = BeliefUpdateRuleSigmaAD(alpha=params.alpha, K=params.K, dt=params.dt)
        else:
            raise ValueError(f"Unknown belief update rule: {params.belief_update_rule}")

    def _initialize_activities(self, activities_params, size):
        # Access 'type' in a more robust manner, to work with dictionary
        activity_type = activities_params.get('type', None)

        if activity_type == "gaussian":
            mean = activities_params.get('params', {}).get('mean', 0.5)
            std = activities_params.get('params', {}).get('std', 0.1)
            return np.clip(np.random.normal(mean, std, size), 0, 1)
        elif activity_type == "exponential":
            scale = activities_params.get('params', {}).get('scale', 1.0)
            return np.clip(np.random.exponential(scale, size), 0, 1)
        elif activity_type == "powerlaw":
            powerlaw_alpha = activities_params.get('params', {}).get('powerlaw_alpha', 2.5)
            return np.clip(np.random.pareto(powerlaw_alpha, size) + 1, 0, 1)
        else:
            raise ValueError(f"Unknown activity initialization type: {activity_type}")


    def forward(self):
        """
        Executes a step in the simulation, adjusting the network structure
        based on the states of the nodes and updating the beliefs.
        """
        # Update network structure based on node states
        self._adapt_network()
        
        # Update beliefs based on network structure
        self.belief_update_rule(self.graph)
        # print('average/std opinion', np.mean(self.graph.ndata["beliefs"].cpu().numpy()), np.std(self.graph.ndata["beliefs"].cpu().numpy()))

        # print('Forward step completed')

    def _adapt_network(self):
        """
        Adapt the network structure by adding or removing edges
        based on the node states.
        """
        # Apply the adaptation rule directly on the DGL graph
        self.adaptation_rule(self.graph)
        

    def _default_belief_update_rule(self, graph):
        """
        Default rule: update beliefs based on the average state of neighbors.
        """
        for node in range(graph.num_nodes()):
            neighbors = graph.successors(node)
            if len(neighbors) > 0:
                neighbor_states = graph.ndata["beliefs"][neighbors]
                graph.ndata["beliefs"][node] = torch.mean(neighbor_states.float()).to(self._device)

class BeliefUpdateRuleSigmaAD(torch.nn.Module):
    def __init__(self, alpha, K, dt):
        super(BeliefUpdateRuleSigmaAD, self).__init__()
        self.alpha = alpha
        self.K = K
        self.dt = dt

    def f_activity(self, sigma, x):
        return sigma - x

    def rkc_activity(self, f, dt, sigma, x):
        k1 = dt * f(sigma, x)
        k2 = dt * f(sigma, x + k1 / 2.)
        k3 = dt * f(sigma, x + k2 / 2.)
        k4 = dt * f(sigma, x + k3)
        return k1, k2, k3, k4

    def integration_step(self, f, dt, sigma, x):
        k1, k2, k3, k4 = self.rkc_activity(f, dt, sigma, x)
        return x + (1. / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    def forward(self, graph):
        """
        Custom belief update rule inspired by the sigma_inputs_AD function and equation 1 of the paper,
        adjusted for one-dimensional opinions.
        """
        adj_matrix = graph.adjacency_matrix().to_dense()
        beliefs = graph.ndata["beliefs"]

        Ax = torch.matmul(adj_matrix, beliefs)
        tanh_alpha_x = torch.tanh(self.alpha * Ax)

        new_beliefs = self.integration_step(self.f_activity, self.dt, self.K * tanh_alpha_x, beliefs)
        graph.ndata["beliefs"] = new_beliefs


def _homophily_based_AD(graph, gamma=2.0, m=10, activities=None, epsilon=1e-2):
    """
    Custom adaptation rule inspired by Starnini and Baumann's work on homophily and polarization.
    This rule aims to adapt the network structure by adding/removing edges based on homophily dynamics.

    Parameters:
        graph (dgl.DGLGraph): The DGL graph with node data attributes.
        gamma (float): Homophily exponent controlling the probability of connections based on opinion similarity.
        m (int): Number of edges to be created for each active node.
        activities (np.array): Array specifying the activity levels of each node.
        epsilon (float): Small value added to distances to prevent division by zero.
    """
    # print('Editing graph based on homophily')

    # Extract the beliefs (opinions) of each node from graph.ndata
    x = graph.ndata['beliefs'].cpu().numpy()

    # Removing all existing edges to reset the graph
    src, dst = graph.edges()
    graph.remove_edges(torch.arange(src.size(0)))

    # Number of nodes in the graph
    N = graph.num_nodes()

    # Function to determine which nodes are active
    def whos_active(activities):
        return np.where(np.random.uniform(0, 1, size=N) < activities)[0]

    # Function to sample targets based on homophily (1D opinions)
    def homophily_sampling_1d(i, x, gamma_hom, m, N, epsilon):
        distances_i = np.abs(x[i] - x) + epsilon
        distances_i[i] = 1  # Prevent division by zero for self

        p = (distances_i ** -gamma_hom)
        p[i] = 0  # No self-coupling

        p_sum = np.sum(p)
        if p_sum == 0:
            p[:] = 1 / (N - 1)  # If all probabilities are zero, use uniform distribution
        else:
            p /= p_sum

        return np.random.choice(range(N), m, replace=False, p=p)

    # Determine active nodes based on activities
    active_nodes = whos_active(activities)

    # Add edges based on homophily sampling
    for ai in active_nodes:
        targets_i = homophily_sampling_1d(ai, x, gamma, m, N, epsilon)
        for ti in targets_i:
            graph.add_edges(ai, ti)
            graph.add_edges(ti, ai)  # Add bidirectional edges
