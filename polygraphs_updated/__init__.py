"""
PolyGraph module
"""
import os
import uuid
import datetime
import random as rnd
import collections
import json

import dgl
import torch
import numpy as np

from . import hyperparameters as hparams
from . import metadata
from . import monitors
from . import graphs
from . import ops
from . import logger
from . import timer
from .init import init as initialize_tensor 
from .graphs import create_subgraphs


# Removed (exporting PolyGraph to JPEG is deprecated for now)
# from . import visualisations as viz


log = logger.getlogger()

# Cache data directory for all results
_RESULTCACHE = os.getenv("POLYGRAPHS_CACHE") or "~/polygraphs-cache/results"


def _mkdir(directory="auto", attempts=10):
    """
    Creates unique directory to store simulation results.
    """
    # Unique exploration or simulation id
    uid = None
    if not directory:
        # Do nothing
        return uid, directory
    head, tail = os.path.split(directory)
    if tail == "auto":
        # If the parent is set, do not create subdirectory for today
        date = datetime.date.today().strftime("%Y-%m-%d") if not head else ""
        head = head or _RESULTCACHE
        for attempt in range(attempts):
            # Generate unique id string
            uid = uuid.uuid4().hex
            # Generate unique directory
            directory = os.path.join(os.path.expanduser(head), date, uid)
            # Likely
            if not os.path.isdir(directory):
                break
        # Unlikely error
        assert (
            attempt + 1 < attempts
        ), f"Failed to generate unique id after {attempts} attempts"
    else:
        # User-defined directory must not exist
        assert not os.path.isdir(directory), "Results directory already exists"
    # Create result directory, or raise an exception if it already exists
    os.makedirs(directory)
    return uid, directory


def _storeresult(params, result):
    """
    Helper function for storing simulation results
    """
    if params.simulation.results is None:
        return
    # Ensure destination directory exists
    assert os.path.isdir(params.simulation.results)
    # Export results
    result.store(params.simulation.results)


def _storeparams(params, explorables=None):
    """
    Helper function for storing configuration parameters
    """
    if params.simulation.results is None:
        return
    # Ensure destination directory exists
    assert os.path.isdir(params.simulation.results)
    # Export hyper-parameters
    params.toJSON(params.simulation.results, filename="configuration.json")
    # Export explorables
    if explorables:
        fname = os.path.join(params.simulation.results, "exploration.json")
        with open(fname, "w") as fstream:
            json.dump(explorables, fstream, default=lambda x: x.__dict__, indent=4)


def _storegraph(params, graph, prefix):
    """
    Helper function for storing simulated graph
    """
    if not params.simulation.results:
        return
    # Ensure destination directory exists
    assert os.path.isdir(params.simulation.results)
    # Export DGL graph in binary format
    fname = os.path.join(params.simulation.results, f"{prefix}.bin")
    dgl.save_graphs(fname, [graph])
    # Export DGL graph as JPEG
    #
    # Important note:
    #
    #    viz.draw is not well suited for drawing a graph.
    #    Remove for now.
    #
    # fname = os.path.join(params.simulation.results, f"{prefix}.jpg")
    # _, _ = viz.draw(graph, layout="circular", fname=fname)


def random(seed=0):
    """
    Set random number generator for PolyGraph simulations.
    """
    # Set PyTorch random number generator (RNG) for all devices (CPU and CUDA)
    torch.manual_seed(seed)
    # Set NumPy RNG
    np.random.seed(seed)
    # Set Python RNG
    rnd.seed(seed)
    # Set GDL RNG
    dgl.random.seed(seed)


def explore(params, explorables):
    """
    Explores multiple PolyGraph configurations.
    """
    # Get exploration options
    options = {var.name: var.values for var in explorables.values()}
    # Get all possible configurations
    configurations = hparams.PolyGraphHyperParameters.expand(params, options)
    # There must be at least two configurations
    assert len(configurations) > 1
    # Exploration results ought to be stored
    assert params.simulation.results
    # Create parent directory to store results
    _, params.simulation.results = _mkdir(params.simulation.results)
    # Store configuration parameters
    _storeparams(params, explorables=explorables)
    # Intermediate result collection
    collection = collections.deque()
    # Run all
    for config in configurations:
        # Store intermediate results?
        config.simulation.results = os.path.join(
            params.simulation.results, "explorations/auto"
        )
        # Set metadata columns
        meta = {key: config.getattr(var.name) for key, var in explorables.items()}
        # Metadata columns to string
        log.info(
            "Explore {} ({} simulations)".format(
                ", ".join([f"{k} = {v}" for k, v in meta.items()]),
                config.simulation.repeats,
            )
        )
        # Run experiment
        result = simulate(config, **meta)
        collection.append(result)

    # Merge simulation results
    results = metadata.merge(*collection)
    # Store simulation results
    _storeresult(params, results)
    return results

def calculate_steps_per_subgraph(num_subgraphs, total_steps):
    """
    Calculates the number of steps to run for each subgraph.

    Args:
        num_subgraphs (int): The number of subgraphs created based on the interval.
        total_steps (int): The total number of steps specified in the simulation configuration.

    Returns:
        int: The number of steps to run for each subgraph.
    """
    if num_subgraphs == 0:
        raise ValueError("Number of subgraphs cannot be zero.")
    # Calculate steps per subgraph, ensuring at least one step per subgraph
    steps_per_subgraph = max(total_steps // num_subgraphs, 1)
    return steps_per_subgraph

def initialize_beliefs(graph, params):
    """
    Initializes the node beliefs for the graph using the existing `init` function.
    
    Args:
        graph (DGLGraph): The graph for which beliefs need to be initialized.
        params (HyperParameters): The simulation hyperparameters.
    """
    num_nodes = graph.number_of_nodes()
    
    # Call the renamed init function to initialize beliefs based on params.init
    init_beliefs = initialize_tensor(size=(num_nodes,), params=params.init)  # Change to params.init
    
   
   # Assign the initialized beliefs to the graph's node data
    graph.ndata['beliefs'] = init_beliefs
   
    return init_beliefs

def transfer_node_states(subgraph, node_states):
    """
    Transfers node states (beliefs) to the subgraph before running the simulation.
    """
    if dgl.NID in subgraph.ndata:
        subgraph.ndata['beliefs'] = node_states[subgraph.ndata[dgl.NID]]
    else:
        print("Node IDs not found in subgraph.")


def store_node_states(subgraph):
    """
    Stores node states (beliefs) from the subgraph after running the simulation.
    """
    return subgraph.ndata['beliefs'].clone()  # Make a copy of the current beliefs to be used in the next interval


    
def print_node_states(graph, step_description):
    """
    Prints the node states (beliefs) for the first few nodes in the graph.
    """
    print(f"\nNode states at {step_description}:")
    node_count = min(5, graph.number_of_nodes())  # Print the first few nodes
    for node in range(node_count):
        belief = graph.ndata['beliefs'][node].item()  # Assuming beliefs is a 1D tensor
        print(f"Node {node}: Belief = {belief}")

@torch.no_grad()
def simulate(params, op=None, **meta):
    """
    Runs a PolyGraph simulation multiple times for both static and temporal networks.
    """
    assert isinstance(params, hparams.PolyGraphHyperParameters)
    
    if (params.op is None) == (op is None):
        if (params.op is None) and (op is None):
            raise ValueError("Operator not set")
        else:
            raise ValueError("Either params.op or op must be set, but not both")
    
    if op is None:
        op = ops.getbyname(params.op)
    else:
        params.op = op.__name__

    # Create result directory
    uid, params.simulation.results = _mkdir(params.simulation.results)
    _storeparams(params)
    results = metadata.PolyGraphSimulation(uid=uid, **meta)

    # Check if the network is temporal
    interval = getattr(params, 'interval', None) if hasattr(params, 'interval') else None
    adaptive = getattr(params, 'adaptive', False) if hasattr(params, 'adaptive') else False
    if interval:
        # Temporal graph (subgraph case)
        graph = graphs.create(params.network)
        
        # Check the edge attributes directly
        if not hasattr(graph, 'edata') or 'timestamp' not in graph.edata:
            raise ValueError("No timestamp edge attribute found in the DGL graph.")
        
        # Create subgraphs
        subgraphs = create_subgraphs(graph, interval)
        print(f"Number of subgraphs created: {len(subgraphs)}")  # Debugging print
        
        if len(subgraphs) == 0:
            raise ValueError("No subgraphs created.")

        steps_per_subgraph = calculate_steps_per_subgraph(len(subgraphs), params.simulation.steps)
        
        # Initialize beliefs for the graph before creating subgraphs
        node_states = initialize_beliefs(graph, params)
        
        # Temporal graph (subgraph) case
        for idx in range(params.simulation.repeats):
            log.debug("Simulation #{:04d} starts".format(idx + 1))
            for i, (interval_label, subgraph) in enumerate(subgraphs):
                print(f"Running simulation {i + 1}/{len(subgraphs)} for interval: {interval_label}, Repetition: {idx + 1}")
        
                # Move the subgraph to the device (e.g., CPU/GPU)
                subgraph = subgraph.to(device=params.device)
        
                # Transfer the node states (beliefs) to the subgraph
                transfer_node_states(subgraph, node_states)
        
                # Run the simulation on the subgraph
                model = op(subgraph, params)
                hooks = create_hooks(params, interval_label)
        
                result = simulate_(
                    subgraph,
                    model,
                    steps=steps_per_subgraph,
                    mistrust=params.mistrust,
                    lowerupper=params.lowerupper,
                    upperlower=params.upperlower,
                    hooks=hooks,
                )
        
                # Create a unique prefix for each subgraph
                prefix = f"{interval_label}"
                print(f"Storing subgraph with prefix: {prefix}")
        
                # Store the updated subgraph after running the simulation
                _storegraph(params, subgraph, prefix)
        
                # Add results to the results object
                results.add(*result)
                log_result(idx + 1, interval_label, steps_per_subgraph, result)
        
                # Save node states for the next iteration
                node_states = store_node_states(subgraph)

                print(f'Updated node states after simulation: {node_states}')
            
            # # Store final beliefs for this interval into the belief matrix
            # final_beliefs = subgraph.ndata['beliefs'].cpu().numpy()  # Get beliefs as numpy array
            # belief_matrix.append(final_beliefs)  # Append to the matrix
    elif adaptive:
        # Adaptive graph handling
        graph = graphs.create(params.network)
        node_states = initialize_beliefs(graph, params)
        graph.ndata['beliefs'] = node_states  # Initialize 'states' attribute in graph.ndata
        total_steps = params.simulation.steps

        # # Ensure graph_update_fn is callable
        # if graph_update_fn is not None and not callable(graph_update_fn):
        #     raise ValueError("The provided graph_update_fn is not callable.")

        # Create the adaptive operator with the provided adaptation rule
        adaptive_op = ops.AdaptivePolyGraphOp(graph, params)

        for step in range(total_steps):
            if step%100==0:
                print(f"Running adaptive simulation step {step + 1}/{total_steps}")
                print('average/std opinion', np.mean(graph.ndata["beliefs"].cpu().numpy()), np.quantile(graph.ndata["beliefs"].cpu().numpy(), .25), np.quantile(graph.ndata["beliefs"].cpu().numpy(), .75))
            adaptive_op.forward()  # Run the adaptive operator's forward method to update both beliefs and graph          
           
               
    else: 
        # Standard execution for static networks

        for idx in range(params.simulation.repeats):
            log.debug("Simulation #{:04d} starts".format(idx + 1))
            graph = graphs.create(params.network)
            graph = graph.to(device=params.device)
            model = op(graph, params)
            hooks = create_hooks(params, f"{(idx + 1):0{len(str(params.simulation.repeats))}d}")

            result = simulate_(
                graph,
                model,
                steps=params.simulation.steps,
                mistrust=params.mistrust,
                lowerupper=params.lowerupper,
                upperlower=params.upperlower,
                hooks=hooks,
            )
            # Store the updated graph after running the simulation
            prefix = f"{(idx + 1):0{len(str(params.simulation.repeats))}d}"
            _storegraph(params, graph, prefix)
            results.add(*result)
            log_result(idx + 1, "", params.simulation.steps, result)

    # Store final results
    _storeresult(params, results)
    return results

def create_hooks(params, label):
    """
    Create MonitorHook and SnapshotHook for the given label.
    """
    hooks = []
    if params.logging.enabled:
        hooks += [monitors.MonitorHook(interval=params.logging.interval)]
    if params.snapshots.enabled:
        hooks += [
            monitors.SnapshotHook(
                interval=params.snapshots.interval,
                messages=params.snapshots.messages,
                location=params.simulation.results,
                filename=f"{label}.hd5",
            )
        ]
    return hooks

def log_result(idx, interval_label, steps, result):
    """
    Log the results of a simulation.
    """
    steps_taken, duration, action, undefined, converged, polarized = result
    log.info(
        f"Sim #{idx:04d} Interval {interval_label}: "
        f"{steps_taken} steps "
        f"{duration:7.2f}s; "
        f"action: {action} "
        f"undefined: {undefined} "
        f"converged: {converged} "
        f"polarized: {polarized}"
    )


def simulate_(
    graph, model, steps=1, hooks=None, mistrust=0.0, lowerupper=0.5, upperlower=0.99
):
    """
    Runs a simulation either for a finite number of steps or until convergence.

    Returns:
        A 4-tuple that consists of (in order):
            a) number of simulation steps
            b) wall-clock time
            c) whether the network has converged or not
            d) whether the network is polarised or not
    """

    def cond(step):
        return step < steps if steps else True

    clock = timer.Timer()
    clock.start()
    step = 0
    terminated = None
    while cond(step):
        step += 1
        # Forward operation on the graph
        _ = model(graph)
        # Monitor progress
        if hooks:
            for hook in hooks:
                hook.mayberun(step, graph)
        # Check termination conditions:
        # - Are beliefs undefined (contain nan or inf)?
        # - Has the network converged?
        # - Is it polarised?
        terminated = (
            undefined(graph),
            converged(graph, upperlower=upperlower, lowerupper=lowerupper),
            polarized(
                graph, upperlower=upperlower, lowerupper=lowerupper, mistrust=mistrust
            ),
        )
        if any(terminated):
            break
    duration = clock.dt()
    if not terminated[0]:
        # Proper exit
        if hooks:
            for hook in hooks:
                hook.conclude(step, graph)
        # Which action did the network decide to take?
        act = consensus(graph, lowerupper=lowerupper)
    else:
        # Beliefs are undefined, and so is the action
        act = "?"

    # Are beliefs undefined (contain nan or inf)?
    # Has the network converged?
    # Is it polarised?
    # How many simulation steps were performed?
    # How long did the simulation take?
    return (
        step,
        duration,
        act,
    ) + terminated


def undefined(graph):
    """
    Returns `True` is graph beliefs contain undefined values (`nan` or `inf`).
    """
    belief = graph.ndata["beliefs"]
    result = torch.any(torch.isnan(belief)) or torch.any(torch.isinf(belief))
    return result.item()


def consensus(graph, lowerupper=0.99):
    """
    Returns action ('A', 'B', or '?') agreed by all agents in the network.
    """
    if converged(graph, lowerupper=lowerupper):
        belief = graph.ndata["beliefs"]
        return "B" if torch.all(torch.gt(belief, lowerupper)) else "A"
    return "?"

def converged(graph, upperlower=0.5, lowerupper=0.99):
    """
    Returns `True` if graph has converged.
    """
    tensor = graph.ndata["beliefs"]
    result = torch.all(torch.gt(tensor, lowerupper)) or torch.all(
        torch.le(tensor, upperlower)
    )
    return result.item()


def polarized(graph, mistrust=0.0, upperlower=0.5, lowerupper=0.99):
    """
    Returns `True` if graph is polarized.
    """
    # pylint: disable=invalid-name
    if not mistrust:
        return False
    tensor = graph.ndata["beliefs"]
    # All nodes have decided which action to take (e.g. A or B)
    c = torch.all(torch.gt(tensor, lowerupper) | torch.le(tensor, upperlower))
    # There is at least one strong believer
    # that action B is better
    b = torch.any(torch.gt(tensor, lowerupper))
    # There is at least one disbeliever
    a = torch.any(torch.le(tensor, upperlower))
    if a and b and c:
        delta = torch.min(tensor[torch.gt(tensor, lowerupper)]) - torch.max(
            tensor[torch.le(tensor, upperlower)]
        )
        return torch.ge(delta * mistrust, 1).item()
    return False
