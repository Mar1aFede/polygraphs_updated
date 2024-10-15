"""
Graphs
"""
import sys
import inspect
import math
import networkx as nx
import dgl
import numpy as np
import os
import torch
import pandas as pd
import copy

from .hyperparameters import HyperParameters
from . import datasets


def _isconnected(graph):
    return nx.is_strongly_connected(dgl.to_networkx(graph))


def _buckleup(graph, exist_ok=False):
    """
    Adds self loops to given graph.
    """
    if not exist_ok:
        # The number of edges in the given graph
        count = len(graph.edges())
        # Remove all self-loops in the graph, if present
        graph = dgl.remove_self_loop(graph)
        # Assert |E'| = |E|
        assert len(graph.edges()) == count
    # Add self-loops for each node in the graph and return a new graph
    return dgl.transforms.add_self_loop(graph)


def sample_(selfloop=True):
    """
    Returns a sample 6-node graph.
    """
    src = np.array([0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 5, 5, 5, 5])
    dst = np.array([2, 5, 2, 3, 4, 5, 0, 1, 3, 5, 1, 2, 4, 5, 1, 3, 0, 1, 2, 3])

    graph = dgl.graph((src, dst))
    # Try adding self-loops
    if selfloop:
        graph = _buckleup(graph)
    return graph


def sample(params):
    """
    Returns a sample 6-node graph from hyper-parameters.
    """
    # Update network size
    params.size = 6
    return sample_(selfloop=params.selfloop)


def wheel_(size, selfloop=True):
    """
    Returns an undirected wheel graph.
    """
    # Check network size
    assert size > 1
    # Get graph from networkx
    graph = dgl.from_networkx(nx.wheel_graph(size))
    # Try adding self-loops
    if selfloop:
        graph = _buckleup(graph)
    return graph


def wheel(params):
    """
    Returns an undirected wheel graph from hyper-parameters.
    """
    return wheel_(params.size, selfloop=params.selfloop)


def cycle_(size, directed=False, selfloop=True):
    """
    Returns a cycle graph.
    """
    # Check network size
    assert size > 1
    # Get networkx constructor
    constructor = nx.DiGraph if directed else nx.Graph
    # Get graph from networkx
    graph = dgl.from_networkx(nx.cycle_graph(size, create_using=constructor))
    # Try adding self-loops
    if selfloop:
        graph = _buckleup(graph)
    return graph


def cycle(params):
    """
    Returns a cycle graph from hyper-parameters.
    """
    return cycle_(params.size, directed=params.directed, selfloop=params.selfloop)


def star_(size, selfloop=True):
    """
    Returns an undirected star graph.
    """
    # Check network size
    assert size > 1
    # Get graph from networkx. The graph has n + 1 nodes for integer n, so substract 1
    graph = dgl.from_networkx(nx.star_graph(size - 1))
    # Try adding self-loops
    if selfloop:
        graph = _buckleup(graph)
    return graph


def star(params):
    """
    Returns an undirected star graph.
    """
    return star_(params.size, selfloop=params.selfloop)


def line_(size, directed=False, selfloop=True):
    """
    Return a line graph.
    """
    # Check network size
    assert size > 1
    # Get networkx constructor
    constructor = nx.DiGraph if directed else nx.Graph
    # Get graph from networkx
    graph = dgl.from_networkx(nx.path_graph(size, create_using=constructor))
    # Try adding self-loops
    if selfloop:
        graph = _buckleup(graph)
    return graph


def line(params):
    """
    Return a line graph from hyper-parameters.
    """
    return line_(params.size, directed=params.directed, selfloop=params.selfloop)


def grid_(size, selfloop=True):
    """
    Returns a 2-D square grid graph.
    """
    # Check network size
    assert size > 1
    # Check network size is a perfect square (approximate solution)
    assert size == math.pow(int(math.sqrt(size) + 0.5), 2)
    rows = columns = int(math.sqrt(size))
    # Get graph from networkx
    graph = dgl.from_networkx(nx.grid_2d_graph(rows, columns))
    # Try adding self-loops
    if selfloop:
        graph = _buckleup(graph)
    return graph


def grid(params):
    """
    Returns a 2-D square grid graph from hyper-parameters.
    """
    return grid_(params.size, selfloop=params.selfloop)


def random_(size, probability, tries=100, seed=None, directed=False, selfloop=True):
    """
    Returns an Erdos-Renyi graph.
    """
    # Check network size
    assert size > 1
    # If seed is not set, use NumPy's global RNG
    if not seed:
        seed = np.random
    attempt = 0
    success = False
    while True:
        attempt += 1
        # Get graph from networkx
        graph = dgl.from_networkx(
            nx.erdos_renyi_graph(size, probability, seed=seed, directed=directed)
        )
        if _isconnected(graph):
            # Connected graph found; exit loop
            success = True
            break
        else:
            # Probability p should be greater than ((1 + e)ln(size))/size
            _p = math.log(size) / size
            if probability > _p and attempt < tries:
                # Probability is set correctly; permit failed attempts
                continue
            else:
                # Most likely all attempts will fail; exit loop
                break
    if not success:
        msg = f"Graph G({size, probability}) is disconnected. Try p > {_p}"
        raise Exception(msg)
    # Try adding self-loops
    if selfloop:
        graph = _buckleup(graph)
    return graph


def random(params):
    """
    Returns an Erdos-Renyi graph from hyper-parameters
    """
    return random_(
        params.size,
        params.random.probability,
        tries=params.random.tries,
        seed=params.random.seed,
        directed=params.directed,
        selfloop=params.selfloop,
    )


def complete_(size, selfloop=True):
    """
    Returns an undirected fully-connected graph.
    """
    # Check network size
    assert size > 1
    # Get graph from networkx
    graph = dgl.from_networkx(nx.complete_graph(size))
    # Try adding self-loops
    if selfloop:
        graph = _buckleup(graph)
    return graph


def complete(params):
    """
    Returns an undirected fully-connected graph from hyper-parameters.
    """
    return complete_(params.size, selfloop=params.selfloop)


def karate_(selfloop=True):
    """
    Returns Zachary's Karate club social network.
    """
    # Get graph from networkx
    graph = dgl.from_networkx(nx.karate_club_graph())
    # Try adding self-loops
    if selfloop:
        graph = _buckleup(graph)
    return graph


def karate(params):
    """
    Returns Zachary's Karate club social network from hyper-parameters.
    """
    # Update network size
    params.size = 34
    return karate_(selfloop=params.selfloop)


def wattsstrogatz_(
    size, knn, probability, tries=100, seed=None, selfloop=True
):  # pylint: disable=too-many-arguments
    """
    Returns a connected Watts–Strogatz small-world graph.
    """
    # Check network size
    assert size > 1
    # Check neighbourhood size
    assert knn > 1
    # If seed is not set, use NumPy's global RNG
    if not seed:
        seed = np.random
    # Get graph from networkx
    graph = dgl.from_networkx(
        nx.connected_watts_strogatz_graph(
            size, knn, probability, tries=tries, seed=seed
        )
    )
    # Try adding self-loops
    if selfloop:
        graph = _buckleup(graph)
    return graph


def wattsstrogatz(params):
    """
    Returns a connected Watts–Strogatz small-world graph from hyper-parameters.
    """
    return wattsstrogatz_(
        params.size,
        params.wattsstrogatz.knn,
        params.wattsstrogatz.probability,
        tries=params.wattsstrogatz.tries,
        seed=params.wattsstrogatz.seed,
        selfloop=params.selfloop,
    )


def barabasialbert_(size, attachments, seed=None, selfloop=True):
    """
    Returns a random graph according to the Barabasi–Albert preferential attachment model.
    """
    # Check network size
    assert size > 1
    # Check neighbourhood size
    assert attachments > 0
    # If seed is not set, use NumPy's global RNG
    if not seed:
        seed = np.random
    # Get graph from networkx
    graph = dgl.from_networkx(nx.barabasi_albert_graph(size, attachments, seed=seed))
    # Try adding self-loops
    if selfloop:
        graph = _buckleup(graph)
    return graph


def barabasialbert(params):
    """
    Returns a random graph according to the Barabasi–Albert
    preferential attachment model from hyper-parameters.
    """
    return barabasialbert_(
        params.size,
        params.barabasialbert.attachments,
        seed=params.barabasialbert.seed,
        selfloop=params.selfloop,
    )


def snap(params):
    """
    Returns a SNAP dataset, identified by `params.snap.name`.
    """
    from .datasets import snap as snp
    graph = snp.getbyname(params.snap.name).read()
    # Update network size
    params.size = graph.num_nodes()
    return graph


def ogb(params):
    """
    Returns an OGB dataset, identified by `params.ogb.name`.
    Current only one dataset is supported, ogbl-collab.
    """
    assert params.ogb.name and isinstance(params.ogb.name, str)
    assert params.ogb.name.lower == "collab"
    dataset = datasets.ogb.Collab()
    graph = dataset.read()
    # Update network size
    params.size = graph.num_nodes()
    return graph


def francisbacon(params):
    """
    Returns a Six Degrees of Francis Bacon Network.
    The GML file of the graph must be generated manually because the
    dataset is dynamic. See the sixdegreesoffrancisbacon notebook in
    the scripts folder.
    """
    from .datasets import francisbacon
    dataset = francisbacon.FrancisBacon()
    graph = dataset.read()
    # Try adding self-loops
    if params.selfloop:
        graph = _buckleup(graph)
    params.size = graph.num_nodes()
    return graph


def gml(params):
    """
    Loads a custom GML graph from a path identified by `params.gml.path`.
    A custom name can be given for the graph using `params.gml.name`
    Directed graphs should be specified by setting `params.gml.directed` to True
    """
    assert params.gml.name, "gml.name GML network name not specified"
    assert params.gml.path, "gml.path GML file not specified"
    # Resolve GML file
    gml_file = os.path.abspath(os.path.expanduser(params.gml.path))
    assert os.path.isfile(gml_file), "GML file not found"

    # Load graph from GML file, converting node IDs to integers
    G = nx.read_gml(gml_file, destringizer=int)

        # Load graph using edge list so that we preserve node IDs and attributes
    edges = []
    for u, v, data in G.edges(data=True):  # Now also get the edge attributes
        # Ensure that the first two elements of the edge are integers (node IDs)
        if not (isinstance(u, int) and isinstance(v, int)):
            raise ValueError("GML File: Node IDs should be specified as integers")
        
        # Append edge with attributes (including timestamp if available)
        edge_data = data.get('timestamp', None)  # Extract timestamp
        edges.append((u, v, edge_data))  # Keep (source, destination, timestamp)
    
    # Create DGL graph from edges while preserving edge attributes
    src, dst, timestamps = zip(*edges) if edges else ([], [], [])  # Unzip into source, destination, and timestamps
    graph = dgl.graph((src, dst))  # Create the DGL graph
    # Handle missing timestamps (for static graphs)
    if any(t is not None for t in timestamps):  # If there are any timestamps
        graph.edata['timestamp'] = torch.tensor(
            [t if t is not None else 0 for t in timestamps], dtype=torch.int64)  # Replace None with 0 or a default
        print("Timestamps added to the graph.")
    else:
        print("No timestamps found, running as a static graph without temporal information.")

    
    # Convert to a bi-directed DGL graph for undirected graphs
    if not params.gml.directed:
        graph = dgl.to_bidirected(graph)
    
    return graph

def create(params):
    """
    Returns a DGL graph of given type and size.
    """
    assert isinstance(params, HyperParameters)

    # Create friendly dictionary from list of (name, function) tuples
    members = dict(inspect.getmembers(sys.modules[__name__], inspect.isfunction))
    constructor = members.get(params.kind)

    if constructor is None:
        raise Exception(f"Invalid graph type: {params.kind}")

    # Construct the graph
    graph = constructor(params=params)

    # Debugging: Check the type of the created graph
    if isinstance(graph, nx.DiGraph):
        pass
    elif isinstance(graph, dgl.DGLGraph):
        pass
    else:
        raise ValueError("The created graph is not a recognized graph type.")

    # Initialize dgl_graph to None
    dgl_graph = None

    # If the graph is a NetworkX graph, convert to DGL graph with edge attributes
    if isinstance(graph, nx.DiGraph):
        # Check for timestamp in the original NetworkX graph
        edges_with_timestamps = [(u, v, d) for u, v, d in graph.edges(data=True) if 'timestamp' in d]
        
        print(f"Number of edges with timestamps in the NetworkX graph: {len(edges_with_timestamps)}")

        # Ensure to specify edge attributes to preserve during conversion
        dgl_graph = dgl.from_networkx(graph, edge_attrs=['timestamp'])  # Convert to DGL

         # Check for 'timestamp' attribute only if it is expected
        if 'timestamp' in dgl_graph.edata:
             if len(dgl_graph.edata['timestamp']) == 0:
                 print("Warning: No timestamp edge attribute found in the DGL graph. Proceeding without it.")
        else:
             print("No timestamp attribute present, proceeding without it.")
    else:
         # If not a directed graph, raise an error
        return graph

    return dgl_graph  # Return the DGL graph

      

def create_subgraphs(graph, interval):
    """
    Groups edges by the specified interval using UNIX timestamps to create subgraphs.

    Args:
        graph (Union[nx.DiGraph, dgl.DGLGraph]): The main directed graph.
        interval (str): The interval type ('month', 'week', etc.) used to group edges.

    Returns:
        List of tuples: Each tuple contains an interval label and a subgraph.
    """
    
    # Check if the graph is a DGL graph; if not, convert from NetworkX
    if not isinstance(graph, dgl.DGLGraph):
        if isinstance(graph, nx.DiGraph):
            graph = dgl.from_networkx(graph, edge_attrs=['timestamp'])
            print("Converted NetworkX graph to DGL graph.")

    # If static configuration, check if timestamp is present
    if 'timestamp' not in graph.edata:
        print("No timestamp data available for this graph. Proceeding with static analysis.")
        return [(None, graph)]  # Return the full graph as a single subgraph without grouping

    # Proceed with edge extraction
    edges_with_timestamps = []
    
    # Get all edges in the DGL graph
    src, dst = graph.edges()

    # Loop through edges and gather timestamps
    for i in range(graph.number_of_edges()):
        u = src[i].item()
        v = dst[i].item()
        if 'timestamp' in graph.edata:
            timestamp = graph.edata['timestamp'][i].item()
            edges_with_timestamps.append((u, v, timestamp))
        else:
            edges_with_timestamps.append((u, v, None))  

    # Raise an error if no edges with valid timestamps are found
    if not edges_with_timestamps:
        raise ValueError("No edges with timestamps found. Check if timestamps are present and correctly formatted.")
    
    
    # Create a DataFrame for edge data and convert UNIX timestamps to datetime
    edges_df = pd.DataFrame(edges_with_timestamps, columns=['source', 'target', 'timestamp'])
    print(f"Total edges: {len(edges_df)}")
    # Group edges based on the specified interval
    if interval:
        if interval == "month":
            edges_df['interval'] = pd.to_datetime(edges_df['timestamp'], unit='s').dt.to_period('M')
        elif interval == "week":
            edges_df['interval'] = pd.to_datetime(edges_df['timestamp'], unit='s').dt.to_period('W')
        else:
            raise ValueError(f"Unsupported interval type: {interval}")

        grouped_edges = edges_df.groupby('interval')
        # Check if any groups were created
        if grouped_edges.ngroups == 0:
            print("No subgraphs created.")
        
        subgraphs = []
       
        for interval_label, group in grouped_edges:
            # Create subgraph
            subgraph_dgl = dgl.graph(([], []), num_nodes=graph.num_nodes())
    
            for _, row in group.iterrows():
                subgraph_dgl.add_edges(int(row['source']), int(row['target']))
            subgraph_dgl.edata['timestamp'] = torch.tensor(group['timestamp'].values, dtype=torch.int64)
            subgraph_dgl.ndata[dgl.NID] = torch.arange(subgraph_dgl.num_nodes())
            
            # Transfer beliefs if present
            if 'beliefs' in graph.ndata:
                subgraph_dgl.ndata['beliefs'] = graph.ndata['beliefs']
            
            # Append the subgraph
            subgraphs.append((interval_label, copy.deepcopy(subgraph_dgl)))
            print(f"Created subgraph for interval {interval_label}: {len(group)} edges.")
        
            print(f"Graph ID for interval {interval_label}: {id(graph)}")
            print(f"SubGraph ID for interval {interval_label}: {id(subgraph_dgl)}")
            
            # Print the first few edges of the created DGL subgraph 
            src_sub, dst_sub = subgraph_dgl.edges()
            edge_pairs = list(zip(src_sub.tolist(), dst_sub.tolist()))
            num_edges_to_print = 5  # Print only the first 5 edges
            for s, d in edge_pairs[:num_edges_to_print]:
                timestamp = subgraph_dgl.edata['timestamp'][subgraph_dgl.edge_ids(s, d)].item()
                print(f"Edge ({s}, {d}) - Timestamp: {timestamp}")
            
        return subgraphs
            
    # If no interval is specified, just return the original graph wrapped in a list
    return [(None, graph)]
