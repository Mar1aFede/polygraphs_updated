#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 15:27:25 2024

@author: mariafedericanorelli
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 16:53:15 2024

Python script to retrieve data from a specified URL or locally stored, 
read the data into pandas DataFrames, creates a network graph using NetworkX, 
and then saves the graph data as a GraphML, GML or GEXF file.

@author: molly
"""

from collections import defaultdict
import shutil
import sys
import requests
import pandas as pd
import networkx as nx
import urllib.request
import os
import csv 
import matplotlib.pyplot as plt
from datetime import datetime


_DATACACHE = "~/polygraphs-cache/data"

def validate_origin(origin):
    """
    Validates the origin, ensuring it's either a URL or a local file.

    Args:
        origin (str): The origin URL or local file path.

    Returns:
        bool: True if the origin is a URL, False if it's a local file.

    Raises:
        ValueError: If the origin is not a string.
        Exception: If the origin is neither a URL nor a local file, 
                   or if a URL is unreachable, or if the local file does not exist.
    """
    # Check if origin is a string
    if not isinstance(origin, str):
        raise ValueError("File origin must be a string")

    # Check if origin is a URL
    remote = urllib.parse.urlparse(origin).scheme is not None

    if not remote:  # If it's not a URL, treat it as a local file path
        if not os.path.isfile(origin):  # Check if the file exists
            raise Exception(f"Local file does not exist: {origin}")

    return remote  # Return True if it's a URL, False if it's a local file

def fetch_file(origin, destination_folder):
    """
    Downloads remote origin to destination folder or copies local file to folder.
    """
    if not os.path.isdir(destination_folder):
        os.makedirs(destination_folder)  # Create the destination folder if it doesn't exist

    if urllib.parse.urlparse(origin).scheme:
        # If remote file, download to destination folder
        filename = os.path.basename(urllib.parse.urlparse(origin).path)
        destination_path = os.path.join(destination_folder, filename)
        response = requests.get(origin)
        with open(destination_path, 'wb') as f:
            f.write(response.content)
    else:
        # If local file, copy to destination folder
        # Remove 'file://' scheme if present
        origin = urllib.parse.urlparse(origin).path if origin.startswith('file://') else origin
        filename = os.path.basename(origin)
        destination_path = os.path.join(destination_folder, filename)
        shutil.copyfile(origin, destination_path)
    return destination_path

def txt_to_csv(input_file, output_file):
    """
    Converts a text file with ';' delimiter to a CSV file.

    Args:
        input_file (str): Path to the input text file.
        output_file (str): Path to the output CSV file.

    Returns:
        None
    """
    with open(input_file, 'r') as txt_file, open(output_file, 'w', newline='') as csv_file:
        txt_reader = csv.reader(txt_file, delimiter=';')
        csv_writer = csv.writer(csv_file)

        for row in txt_reader:
            csv_writer.writerow(row)

    print(f"Conversion completed. CSV file saved at: {output_file}")

def read_csv(file, source_column_name, destination_column_name, timestamp_column_name = None):
    """Reads csv and inspects it to get source and destination for edges"""
    df = pd.read_csv(file)  # Read the CSV file
    
    #Check if the first column contains space-separated values
    try:
       if df.iloc[:, 0].str.contains(' ').any():
           # Split the values into up to three columns
           split_values = df.iloc[:, 0].str.split(' ', expand=True)
    
   #Assign the split values to source, destination, and timestamp if present
           df[source_column_name] = split_values[0].astype(int)  # First value is the source node
           df[destination_column_name] = split_values[1].astype(int)  # Second value is the destination node

            # Check if a third column (timestamp) exists and assign it
           if split_values.shape[1] == 3:
               if timestamp_column_name:
                    df[timestamp_column_name] = split_values[2].astype(int)  # Third value is the timestamp
               else:
                    df['timestamp'] = split_values[2].astype(int)  # Default to 'timestamp' if not specified

            # Drop the original first column after splitting
           df = df.drop(columns=[df.columns[0]])

    except AttributeError:
        # Handle the case where splitting is not needed
        pass

    # Extract the source, destination, and optionally timestamp columns
    source = df[source_column_name].astype(int)
    destination = df[destination_column_name].astype(int)
    timestamps = df[timestamp_column_name].astype(int) if timestamp_column_name and timestamp_column_name in df.columns else None

    return df, source, destination, timestamps

def normalize_graph(graph):
    """Normalize all numerical attributes in the graph and keep original values as labels."""
    # Create a dict with original id
    original_id = dict(zip(list(graph.nodes()), list(graph.nodes())))
    # Set the original_id dictionary as node attributes
    nx.set_node_attributes(graph, original_id, "original_id")

    # Normalize node identifiers (from 0 to N) using default dict
    tbl = defaultdict(lambda: len(tbl))
    #normalized_node_edges = [(tbl[edge[0]], tbl[edge[1]]) for edge in graph.edges()]

    # Relabel nodes using lookup table
    normalized_graph = nx.relabel_nodes(graph, tbl)

    # Extract and normalize weights, preserving timestamps if they exist
    for u, v, data in graph.edges(data=True):
        if 'weight' in data:
            normalized_weight = (data['weight'] - min(data['weight'] for _, _, data in graph.edges(data=True))) / (
                max(data['weight'] for _, _, data in graph.edges(data=True)) - min(data['weight'] for _, _, data in graph.edges(data=True))
            ) if max(data['weight'] for _, _, data in graph.edges(data=True)) != min(data['weight'] for _, _, data in graph.edges(data=True)) else 0
            data['original_weight'] = data['weight']
            data['weight'] = normalized_weight
        
        # Preserve timestamp attribute if it exists without adding it twice
        if 'timestamp' in data:
            timestamp = data['timestamp']
            # Avoid adding 'timestamp' again if already in data
            data.pop('timestamp', None)
            normalized_graph.add_edge(tbl[u], tbl[v], timestamp=timestamp, **data)
        else:
            normalized_graph.add_edge(tbl[u], tbl[v], **data)

    return normalized_graph

def create_graph(source, destination):
    """Create a static graph based on the source and destination columns."""
    G = nx.DiGraph()
    for src, dst in zip(source, destination):
        G.add_edge(int(src), int(dst))
    print(f"The graph has {len(G.nodes())} nodes and {len(G.edges())} edges")
    return G

def create_temporal_graph(source, destination, timestamps):
    """Create a temporal graph using source, destination, and timestamp data."""
    G = nx.DiGraph()
    for src, dst, ts in zip(source, destination, timestamps):
        G.add_edge(int(src), int(dst), timestamp=int(ts))

    print(f"The temporal graph has {len(G.nodes())} nodes and {len(G.edges())} edges")
    # Inspect edge attributes after creating the NetworkX graph
    return G

def save_graph(graph, format_choice, destination_folder, filename):
    """
    Saves the graph in the specified format (GML, GraphML, or GEXF) to the destination folder.

    """
    filepath = os.path.join(destination_folder, filename + "." + format_choice.lower())

    if format_choice.lower() == "gml":
        nx.write_gml(graph, filepath)
        print(f"Graph saved as {filepath}")
    elif format_choice.lower() == "graphml":
        nx.write_graphml(graph, filepath)
        print(f"Graph saved as {filepath}")
    elif format_choice.lower() == "gexf":
        nx.write_gexf(graph, filepath)
        print(f"Graph saved as {filepath}")
    else:
        print("Invalid file format. Please choose either GML, GraphML, or GEXF.")

def print_graph_details(graph, num_edges=20, output_file="subgraph.png"):
    """
    Print details of a subset of nodes and edges and save the subgraph as an image.
    """
    # print("\nGraph Details:")
    # print(f"Number of nodes: {len(graph.nodes())}")
    # print(f"Number of edges: {len(graph.edges())}")

    # Sample the first `num_edges` edges (with attributes)
    subset_edges = list(graph.edges(data=True))[:num_edges]  # Get the first num_edges with data

    # Collect unique nodes from the selected edges
    subset_nodes = set()
    for edge in subset_edges:
        subset_nodes.update(edge[:2])

    #print("\nSample nodes:")
     #for node in subset_nodes:
        #print(node)

     #print("\nSample edges:")
     #for edge in subset_edges:
         #print(edge)

    # Create a subgraph
    subgraph = graph.edge_subgraph([e[:2] for e in subset_edges]).copy() 

    # Draw the subgraph using spring layout
    plt.figure(figsize=(12, 12))  # Set larger figure size
    pos = nx.spring_layout(subgraph)  # Use spring layout
    nx.draw(subgraph, pos, with_labels=True, node_color='skyblue', node_size=2000, edge_color='gray', font_size=15, font_weight='bold')

   # Add labels for the edges with timestamps
    for edge in subset_edges:
        if 'timestamp' in edge[2]:  # Ensure there's a timestamp attribute
            # Convert Unix timestamp to human-readable format
            timestamp = datetime.fromtimestamp(edge[2]['timestamp']).strftime('%Y-%m-%d')
            # Get the position for the edge to place the label
            x = (pos[edge[0]][0] + pos[edge[1]][0]) / 2
            y = (pos[edge[0]][1] + pos[edge[1]][1]) / 2
            plt.text(x, y + 0.1, timestamp, fontsize=10, color='black', bbox=dict(facecolor='white', alpha=0.5))


    # Save the subgraph as an image
    plt.savefig(output_file)
    print(f"\nSubgraph saved as {output_file}")

def extract_dataset_name(filepath):
    """
    Extracts the dataset name from the file path.
    """
    return os.path.splitext(os.path.basename(filepath))[0]



#Examples (use base, source and dest as inputs)
#Ex1 Bacon online dataset 
#Get today's date
# dt = datetime.datetime.now()
# date = "_{y}_{m:02d}_{d:02d}.csv".format(y=dt.year, m=dt.month, d=dt.day)
# base_undated = 'http://sixdegreesoffrancisbacon.com/data/SDFB_relationships'
# base = base_undated + date # input data
# source = 'person1_index' # input source
# dest = 'person2_index' #input dest

# # wikipedia dataset locally saved 
# base ='file:///Users/mariafedericanorelli/Desktop/humannetworkscience/datasets/tkgl-smallpedia/tkgl-smallpedia_base.csv'
# source= 'head'  #  column name for source
# dest = 'tail'  # column name for destination


#messages temporal dataset 
base ='file:///Users/mariafedericanorelli/Desktop/humannetworkscience/datasets/CollegeMsg.csv'
source = 'SRC'
dest = 'DST'




if __name__ == "__main__":
    # Dictionary to map variable names to their values
    predefined_vars = {
        'base': base,
        'source': source,
        'dest': dest
    }

    # User input for variable names
    origin = input("Enter URL, local file path, or private repository URL (or predefined variable name): ")
    source_column_name = input("Enter the name of the source column (or predefined variable name): ")
    destination_column_name = input("Enter the name of the destination column (or predefined variable name): ")

    # Retrieve the values of the variables if they are predefined variable names
    origin = predefined_vars.get(origin, origin)
    source_column_name = predefined_vars.get(source_column_name, source_column_name)
    destination_column_name = predefined_vars.get(destination_column_name, destination_column_name)

    destination_folder = os.path.expanduser(_DATACACHE)

    # Fetch the file based on the origin
    downloaded_file = None  # Initialize downloaded_file to None
    if origin.startswith("file://"):  # Check if it's a local file path
        downloaded_file = urllib.parse.urlparse(origin).path  # Extract the file path
    # Check if the downloaded file is already a CSV, otherwise convert it to CSV
        if downloaded_file.endswith('.csv'):
            print("Input file is already a CSV file.")
        else:
            output_file = downloaded_file.replace('.txt', '.csv')  # Define output_file here
            if not os.path.exists(output_file):
                txt_to_csv(downloaded_file, output_file)
                downloaded_file = output_file
            else:
                print(f"Output file {output_file} already exists. Please choose a different output file name.")
    elif os.path.isfile(origin):  # Check if it's a local file path without the "file://" scheme
        downloaded_file = origin
    # Check if the downloaded file is already a CSV, otherwise convert it to CSV
        if downloaded_file.endswith('.csv'):
            print("Input file is already a CSV file.")
        else:
            output_file = downloaded_file.replace('.txt', '.csv')  # Define output_file here
        if not os.path.exists(output_file):
            txt_to_csv(downloaded_file, output_file)
            downloaded_file = output_file
        else:
            print(f"Output file {output_file} already exists. Please choose a different output file name.")

    elif os.path.isfile(origin):  # Check if it's a local file path without the "file://" scheme
        downloaded_file = origin
        # Check if the downloaded file is already a CSV, otherwise convert it to CSV
        if downloaded_file.endswith('.csv'):
            print("Input file is already a CSV file.")
        else:
            output_file = downloaded_file.replace('.txt', '.csv')
            if not os.path.exists(output_file):
                txt_to_csv(downloaded_file, output_file)
                downloaded_file = output_file
            else:
                print("Output file already exists. Please choose a different output file name.")
    else:
    # If it's not a local file path, treat it as a URL and proceed with the existing code
        is_remote = validate_origin(origin)
        if is_remote:
            try:
                downloaded_file = fetch_file(origin, destination_folder)
                print("Downloaded file:", downloaded_file)

            # Check if the downloaded file is already a CSV, otherwise convert it to CSV
                if downloaded_file.endswith('.csv'):
                    print("Input file is already a CSV file.")
                else:
                    output_file = downloaded_file.replace('.txt', '.csv')
                    if not os.path.exists(output_file):
                        txt_to_csv(downloaded_file, output_file)
                        downloaded_file = output_file
                    else:
                        print("Output file already exists. Please choose a different output file name.")

            except Exception as e:
                print("Error:", e)
                sys.exit(1)  # Exit the script if there's an error downloading the file
        else:
            downloaded_file = origin

    # Extract dataset name
    dataset_name = extract_dataset_name(downloaded_file)
    
   # Read the CSV file and assign source and destination columns
    try:
        df, source, destination, timestamps = read_csv(downloaded_file, source_column_name, destination_column_name, timestamp_column_name='timestamp')   
        if df is not None:
            print("Source column:", source_column_name)
            print("Destination column:", destination_column_name)
    except Exception as e:
        print("Error:", e)
        sys.exit(1)  # Exit the script if there's an error reading the CSV file
    
   # Verify variables before creating the graph
    print("Source:", source.head())  # Print first few values for verification
    print("Destination:", destination.head())
    if timestamps is not None:
        print("Timestamps:", timestamps.head())  # Print timestamps if present

    # Determine if the graph is temporal or static, save it and normalize it 
    try:
        if timestamps is not None:  # Check if timestamps were successfully extracted
            graph = create_temporal_graph(source, destination, timestamps)
        else:
            graph = create_graph(source, destination)
        
        save_graph(graph, "GML", destination_folder, f"{dataset_name}_graph")

        # Normalize the graph for both static and temporal cases
        normalized_graph = normalize_graph(graph)
        print("Graph normalized successfully.")
        save_graph(normalized_graph, "GML", destination_folder, f"{dataset_name}_graph_normalized")
            
    except Exception as e:
          print("Error:", e)
          sys.exit(1)     

   # Create image of graph 
    try: 
    
        print_graph_details(graph, num_nodes=50, num_edges=50, output_file=f"{dataset_name}_subgraph.png")  # Print details of the graph

        # Save the non-normalized graph
        
        print_graph_details(normalized_graph, num_nodes=50, num_edges=50, output_file=f"{dataset_name}_subgraph_normalized.png")
       

    except Exception as e:
        print("Error:", e)
        sys.exit(1)  # Exit the script if there's an error creating or saving the graph




