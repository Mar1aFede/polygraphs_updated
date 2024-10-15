import pandas as pd  # Importing pandas library for data manipulation
import h5py  # Importing h5py library for working with HDF5 files


class BeliefProcessor:
    def get_beliefs(self, hd5_file_path, bin_file_path, graph_converter):
        # Retrieve graph object(s) from bin file using the provided graph_converter
        graph_list = graph_converter.get_graph_object(bin_file_path)

        # Ensure we are dealing with a list (even if it's a single graph, treat uniformly)
        if not isinstance(graph_list, list):
            graph_list = [graph_list]  # Convert to a list if it's a single graph

        # Initialize a list to collect beliefs for all graphs (for temporal networks)
        all_beliefs = []

        # Iterate over each graph in the list
        for idx, graph in enumerate(graph_list):
            # Check if the graph is valid
            if graph is None or graph.number_of_nodes() == 0:
                print(f"[ERROR] No nodes in graph from {bin_file_path}, interval: {idx}.")
                continue  # Skip invalid graphs

            # Open the HDF5 file in read mode
            with h5py.File(hd5_file_path, "r") as fp:
                # Extract the keys (iteration numbers) from the 'beliefs' group in the HDF5 file
                _keys = sorted(map(int, fp["beliefs"].keys()))
                # Initialize a list to store iteration number and corresponding beliefs
                initial_beliefs = graph.ndata["beliefs"].tolist()
                iterations = [(0, initial_beliefs)]

                # Iterate over each key (iteration number) in the HDF5 file
                for key in _keys:
                    # Retrieve beliefs data for the current iteration
                    beliefs = fp["beliefs"][str(key)]
                    # Append the iteration number and beliefs data to the list
                    iterations.append((key, list(beliefs)))

            # Create a DataFrame for this graph without the "interval" column initially
            data = []
            for key, beliefs in iterations:
                for node, belief in enumerate(beliefs):
                    data.append((key, node, belief))  # Append step, node, belief

            # Convert the collected data into a DataFrame
            iterations_df = pd.DataFrame(
                data, columns=["step", "node", "beliefs"]
            )

            # Append the DataFrame for this graph to the overall beliefs list
            all_beliefs.append(iterations_df)

        # Concatenate all subgraph DataFrames along the first axis
        if len(all_beliefs) > 0:
            final_beliefs_df = pd.concat(all_beliefs, ignore_index=True)
        else:
            # If no valid graphs were processed, return an empty DataFrame
            return pd.DataFrame()

        # Set the updated index back to "step" and "node"
        final_beliefs_df.set_index(["step", "node"], inplace=True)

        # Return the final DataFrame containing beliefs data for all iterations and subgraphs
        return final_beliefs_df





class Beliefs:
    """
    The Beliefs class stores the beliefs of simulations that have been
    explicitly loaded for analysis using the Belief Processor

    This class provides an iterator and get item to access beliefs
    """

    def __init__(self, dataframe, belief_processor, graph_converter):
        self.bin_file_path = dataframe["bin_file_path"]
        self.hd5_file_path = dataframe["hd5_file_path"]
        self.belief_processor = belief_processor
        self.graph_converter = graph_converter
        self.beliefs = [None] * len(dataframe)
        self.index = 0

    def __getitem__(self, index):
        if index > len(self.beliefs):
            raise IndexError("Simulation index out of range")
        return self.get(index)

    def __len__(self):
        return len(self.beliefs)

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.beliefs):
            self.index = 0
            raise StopIteration
        value = self.get(self.index)
        self.index += 1
        return value

    def get(self, index):
        # Return a saved beliefs dataframe using its index or load from file
        if self.beliefs[index] is not None:
            return self.beliefs[index]
        elif index < len(self.beliefs):
            self.beliefs[index] = self.belief_processor.get_beliefs(
                self.hd5_file_path[index],
                self.bin_file_path[index],
                self.graph_converter,
            )
            return self.beliefs[index]
        else:
            raise IndexError("Simulation index out of range")
