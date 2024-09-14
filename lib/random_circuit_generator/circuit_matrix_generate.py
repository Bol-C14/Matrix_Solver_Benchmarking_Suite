import networkx as nx
import random
import matplotlib.pyplot as plt
import importlib
import inspect
import numpy as np
import os
from MyNode import MyNode
from device import Vs
from collections import defaultdict
from scipy.sparse import coo_matrix

# Load the modules
circuit_module = importlib.import_module("model")
circuit_module_analog = importlib.import_module("AnalogModels")
circuit_module_digital = importlib.import_module("DigitalModels")

# Specify the modules to be used
modules = [obj for name, obj in inspect.getmembers(circuit_module) if
           inspect.isclass(obj) and obj.__module__ == circuit_module.__name__]
modules_analog = [obj for name, obj in inspect.getmembers(circuit_module_analog) if
                  inspect.isclass(obj) and obj.__module__ == circuit_module_analog.__name__]
modules_digital = [obj for name, obj in inspect.getmembers(circuit_module_digital) if
                   inspect.isclass(obj) and obj.__module__ == circuit_module_digital.__name__]
target_model_library = modules

a_matrix = []
b_matrix = []
number_of_rows: int = 0

def generate_circuit(num_nodes, extra_connections):
    '''
    Generates a random circuit graph with the specified number of nodes and extra connections.
    
    @param num_nodes: the number of nodes in the graph
    @param extra_connections: the number of extra connections to add to the graph
    @return (networkx.DiGraph): a directed graph representing the circuit
    '''    
    # Create an empty directed graph
    G = nx.DiGraph()

    # Create a list of nodes
    nodes = [MyNode(i) for i in range(num_nodes)]

    # Create a map from ids to MyNode objects
    node_map = {node.id: node for node in nodes}

    # Generate a connected graph
    start_node = random.choice(nodes)
    G.add_node(start_node)
    unconnected_nodes = set(nodes) - {start_node}

    while unconnected_nodes:
        node = random.choice(list(unconnected_nodes))
        predecessor = random.choice(list(G.nodes))

        # limit the number of connections to 2
        if len(predecessor.successors) < 2 and len(node.predecessors) < 2:
            predecessor.successors.append(node)  # add node to predecessor's successors
            node.predecessors.append(predecessor)  # add predecessor to node's predecessors
            G.add_edge(predecessor, node)  # Add a directed edge
            unconnected_nodes.remove(node)

    # Add additional connections
    for _ in range(extra_connections):
        node1 = random.choice(list(G.nodes))  # Node1 can be any node
        # Node2 can be any node but the start node and limit the number of its predecessors to 2
        node2 = random.choice(list(G.nodes - {start_node}))
        while G.in_degree(node2) >= 2:
            node2 = random.choice(list(G.nodes - {start_node}))

        node1.successors.append(node2)  # add node2 to node1's successors
        node2.predecessors.append(node1)  # add node1 to node2's predecessors
        G.add_edge(node1, node2)

    return G

def visualize_graph(G):
    plt.figure(figsize=(10, 8))  # Increase plot size

    # Layout
    pos = nx.kamada_kawai_layout(G)

    # Get unique modules and assign colors
    modules = set(node.chosen_module for node in G.nodes())
    color_map = {module: plt.cm.jet(i / len(modules)) for i, module in enumerate(modules)}
    node_colors = [color_map[node.chosen_module] for node in G.nodes()]

    # Nodes
    nx.draw_networkx_nodes(G, pos, node_size=400, alpha=0.85, node_color=node_colors)

    # Edges
    nx.draw_networkx_edges(G, pos, width=2, edge_color="gray")

    # Labels
    labels = {node: node.id for node in G.nodes}
    label_pos = {k: [v[0], v[1] + 0.05] for k, v in pos.items()}  # Slightly shift label position
    nx.draw_networkx_labels(G, label_pos, labels, font_size=10)

    # Legend
    for module, color in color_map.items():
        plt.plot([], [], 'o', color=color, label=module)
    plt.legend()

    plt.title("Node Graph")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("node_graph.png", dpi=600)
    plt.show()

def print_degrees_and_connections(G):
    for node in G.nodes:
        print(f"Node {node.id} has in-degree {G.in_degree(node)} and out-degree {G.out_degree(node)}")
        print(f"Node {node.id} connects to: {[n.id for n in G.successors(node)]}")
        print(f"Node {node.id} is connected from: {[n.id for n in G.predecessors(node)]}")
        print("---")

def dfs_traversal(G):
    visited_nodes = []  # Keep track of visited nodes

    # Traverse all nodes using DFS
    for node in nx.dfs_preorder_nodes(G):
        visited_nodes.append(node)  # Mark node as visited
        print(f"Visited node: {node.id}")  # Node is now an instance of the Node class

    return visited_nodes

def assign_modules_to_nodes(G):
    # Find the start node in Graph G
    start_node = None
    for node in G.nodes:
        if G.in_degree(node) == 0:
            start_node = node
            break

    for node in G.nodes:
        possible_classes = []
        for cls in target_model_library:
            if G.in_degree(node) == cls.number_of_inputs:
                possible_classes.append(cls)

        # If the node has in, then it can escape the input degree check
        if node == start_node:
            for cls in target_model_library:
                possible_classes.append(cls)

        if possible_classes:
            # Choose a random class from the possible classes
            chosen_class = random.choice(possible_classes)
            node.assign_chosen_module(chosen_class)
        else:
            print(f"Node {node.id} has no possible classes")
            exit(1)

def assign_voltage_sources(G, num_nodes):
    Vdd = Vs(1, 0, 3.3, num_nodes)
    global a_matrix, b_matrix
    a_matrix, b_matrix = Vdd.assign_matrix()
    for node in G.nodes:
        if node.my_module.is_voltage_node:
            node.my_module.assign_voltage_node(1)

def assign_start_index(G):
    start_index = 1
    for node in nx.dfs_preorder_nodes(G):
        if node.number_of_internalNodes is None:
            print(f"Node {node.id} has not been assigned a number_of_internalNodes value")
            continue
        node.assign_start_index(start_index)
        start_index += node.number_of_internalNodes

    return start_index

def build_modules(G, num_nodes):
    # Create an instance according to the chosen class
    for index, node in enumerate(G.nodes):
        node.create_modules()

        # Assign the input node to the first node (0)
        if index == 0:
            input_number = node.my_module.number_of_inputs
            # Assign the length of input_number of a list of 0 to the input node
            for i in range(input_number):
                node.my_module.assign_input_node(0)

    # Assign the global voltage source node
    assign_voltage_sources(G, num_nodes)

    # Traverse all nodes using DFS and assign the input and output nodes
    for node in nx.dfs_preorder_nodes(G):
        node.assign_nodes()

    for node in G.nodes:
        a = []
        b = []
        a, b = node.build_matrix()
        a_matrix.extend(a)
        b_matrix.extend(b)

def write_to_mtx(a_matrix, b_matrix, file_path, num_rows):
    """
    Write a_matrix to a Matrix Market file and generate a sparse matrix image.
    
    Parameters:
    a_matrix (list of dicts): List of dictionaries with keys 'x', 'y', 'value'.
    b_matrix (list of dicts): List of dictionaries with keys 'x', 'y', 'value'.
    file_path (str): Path to the output file.
    num_rows (int): Number of rows/columns in the matrix.
    """
    
    # Ensure the directory exists, if not, create it
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Sort the list of dictionaries by 'y' and then by 'x'
    a_matrix = sorted(a_matrix, key=lambda coord: (coord['y'], coord['x']))

    # Iterate over the dictionaries in the list
    result_dict = defaultdict(float)
    for d in a_matrix:
        # Use tuple (x, y) as a key for the dictionary and add the value to the current sum
        result_dict[(d['x'], d['y'])] += d['value']
    a_matrix = [{"x": x, "y": y, "value": v} for (x, y), v in result_dict.items()]
    total_nonzero_elements = len(a_matrix)

    # Write to Matrix Market file
    with open(file_path, 'w') as file:
        # Write the Matrix Market header
        file.write("%%MatrixMarket matrix coordinate real general\n")
        file.write(f"{num_rows} {num_rows} {total_nonzero_elements}\n")

        # Write the coordinate values
        for i in a_matrix:
            x = i['x']
            y = i['y']
            value = i['value']
            file.write(f"{x} {y} {value}\n")

    # Prepare data for plotting the sparse matrix
    x = [item['x'] for item in a_matrix]
    y = [item['y'] for item in a_matrix]
    data = [item['value'] for item in a_matrix]

    # Create sparse matrix
    sparse_matrix = coo_matrix((data, (x, y)))
    plt.spy(sparse_matrix, markersize=1)
    # plt.savefig('sparse_matrix.png', dpi=600)
    # plt.show()

def run_circuit_generation(node_of_iterations=2, node_range_1=(5, 100, 5), node_range_2=(100, 2000, 20)):
    """
    Generates circuits and writes the corresponding matrices to files.

    Parameters:
    node_of_iterations (int): Number of iterations for generating circuits.
    node_range_1 (tuple): Range of node numbers for the first part of the sweep (start, end, step).
    node_range_2 (tuple): Range of node numbers for the second part of the sweep (start, end, step).
    """
    node_numbers = np.arange(*node_range_1)
    node_numbers = np.concatenate((node_numbers, np.arange(*node_range_2)))

    for i in range(node_of_iterations):
        for node_number in node_numbers:
            # Set extra_connections
            extra_connections = node_number // 4
            
            # Generate the circuit
            graph = generate_circuit(node_number, extra_connections)
            assign_modules_to_nodes(graph)
            number_of_rows = assign_start_index(graph)
            print(number_of_rows)
            build_modules(graph, number_of_rows)
            
            # Write the matrices to a file
            filename = f'Matrix_Classification_ML/Circuit_Matrix_Analysis/generated_circuits/random_circuit_{node_number}_{i + 1}.mtx'
            write_to_mtx(a_matrix, b_matrix, filename, number_of_rows)

            # Clear the matrices and number_of_rows for next loop
            number_of_rows = 0
            a_matrix.clear()
            b_matrix.clear()

            # Print number of nodes done and total nodes
            print(f"Generated matrices for {node_number} nodes")
            
    print("Circuit generation completed.")
    # Print total number of matrices generated
    print(f"Total number of matrices generated: {node_of_iterations * len(node_numbers)}")

# Example usage
if __name__ == "__main__":
    run_circuit_generation()
