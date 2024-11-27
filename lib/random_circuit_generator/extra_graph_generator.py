import networkx as nx
import random
import matplotlib.pyplot as plt
import importlib
import inspect

from rich.progress import Progress

from .MyNode import MyNode
from .device import Vs
from collections import defaultdict
from scipy.sparse import coo_matrix
import os
import numpy as np
from rich.console import Console
import logging

# Initialize logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Initialize console for pretty output
console = Console()

# Load the modules
circuit_module = importlib.import_module(".model", package="lib.random_circuit_generator")
circuit_module_analog = importlib.import_module(".AnalogModels", package="lib.random_circuit_generator")
circuit_module_digital = importlib.import_module(".DigitalModels", package="lib.random_circuit_generator")

# Specify the modules to be used
modules = [obj for name, obj in inspect.getmembers(circuit_module) if
           inspect.isclass(obj) and obj.__module__ == circuit_module.__name__]
modules_analog = [obj for name, obj in inspect.getmembers(circuit_module_analog) if
                  inspect.isclass(obj) and obj.__module__ == circuit_module_analog.__name__]
modules_digital = [obj for name, obj in inspect.getmembers(circuit_module_digital) if
                   inspect.isclass(obj) and obj.__module__ == circuit_module_digital.__name__]
targetModelLibrary = modules + modules_analog + modules_digital  # Combine all modules

a_matrix = []
b_matrix = []
numberOfRows: int = 0


def generate_circuit(num_nodes, extra_connections):
    """
    Generates a circuit graph with a specified number of nodes and extra connections.
    Supports multiple connection strategies: random, star, ring, mesh.
    """
    # Create an empty directed graph
    G = nx.DiGraph()

    # Create a list of nodes
    nodes = [MyNode(i) for i in range(num_nodes)]
    G.add_nodes_from(nodes)

    # Randomly choose a connection strategy
    connection_strategy = random.choice(['random', 'star', 'ring', 'mesh'])
    logger.info(f"Using connection strategy: {connection_strategy}")

    if connection_strategy == 'random':
        connect_random(G, nodes, extra_connections)
    elif connection_strategy == 'star':
        connect_star(G, nodes)
    elif connection_strategy == 'ring':
        connect_ring(G, nodes)
    elif connection_strategy == 'mesh':
        connect_mesh(G, nodes)
    else:
        # Default to random if unknown strategy
        connect_random(G, nodes, extra_connections)

    return G


def connect_random(G, nodes, extra_connections):
    """
    Connects nodes randomly ensuring the graph is connected.
    Adds a specified number of extra random connections.
    """
    # Start with a random node
    start_node = random.choice(nodes)
    connected_nodes = {start_node}
    unconnected_nodes = set(nodes) - connected_nodes

    while unconnected_nodes:
        node = random.choice(list(unconnected_nodes))
        predecessor = random.choice(list(connected_nodes))
        G.add_edge(predecessor, node)
        predecessor.successors.append(node)
        node.predecessors.append(predecessor)
        connected_nodes.add(node)
        unconnected_nodes.remove(node)

    # Add extra random connections
    for _ in range(extra_connections):
        node1, node2 = random.sample(nodes, 2)
        if not G.has_edge(node1, node2):
            G.add_edge(node1, node2)
            node1.successors.append(node2)
            node2.predecessors.append(node1)


def connect_star(G, nodes):
    """
    Connects all nodes to a central node, forming a star topology.
    """
    if not nodes:
        return

    center_node = random.choice(nodes)
    for node in nodes:
        if node != center_node:
            G.add_edge(center_node, node)
            center_node.successors.append(node)
            node.predecessors.append(center_node)


def connect_ring(G, nodes):
    """
    Connects nodes in a ring (circular) topology.
    """
    num_nodes = len(nodes)
    if num_nodes < 2:
        return

    for i in range(num_nodes):
        node1 = nodes[i]
        node2 = nodes[(i + 1) % num_nodes]
        G.add_edge(node1, node2)
        node1.successors.append(node2)
        node2.predecessors.append(node1)


def connect_mesh(G, nodes):
    """
    Fully connects all nodes to each other, forming a mesh topology.
    """
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if i != j:
                node1 = nodes[i]
                node2 = nodes[j]
                if not G.has_edge(node1, node2):
                    G.add_edge(node1, node2)
                    node1.successors.append(node2)
                    node2.predecessors.append(node1)


def visualize_graph(G):
    plt.figure(figsize=(10, 8))  # Increase plot size

    # Layout
    pos = nx.kamada_kawai_layout(G)

    # Get unique modules and assign colors
    modules = set(node.chosen_module.__class__.__name__ for node in G.nodes)
    color_map = {module: plt.cm.jet(i / len(modules)) for i, module in enumerate(modules)}
    node_colors = [color_map[node.chosen_module.__class__.__name__] for node in G.nodes]

    # Nodes
    nx.draw_networkx_nodes(G, pos, node_size=400, alpha=0.85, node_color=node_colors)

    # Edges
    nx.draw_networkx_edges(G, pos, width=2, edge_color="gray")

    # Labels
    labels = {node: node.id for node in G.nodes}
    label_pos = {k: [v[0], v[1] + 0.05] for k, v in pos.items()}  # slightly shift label position
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
        print(f"Node {node.id} connects to: {[n.id for n in node.successors]}")
        print(f"Node {node.id} is connected from: {[n.id for n in node.predecessors]}")
        print("---")


def dfs_traversal(G):
    visited_nodes = []  # Keep track of visited nodes

    # Traverse all nodes using DFS
    for node in nx.dfs_preorder_nodes(G):
        visited_nodes.append(node)  # Mark node as visited
        print(f"Visited node: {node.id}")  # Node is now an instance of the Node class

    return visited_nodes


def assign_modules_to_nodes(G):
    """
    Assigns a module (circuit component) to each node based on its in-degree.
    If no exact match is found, assigns the closest matching module.
    """
    # find the start node in Graph G
    start_node = None
    for node in G.nodes:
        if G.in_degree(node) == 0:
            start_node = node
            break

    for node in G.nodes:
        possible_classes = []
        for cls in targetModelLibrary:
            if G.in_degree(node) == cls.number_of_inputs:
                possible_classes.append(cls)

        # If the node has no predecessors, it's a start node
        if node == start_node:
            possible_classes.extend(targetModelLibrary)

        if possible_classes:
            # Choose a random class from the possible classes
            chosen_class = random.choice(possible_classes)
            node.assign_chosen_module(chosen_class)
            # logger.info(f"Node {node.id} is assigned class {chosen_class.__name__}")
        else:
            # If no exact match, choose the closest match based on number of inputs
            closest_match = min(targetModelLibrary, key=lambda cls: abs(cls.number_of_inputs - G.in_degree(node)))
            node.assign_chosen_module(closest_match)
            logger.warning(f"Node {node.id} has no exact module match; assigned closest match {closest_match.__name__}")


def assign_voltageSources(G, num_nodes):
    """
    Assigns voltage sources to nodes that require them.
    """
    Vdd = Vs(1, 0, 3.3, num_nodes)
    [a_matrix_local, b_matrix_local] = Vdd.assign_matrix()
    for node in G.nodes:
        if hasattr(node.chosen_module, 'is_voltage_node') and node.chosen_module.is_voltage_node:
            node.chosen_module.assign_voltage_node(1)

    # Merge local matrices into global
    a_matrix.extend(a_matrix_local)
    b_matrix.extend(b_matrix_local)


def assign_start_index(G):
    """
    Assigns start indices to nodes based on DFS traversal.
    """
    start_index = 1
    for node in nx.dfs_preorder_nodes(G):
        if node.number_of_internalNodes is None:
            print(f"Node {node.id} has not been assigned a number_of_internalNodes value")
            continue
        node.assign_start_index(start_index)
        start_index += node.number_of_internalNodes

    return start_index


def building_modules(G, num_nodes):
    """
    Creates module instances, assigns inputs and outputs, and builds the circuit matrices.
    """
    # Create an Instance according to the chosen class
    for index, node in enumerate(G.nodes):
        node.create_modules()

        # Assign the input node to the first node (0)
        if index == 0:
            inputnumber = node.my_module.number_of_inputs
            # assign the length of inputnumber of a list of 0 to the input node
            for i in range(inputnumber):
                node.my_module.assign_input_node(0)

    # Assign the global voltage source node
    assign_voltageSources(G, num_nodes)

    # Traverse all nodes using DFS and assign the input and output nodes
    for node in nx.dfs_preorder_nodes(G):
        node.assign_nodes()

    for node in G.nodes:
        # # print the node's module's output_nodes and input_nodes
        # print(f"Node {node.id} has output_nodes: {node.my_module.output_nodes}")
        # print(f"Node {node.id} has input_nodes: {node.my_module.input_nodes}")
        a = []
        b = []
        [a, b] = node.build_matrix()
        a_matrix.extend(a)
        b_matrix.extend(b)


def write_to_mtx(a_matrix, b_matrix, file_path, num_rows):
    """
    Writes the circuit matrices to a Matrix Market (.mtx) file.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Sort the list of dictionaries by y and then by x
    a_matrix_sorted = sorted(a_matrix, key=lambda coord: (coord['y'], coord['x']))

    # Iterate over the dictionaries in the list
    result_dict = defaultdict(float)
    for d in a_matrix_sorted:
        # Use tuple (x, y) as a key for the dictionary and add the value to the current sum
        result_dict[(d['x'], d['y'])] += d['value']
    a_matrix_aggregated = [{"x": x, "y": y, "value": v} for (x, y), v in result_dict.items()]
    Total_nonzero_elements = len(a_matrix_aggregated)

    with open(file_path, 'w') as file:
        # Write the Matrix Market header
        file.write("%%MatrixMarket matrix coordinate real general\n")
        file.write(f"{num_rows} {num_rows} {Total_nonzero_elements}\n")

        # Write the coordinate values
        for i in a_matrix_aggregated:
            x = i['x']
            y = i['y']
            value = i['value']
            file.write(f"{x} {y} {value}\n")

    # print the sparse image of a_matrix
    x = [item['x'] for item in a_matrix_aggregated]
    y = [item['y'] for item in a_matrix_aggregated]
    data = [item['value'] for item in a_matrix_aggregated]

    # Create sparse matrix
    sparse_matrix = coo_matrix((data, (x, y)))
    plt.spy(sparse_matrix, markersize=1)
    # plt.savefig('sparse_matrix.png', dpi=600)
    plt.show()


def run_circuit_generation(nodeofiterations, nodenumbers, output_directory="random_circuit_matrixs", verbose=False):
    """
    Runs the circuit generation process for a given number of iterations and node numbers.
    """
    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Setup progress bar using rich
    with Progress() as progress:
        task = progress.add_task("[green]Generating random circuits...", total=nodeofiterations * len(nodenumbers))

        for i in range(nodeofiterations):
            for nodenumber in nodenumbers:
                extra_connections = nodenumber // 4

                if verbose:
                    logger.info(
                        f"Generating circuit with {nodenumber} nodes and {extra_connections} extra connections.")

                # Generate circuit
                Graph = generate_circuit(nodenumber, extra_connections)
                assign_modules_to_nodes(Graph)
                numberOfRows = assign_start_index(Graph)

                if verbose:
                    logger.info(f"Number of rows for node {nodenumber}: {numberOfRows}")

                # Build modules and matrices
                building_modules(Graph, numberOfRows)

                # Define the file name for the matrix file
                filename = os.path.join(output_directory, f'random_circuit_{nodenumber}_{i + 1}.mtx')
                write_to_mtx(a_matrix, b_matrix, filename, numberOfRows)

                # Reset variables for the next iteration
                numberOfRows = 0
                a_matrix.clear()
                b_matrix.clear()

                # Update the progress bar
                progress.advance(task)

                if verbose:
                    logger.info(f'Completed {nodenumber} nodes for iteration {i + 1}/{nodeofiterations}')


# Example usage from an outer script:
if __name__ == "__main__":
    # Parameters
    nodeofiterations = 10
    nodenumbers = np.arange(5, 100, 5)
    nodenumbers = np.concatenate((nodenumbers, np.arange(100, 1000, 50)))

    # Run the circuit generation and matrix writing
    run_circuit_generation(nodeofiterations, nodenumbers)
