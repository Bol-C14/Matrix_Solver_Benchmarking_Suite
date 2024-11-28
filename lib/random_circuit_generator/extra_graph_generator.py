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
targetModelLibrary = modules
print("Available classes and their number_of_inputs:")
for cls in targetModelLibrary:
    print(f"Class {cls.__name__} has number_of_inputs {cls.number_of_inputs}")

a_matrix = []
b_matrix = []
numberOfRows: int = 0

def generate_circuit(num_nodes, connection_method='random', extra_connections=0, max_in_degree=2):
    G = nx.DiGraph()

    # Create a list of nodes
    nodes = [MyNode(i) for i in range(num_nodes)]
    node_map = {node.id: node for node in nodes}
    G.add_nodes_from(nodes)

    # Step 1: Create an initial cycle to ensure every node has at least one incoming edge
    if num_nodes > 1:
        for i in range(num_nodes):
            G.add_edge(nodes[i], nodes[(i + 1) % num_nodes])

    # Step 2: Add additional connections based on the connection method
    if connection_method == 'random':
        # Randomly connect nodes without exceeding max_in_degree
        for node in nodes:
            current_in_degree = G.in_degree(node)
            possible_targets = [n for n in nodes if n != node and G.in_degree(n) < max_in_degree and not G.has_edge(node, n)]
            # Determine how many additional edges to add for this node
            max_additional_edges = min(2 - current_in_degree, len(possible_targets))
            num_edges = random.randint(0, max_additional_edges) if max_additional_edges > 0 else 0
            targets = random.sample(possible_targets, num_edges)
            for target in targets:
                G.add_edge(node, target)

    elif connection_method == 'fully_connected':
        # Fully connect all nodes without exceeding in-degree
        for node in nodes:
            for target in nodes:
                if node != target and G.in_degree(target) < max_in_degree and not G.has_edge(node, target):
                    G.add_edge(node, target)

    elif connection_method == 'scale_free':
        # Generate a scale-free graph and enforce in-degree constraints
        temp_graph = nx.scale_free_graph(num_nodes)
        mapping = {i: node for i, node in enumerate(nodes)}
        G_scale = nx.DiGraph(nx.relabel_nodes(temp_graph, mapping))

        # Remove edges that cause in-degree to exceed max_in_degree
        for node in G_scale.nodes():
            while G_scale.in_degree(node) > max_in_degree:
                in_edges = list(G_scale.in_edges(node))
                if not in_edges:
                    break
                edge_to_remove = random.choice(in_edges)
                G_scale.remove_edge(*edge_to_remove)

        # Merge the scale-free edges into the initial cycle graph
        for edge in G_scale.edges():
            G.add_edge(*edge)

    elif connection_method == 'small_world':
        # Generate a small-world graph and enforce in-degree constraints
        temp_graph = nx.newman_watts_strogatz_graph(num_nodes, k=4, p=0.5)
        for edge in temp_graph.edges():
            source = nodes[edge[0]]
            target = nodes[edge[1]]
            if G.in_degree(target) < max_in_degree and not G.has_edge(source, target):
                G.add_edge(source, target)

    else:
        # Default to connected graph with extra connections and enforce in-degree
        for _ in range(extra_connections):
            node1 = random.choice(list(G.nodes))
            possible_targets = [n for n in G.nodes if n != node1 and G.in_degree(n) < max_in_degree and not G.has_edge(node1, n)]
            if possible_targets:
                node2 = random.choice(possible_targets)
                G.add_edge(node1, node2)

    # Step 3: Update successors and predecessors for all nodes
    for node in G.nodes:
        node.successors = list(G.successors(node))
        node.predecessors = list(G.predecessors(node))

    # Step 4: Verify all nodes have in-degree within the allowed range
    for node in G.nodes:
        in_deg = G.in_degree(node)
        if in_deg > max_in_degree:
            logger.error(f"Node {node.id} has in-degree {in_deg}, which exceeds the maximum allowed.")
            exit(1)
        elif in_deg < 1:
            logger.error(f"Node {node.id} has in-degree {in_deg}, which is below the minimum required.")
            exit(1)

    return G

def visualize_graph(G, filename):
    plt.figure(figsize=(10, 8))

    # Layout
    pos = nx.spring_layout(G)

    # Nodes
    nx.draw_networkx_nodes(G, pos, node_size=400, alpha=0.85)

    # Edges
    nx.draw_networkx_edges(G, pos, width=2, edge_color="gray")

    # Labels
    labels = {node: node.id for node in G.nodes}
    nx.draw_networkx_labels(G, pos, labels, font_size=10)

    plt.title("Node Graph")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(filename, dpi=600)
    plt.close()

def print_degrees_and_connections(G):
    for node in G.nodes:
        print(f"Node {node.id} has in-degree {G.in_degree(node)} and out-degree {G.out_degree(node)}")
        print(f"Node {node.id} connects to: {[n.id for n in G.successors(node)]}")
        print(f"Node {node.id} is connected from: {[n.id for n in G.predecessors(node)]}")
        print("---")

def dfs_traversal(G):
    visited_nodes = []

    for node in nx.dfs_preorder_nodes(G):
        visited_nodes.append(node)
        print(f"Visited node: {node.id}")

    return visited_nodes

def assign_modules_to_nodes(G):
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

        # If the node has in, then it can escape the input degree check
        if node == start_node:
            for cls in targetModelLibrary:
                possible_classes.append(cls)

        if possible_classes:
            # Choose a random class from the possible classes
            chosen_class = random.choice(possible_classes)
            node.assign_chosen_module(chosen_class)
            # print(f"Node {node.id} is assigned class {chosen_class.__name__}")
        else:
            print(f"Node {node.id} has no possible classes")
            exit(1)

def assign_voltageSources(G, num_nodes):
    Vdd = Vs(1, 0, 3.3, num_nodes)
    [a_matrix_v, b_matrix_v] = Vdd.assign_matrix()
    a_matrix.extend(a_matrix_v)
    b_matrix.extend(b_matrix_v)
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

def building_modules(G, num_nodes):
    for index, node in enumerate(G.nodes):
        node.create_modules()

        if index == 0:
            inputnumber = node.my_module.number_of_inputs
            for i in range(inputnumber):
                node.my_module.assign_input_node(0)

    assign_voltageSources(G, num_nodes)

    for node in nx.dfs_preorder_nodes(G):
        node.assign_nodes()

    for node in G.nodes:
        a, b = node.build_matrix()
        a_matrix.extend(a)
        b_matrix.extend(b)

def write_to_mtx(a_matrix, b_matrix, file_path, num_rows):

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Sort the list of dictionaries by y and then by x
    a_matrix = sorted(a_matrix, key=lambda coord: (coord['y'], coord['x']))

    # Iterate over the dictionaries in the list
    result_dict = defaultdict(float)
    for d in a_matrix:
        # Use tuple (x, y) as a key for the dictionary and add the value to the current sum
        result_dict[(d['x'], d['y'])] += d['value']
    a_matrix = [{"x": x, "y": y, "value": v} for (x, y), v in result_dict.items()]
    Total_nonzero_elements = len(a_matrix)

    with open(file_path, 'w') as file:
        # Write the Matrix Market header
        file.write("%%MatrixMarket matrix coordinate real general\n")
        file.write(f"{num_rows + 1} {num_rows + 1} {Total_nonzero_elements}\n")

        # Write the coordinate values
        for i in a_matrix:
            x = i['x']
            y = i['y']
            value = i['value']
            file.write(f"{x} {y} {value}\n")

    # print the sparse image of a_matrix
    x = [item['x'] for item in a_matrix]
    y = [item['y'] for item in a_matrix]
    data = [item['value'] for item in a_matrix]

    # Create sparse matrix
    sparse_matrix = coo_matrix((data, (x, y)))
    plt.spy(sparse_matrix, markersize=1)
    # plt.savefig('sparse_matrix.png', dpi=600)
    # plt.show()

def run_circuit_generation(nodeofiterations, nodenumbers, connection_method='random', output_directory="random_circuit_matrixs", verbose=False):
    os.makedirs(output_directory, exist_ok=True)

    with Progress() as progress:
        task = progress.add_task("[green]Generating random circuits...", total=nodeofiterations * len(nodenumbers))

        for i in range(nodeofiterations):
            for nodenumber in nodenumbers:
                extra_connections = nodenumber // 4

                if verbose:
                    logger.info(
                        f"Generating circuit with {nodenumber} nodes and {extra_connections} extra connections using {connection_method} method.")

                Graph = generate_circuit(nodenumber, connection_method, extra_connections)
                assign_modules_to_nodes(Graph)
                numberOfRows = assign_start_index(Graph)

                if verbose:
                    logger.info(f"Number of rows for node {nodenumber}: {numberOfRows}")

                building_modules(Graph, numberOfRows)

                # Generate filenames
                base_filename = os.path.join(output_directory, f'random_circuit_{nodenumber}_{i + 1}')
                mtx_filename = f'{base_filename}.mtx'
                png_filename = f'{base_filename}.png'

                # Write the .mtx file
                write_to_mtx(a_matrix, b_matrix, mtx_filename, numberOfRows)

                # Generate and save the graph visualization
                visualize_graph(Graph, png_filename)

                numberOfRows = 0
                a_matrix.clear()
                b_matrix.clear()

                progress.advance(task)

                if verbose:
                    logger.info(f'Completed {nodenumber} nodes for iteration {i + 1}/{nodeofiterations}')

if __name__ == "__main__":
    # Parameters
    nodeofiterations = 10
    nodenumbers = np.arange(5, 100, 5)
    nodenumbers = np.concatenate((nodenumbers, np.arange(100, 1000, 50)))

    # Run the circuit generation and matrix writing
    run_circuit_generation(
        nodeofiterations,
        nodenumbers,
        connection_method='scale_free',  # Options: 'random', 'fully_connected', 'scale_free', 'small_world'
        verbose=False
    )