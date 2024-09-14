import networkx as nx
import random
import matplotlib.pyplot as plt
import importlib
import inspect
import numpy as np
import os
import argparse
from MyNode import MyNode
from lib.random_circuit_generator.utils.device import Vs
from collections import defaultdict
from scipy.sparse import coo_matrix

# Load external circuit modules
circuit_module = importlib.import_module("utils.model")
circuit_module_analog = importlib.import_module("utils.AnalogModels")
circuit_module_digital = importlib.import_module("utils.DigitalModels")


def load_module_classes(module):
    """
    Loads classes from a given module.
    """
    return [obj for name, obj in inspect.getmembers(module) if
            inspect.isclass(obj) and obj.__module__ == module.__name__]


# Load classes from each module
target_model_library = load_module_classes(circuit_module)
modules_analog = load_module_classes(circuit_module_analog)
modules_digital = load_module_classes(circuit_module_digital)


def generate_circuit(num_nodes, extra_connections):
    """
    Generates a random circuit graph with the specified number of nodes and extra connections.
    """
    graph = nx.DiGraph()
    nodes = [MyNode(i) for i in range(num_nodes)]
    node_map = {node.id: node for node in nodes}

    start_node = random.choice(nodes)
    graph.add_node(start_node)
    unconnected_nodes = set(nodes) - {start_node}

    while unconnected_nodes:
        node = random.choice(list(unconnected_nodes))
        predecessor = random.choice(list(graph.nodes))

        if len(predecessor.successors) < 2 and len(node.predecessors) < 2:
            predecessor.successors.append(node)
            node.predecessors.append(predecessor)
            graph.add_edge(predecessor, node)
            unconnected_nodes.remove(node)

    for _ in range(extra_connections):
        node1 = random.choice(list(graph.nodes))
        node2 = random.choice(list(graph.nodes - {start_node}))
        while graph.in_degree(node2) >= 2:
            node2 = random.choice(list(graph.nodes - {start_node}))

        node1.successors.append(node2)
        node2.predecessors.append(node1)
        graph.add_edge(node1, node2)

    return graph


def visualize_graph(graph, output_file="node_graph.png"):
    """
    Visualizes the directed graph using matplotlib.
    """
    plt.figure(figsize=(10, 8))
    pos = nx.kamada_kawai_layout(graph)

    modules = set(node.chosen_module for node in graph.nodes())
    color_map = {module: plt.cm.jet(i / len(modules)) for i, module in enumerate(modules)}
    node_colors = [color_map[node.chosen_module] for node in graph.nodes()]

    nx.draw_networkx_nodes(graph, pos, node_size=400, alpha=0.85, node_color=node_colors)
    nx.draw_networkx_edges(graph, pos, width=2, edge_color="gray")
    labels = {node: node.id for node in graph.nodes}
    label_pos = {k: [v[0], v[1] + 0.05] for k, v in pos.items()}
    nx.draw_networkx_labels(graph, label_pos, labels, font_size=10)

    for module, color in color_map.items():
        plt.plot([], [], 'o', color=color, label=module)
    plt.legend()

    plt.title("Node Graph")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_file, dpi=600)
    plt.show()


def assign_modules_to_nodes(graph):
    """
    Assigns modules from the target model library to nodes based on their in-degree.
    """
    start_node = next(node for node in graph.nodes if graph.in_degree(node) == 0)

    for node in graph.nodes:
        possible_classes = [cls for cls in target_model_library if graph.in_degree(node) == cls.number_of_inputs]

        if node == start_node:
            possible_classes = target_model_library

        if possible_classes:
            chosen_class = random.choice(possible_classes)
            node.assign_chosen_module(chosen_class)
        else:
            print(f"Node {node.id} has no possible classes")
            exit(1)


def assign_voltage_sources(graph, num_nodes, a_matrix, b_matrix):
    """
    Assigns voltage sources to the graph and updates the global matrices.
    """
    Vdd = Vs(1, 0, 3.3, num_nodes)
    a_matrix, b_matrix = Vdd.assign_matrix()
    for node in graph.nodes:
        if node.my_module.is_voltage_node:
            node.my_module.assign_voltage_node(1)
    return a_matrix, b_matrix


def assign_start_index(graph):
    """
    Assigns start indexes to nodes using DFS traversal.
    """
    start_index = 1
    for node in nx.dfs_preorder_nodes(graph):
        if node.number_of_internalNodes is None:
            print(f"Node {node.id} has not been assigned a number_of_internalNodes value")
            continue
        node.assign_start_index(start_index)
        start_index += node.number_of_internalNodes
    return start_index


def build_modules(graph, num_nodes, a_matrix, b_matrix):
    """
    Builds modules for each node in the graph and assigns inputs and outputs.
    """
    for index, node in enumerate(graph.nodes):
        node.create_modules()

        if index == 0:
            for i in range(node.my_module.number_of_inputs):
                node.my_module.assign_input_node(0)

    a_matrix, b_matrix = assign_voltage_sources(graph, num_nodes, a_matrix, b_matrix)

    for node in nx.dfs_preorder_nodes(graph):
        node.assign_nodes()

    for node in graph.nodes:
        a, b = node.build_matrix()
        a_matrix.extend(a)
        b_matrix.extend(b)

    return a_matrix, b_matrix


def write_to_mtx(a_matrix, b_matrix, file_path, num_rows):
    """
    Write the a_matrix to a Matrix Market file and plot the sparse matrix.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    a_matrix = sorted(a_matrix, key=lambda coord: (coord['y'], coord['x']))
    result_dict = defaultdict(float)
    for d in a_matrix:
        result_dict[(d['x'], d['y'])] += d['value']

    a_matrix = [{"x": x, "y": y, "value": v} for (x, y), v in result_dict.items()]
    total_nonzero_elements = len(a_matrix)

    with open(file_path, 'w') as file:
        file.write("%%MatrixMarket matrix coordinate real general\n")
        file.write(f"{num_rows} {num_rows} {total_nonzero_elements}\n")
        for i in a_matrix:
            file.write(f"{i['x']} {i['y']} {i['value']}\n")

    x = [item['x'] for item in a_matrix]
    y = [item['y'] for item in a_matrix]
    data = [item['value'] for item in a_matrix]

    sparse_matrix = coo_matrix((data, (x, y)))
    plt.spy(sparse_matrix, markersize=1)


def run_circuit_generation(node_of_iterations=2, node_range_1=(5, 100, 5), node_range_2=(100, 2000, 20), output_dir="generated_circuits"):
    """
    Generates circuits and writes corresponding matrices to files.
    """
    a_matrix, b_matrix = [], []
    node_numbers = np.concatenate((np.arange(*node_range_1), np.arange(*node_range_2)))

    for i in range(node_of_iterations):
        for node_number in node_numbers:
            extra_connections = node_number // 4
            graph = generate_circuit(node_number, extra_connections)
            assign_modules_to_nodes(graph)
            num_rows = assign_start_index(graph)
            a_matrix, b_matrix = build_modules(graph, num_rows, a_matrix, b_matrix)

            filename = os.path.join(output_dir, f'random_circuit_{node_number}_{i + 1}.mtx')
            write_to_mtx(a_matrix, b_matrix, filename, num_rows)

            a_matrix.clear()
            b_matrix.clear()
            print(f"Generated matrices for {node_number} nodes")

    print("Circuit generation completed.")
    print(f"Total number of matrices generated: {node_of_iterations * len(node_numbers)}")


def main():
    parser = argparse.ArgumentParser(description="Random Circuit Matrix Generator")
    parser.add_argument('--iterations', type=int, default=2, help='Number of iterations for circuit generation')
    parser.add_argument('--node_range_1', type=int, nargs=3, default=[5, 100, 5], help='Node range 1 (start, end, step)')
    parser.add_argument('--node_range_2', type=int, nargs=3, default=[100, 2000, 20], help='Node range 2 (start, end, step)')
    parser.add_argument('--output_dir', type=str, default="generated_circuits", help='Output directory for matrix files')

    args = parser.parse_args()

    run_circuit_generation(
        node_of_iterations=args.iterations,
        node_range_1=args.node_range_1,
        node_range_2=args.node_range_2,
        output_dir=args.output_dir
    )


# Example usage
if __name__ == "__main__":
    main()
