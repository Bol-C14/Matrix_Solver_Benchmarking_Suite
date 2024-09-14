from lib.random_circuit_generator.utils.device import *

class NANDGate:
    number_of_internalNodes = 3 + 4 * 4 - 1
    number_of_inputs = 2
    number_of_outputs = 1
    is_voltage_node = True

    def __init__(self, id):
        self.id = id
        self.start_index = None
        self.input_nodes = []
        self.output_nodes = []
        self.a_matrix = []
        self.b_matrix = []
        self.node_3 = None
        self.node_2 = None
        self.node_1 = None  # Voltage node

        self.voltage_node = None

    def assign_start_index(self, index):
        self.start_index = index
        # self.node_1 = index + 1
        self.node_2 = index + 1
        self.node_3 = index + 2
        self.output_nodes.append(self.node_2)

    def assign_input_node(self, node):
        self.input_nodes.append(node)

    def assign_voltage_node(self, node):
        self.voltage_node = node
        self.node_1 = node

    def build_matrix(self):
        componentList = []
        input_node_1 = self.input_nodes[0]
        input_node_2 = self.input_nodes[1]

        NMOS_1 = NMOS(node_g=input_node_1, node_s=self.node_3, node_d=self.node_2)
        NMOS_2 = NMOS(node_g=input_node_2, node_s=0, node_d=self.node_3)
        PMOS_1 = NMOS(node_g=input_node_1, node_s=self.node_1, node_d=self.node_2)
        PMOS_2 = NMOS(node_g=input_node_2, node_s=self.node_1, node_d=self.node_2)
        # Vdd = Vs(node_1=self.node_1, node_2=0)

        # start to assign the internal index for the devices with internal nodes
        internal_index = self.node_3
        internal_index = NMOS_1.assign_index(internal_index)
        internal_index = NMOS_2.assign_index(internal_index)
        internal_index = PMOS_1.assign_index(internal_index)
        internal_index = PMOS_2.assign_index(internal_index)
        # internal_index = Vdd.assign_index(internal_index)

        componentList.extend(
            [NMOS_1, NMOS_2, PMOS_1, PMOS_2])

        # Build a matrix and b matrix
        for component in componentList:
            self.a_matrix += component.assign_matrix()[0]
            self.b_matrix += component.assign_matrix()[1]

        return [self.a_matrix, self.b_matrix]


class ANDGate:
    number_of_internalNodes = 4 + 4 * 6 - 1
    number_of_inputs = 2
    number_of_outputs = 1
    is_voltage_node = True

    def __init__(self, id):
        self.id = id
        self.start_index = None
        self.input_nodes = []
        self.output_nodes = []
        self.a_matrix = []
        self.b_matrix = []
        self.node_4 = None
        self.node_3 = None
        self.node_2 = None
        self.node_1 = None  # Voltage node

        self.voltage_node = None

    def assign_start_index(self, index):
        self.start_index = index
        # self.node_1 = index + 1
        self.node_2 = index + 1
        self.node_3 = index + 2
        self.node_4 = index + 3
        self.output_nodes.append(self.node_4)

    def assign_input_node(self, node):
        self.input_nodes.append(node)

    def assign_voltage_node(self, node):
        self.voltage_node = node
        self.node_1 = node

    def build_matrix(self):
        componentList = []
        input_node_1 = self.input_nodes[0]
        input_node_2 = self.input_nodes[1]

        # # VDD
        # Vdd = Vs(node_1=self.node_1, node_2=0)

        # NAND gate
        NMOS_1 = NMOS(node_g=input_node_1, node_s=self.node_3, node_d=self.node_2)
        NMOS_2 = NMOS(node_g=input_node_2, node_s=0, node_d=self.node_3)
        PMOS_1 = NMOS(node_g=input_node_1, node_s=self.node_1, node_d=self.node_2)
        PMOS_2 = NMOS(node_g=input_node_2, node_s=self.node_1, node_d=self.node_2)

        # Inverter
        NMOS_3 = NMOS(node_g=self.node_2, node_s=0, node_d=self.node_4)
        PMOS_3 = NMOS(node_g=self.node_2, node_s=self.node_1, node_d=self.node_4)

        # start to assign the internal index for the devices with internal nodes
        internal_index = self.node_4
        internal_index = NMOS_1.assign_index(internal_index)
        internal_index = NMOS_2.assign_index(internal_index)
        internal_index = NMOS_3.assign_index(internal_index)
        internal_index = PMOS_1.assign_index(internal_index)
        internal_index = PMOS_2.assign_index(internal_index)
        internal_index = PMOS_3.assign_index(internal_index)
        # internal_index = Vdd.assign_index(internal_index)

        componentList.extend(
            [NMOS_1, NMOS_2, NMOS_3, PMOS_1, PMOS_2, PMOS_3])

        # Build a matrix and b matrix
        for component in componentList:
            self.a_matrix += component.assign_matrix()[0]
            self.b_matrix += component.assign_matrix()[1]

        return [self.a_matrix, self.b_matrix]


class NORGate:
    number_of_internalNodes = 3 + 4 * 4 - 1
    number_of_inputs = 2
    number_of_outputs = 1
    is_voltage_node = True

    def __init__(self, id):
        self.id = id
        self.start_index = None
        self.input_nodes = []
        self.output_nodes = []
        self.a_matrix = []
        self.b_matrix = []
        self.node_3 = None
        self.node_2 = None
        self.node_1 = None  # Voltage node

        self.voltage_node = None

    def assign_start_index(self, index):
        self.start_index = index
        # self.node_1 = index + 1
        self.node_2 = index + 1
        self.node_3 = index + 2

        self.output_nodes.append(self.node_3)

    def assign_input_node(self, node):
        self.input_nodes.append(node)

    def assign_voltage_node(self, node):
        self.voltage_node = node
        self.node_1 = node

    def build_matrix(self):
        componentList = []
        input_node_1 = self.input_nodes[0]
        input_node_2 = self.input_nodes[1]

        NMOS_1 = NMOS(node_g=input_node_1, node_s=0, node_d=self.node_3)
        NMOS_2 = NMOS(node_g=input_node_2, node_s=0, node_d=self.node_3)
        PMOS_1 = NMOS(node_g=input_node_1, node_s=self.node_1, node_d=self.node_2)
        PMOS_2 = NMOS(node_g=input_node_2, node_s=self.node_2, node_d=self.node_3)
        # Vdd = Vs(node_1=self.node_1, node_2=0)

        # start to assign the internal index for the devices with internal nodes
        internal_index = self.node_3
        internal_index = NMOS_1.assign_index(internal_index)
        internal_index = NMOS_2.assign_index(internal_index)
        internal_index = PMOS_1.assign_index(internal_index)
        internal_index = PMOS_2.assign_index(internal_index)
        # internal_index = Vdd.assign_index(internal_index)

        componentList.extend(
            [NMOS_1, NMOS_2, PMOS_1, PMOS_2])

        # Build a matrix and b matrix
        for component in componentList:
            self.a_matrix += component.assign_matrix()[0]
            self.b_matrix += component.assign_matrix()[1]

        return [self.a_matrix, self.b_matrix]


class ORGate:
    number_of_internalNodes = 4 + 4 * 6 - 1
    number_of_inputs = 2
    number_of_outputs = 1
    is_voltage_node = True

    def __init__(self, id):
        self.id = id
        self.start_index = None
        self.input_nodes = []
        self.output_nodes = []
        self.a_matrix = []
        self.b_matrix = []
        self.node_4 = None
        self.node_3 = None
        self.node_2 = None
        self.node_1 = None  # Voltage node

        self.voltage_node = None

    def assign_start_index(self, index):
        self.start_index = index
        # self.node_1 = index + 1
        self.node_2 = index + 1
        self.node_3 = index + 2
        self.node_4 = index + 3

        self.output_nodes.append(self.node_4)

    def assign_input_node(self, node):
        self.input_nodes.append(node)

    def assign_voltage_node(self, node):
        self.voltage_node = node
        self.node_1 = node

    def build_matrix(self):
        componentList = []
        input_node_1 = self.input_nodes[0]
        input_node_2 = self.input_nodes[1]

        # XOR gate
        NMOS_1 = NMOS(node_g=input_node_1, node_s=0, node_d=self.node_3)
        NMOS_2 = NMOS(node_g=input_node_2, node_s=0, node_d=self.node_3)
        PMOS_1 = NMOS(node_g=input_node_1, node_s=self.node_1, node_d=self.node_2)
        PMOS_2 = NMOS(node_g=input_node_2, node_s=self.node_2, node_d=self.node_3)

        # inverter
        NMOS_3 = NMOS(node_g=self.node_3, node_s=0, node_d=self.node_4)
        PMOS_3 = NMOS(node_g=self.node_3, node_s=self.node_1, node_d=self.node_4)
        # Vdd = Vs(node_1=self.node_1, node_2=0)

        # start to assign the internal index for the devices with internal nodes
        internal_index = self.node_4
        internal_index = NMOS_1.assign_index(internal_index)
        internal_index = NMOS_2.assign_index(internal_index)
        internal_index = NMOS_3.assign_index(internal_index)
        internal_index = PMOS_1.assign_index(internal_index)
        internal_index = PMOS_2.assign_index(internal_index)
        internal_index = PMOS_3.assign_index(internal_index)
        # internal_index = Vdd.assign_index(internal_index)

        componentList.extend(
            [NMOS_1, NMOS_2, NMOS_3, PMOS_1, PMOS_2, PMOS_3])

        # Build a matrix and b matrix
        for component in componentList:
            self.a_matrix += component.assign_matrix()[0]
            self.b_matrix += component.assign_matrix()[1]

        return [self.a_matrix, self.b_matrix]

class SRAM:
    number_of_internalNodes = 5 + 4 * 6 - 1
    number_of_inputs = 1
    number_of_outputs = 2
    is_voltage_node = True

    def __init__(self, id):
        self.id = id
        self.start_index = None
        self.input_nodes = []
        self.output_nodes = []
        self.a_matrix = []
        self.b_matrix = []
        self.node_5 = None  # Voltage node
        self.node_4 = None
        self.node_3 = None
        self.node_2 = None
        self.node_1 = None

        self.voltage_node = None

    def assign_start_index(self, index):
        self.start_index = index
        self.node_1 = index + 1
        self.node_2 = index + 2
        self.node_3 = index + 3
        self.node_4 = index + 4
        # self.node_5 = index + 5

        self.output_nodes.append(self.node_1)
        self.output_nodes.append(self.node_4)

    def assign_input_node(self, node):
        self.input_nodes.append(node)

    def assign_voltage_node(self, node):
        self.voltage_node = node
        self.node_5 = node

    def build_matrix(self):
        componentList = []
        input_node_1 = self.input_nodes[0]

        # Switch
        NMOS_1 = NMOS(node_g=input_node_1, node_s=self.node_1, node_d=self.node_2, node_b=0)
        NMOS_2 = NMOS(node_g=input_node_1, node_s=self.node_3, node_d=self.node_4, node_b=0)

        # inverter 1
        PMOS_3 = NMOS(node_g=self.node_3, node_s=self.node_5, node_d=self.node_2)
        NMOS_3 = NMOS(node_g=self.node_3, node_s=0, node_d=self.node_2)

        # inverter 2
        PMOS_4 = PMOS(node_g=self.node_2, node_s=self.node_5, node_d=self.node_3)
        NMOS_4 = NMOS(node_g=self.node_2, node_s=0, node_d=self.node_3)

        # start to assign the internal index for the devices with internal nodes
        internal_index = self.node_4
        internal_index = NMOS_1.assign_index(internal_index)
        internal_index = NMOS_2.assign_index(internal_index)
        internal_index = NMOS_3.assign_index(internal_index)
        internal_index = NMOS_4.assign_index(internal_index)
        internal_index = PMOS_3.assign_index(internal_index)
        internal_index = PMOS_4.assign_index(internal_index)

        componentList.extend(
            [NMOS_1, NMOS_2, NMOS_3, NMOS_4, PMOS_3, PMOS_4])

        # Build a matrix and b matrix
        for component in componentList:
            self.a_matrix += component.assign_matrix()[0]
            self.b_matrix += component.assign_matrix()[1]

        return [self.a_matrix, self.b_matrix]