from lib.random_circuit_generator.utils.device import *

class Second_order_RC:
    number_of_internalNodes = 2
    number_of_inputs = 1
    number_of_outputs = 1
    is_voltage_node = False

    def __init__(self, id):
        self.id = id
        self.start_index = None  # the start index to assign the internal nodes
        self.input_nodes = []
        self.output_nodes = []
        self.a_matrix = []
        self.b_matrix = []
        self.node_2 = None
        self.node_1 = None

    def get_matrix(self):
        return [self.a_matrix, self.b_matrix]

    def assign_start_index(self, index):
        self.start_index = index
        self.node_1 = index + 1
        self.node_2 = index + 2
        self.output_nodes.append(self.node_2)

    def assign_input_node(self, node):
        self.input_nodes.append(node)

    def build_matrix(self):
        componentList = []
        input_node = self.input_nodes[0]
        resistor_1 = Resistor(node_1=input_node, node_2=self.node_1)
        resistor_2 = Resistor(node_1=self.node_1, node_2=self.node_2)
        capacitor_1 = Capacitor(node_1=self.node_1, node_2=0)
        capacitor_2 = Capacitor(node_1=self.node_2, node_2=0)
        componentList.extend([resistor_1, resistor_2, capacitor_1, capacitor_2])

        # Build a matrix and b matrix
        for component in componentList:
            self.a_matrix += component.assign_matrix()[0]
            self.b_matrix += component.assign_matrix()[1]

        return [self.a_matrix, self.b_matrix]


class Forth_order_RC:
    number_of_internalNodes = 4
    number_of_inputs = 1
    number_of_outputs = 2
    is_voltage_node = False

    def __init__(self, id):
        self.id = id
        self.start_index = None  # the start index to assign the internal nodes
        self.input_nodes = []
        self.output_nodes = []
        self.a_matrix = []
        self.b_matrix = []
        self.node_4 = None
        self.node_3 = None
        self.node_2 = None
        self.node_1 = None

    def get_matrix(self):
        return [self.a_matrix, self.b_matrix]

    def assign_start_index(self, index):
        self.start_index = index
        self.node_1 = index + 1
        self.node_2 = index + 2
        self.node_3 = index + 3
        self.node_4 = index + 4
        self.output_nodes.append(self.node_2)
        self.output_nodes.append(self.node_4)

    def assign_input_node(self, node):
        self.input_nodes.append(node)

    def build_matrix(self):
        componentList = []
        input_node = self.input_nodes[0]
        resistor_1 = Resistor(node_1=input_node, node_2=self.node_1)
        resistor_2 = Resistor(node_1=self.node_1, node_2=self.node_2)
        resistor_3 = Resistor(node_1=self.node_2, node_2=self.node_3)
        resistor_4 = Resistor(node_1=self.node_3, node_2=self.node_4)
        capacitor_1 = Capacitor(node_1=self.node_1, node_2=0)
        capacitor_2 = Capacitor(node_1=self.node_2, node_2=0)
        capacitor_3 = Capacitor(node_1=self.node_3, node_2=0)
        capacitor_4 = Capacitor(node_1=self.node_4, node_2=0)
        componentList.extend(
            [resistor_1, resistor_2, resistor_3, resistor_4, capacitor_1, capacitor_2, capacitor_3, capacitor_4])

        # Build a matrix and b matrix
        for component in componentList:
            self.a_matrix += component.assign_matrix()[0]
            self.b_matrix += component.assign_matrix()[1]

        return [self.a_matrix, self.b_matrix]


class DifferentialAmp:
    # The number of internal nodes is 4, minus the voltage node and the voltage source internal node
    number_of_internalNodes = 4 + 4 * 2 - 1
    number_of_inputs = 2
    number_of_outputs = 2
    is_voltage_node = True

    def __init__(self, id):
        self.id = id
        self.start_index = None  # the start index to assign the internal nodes
        self.input_nodes = []
        self.output_nodes = []
        self.a_matrix = []
        self.b_matrix = []

        self.node_4 = None
        self.node_3 = None  # Voltage nodes
        self.node_2 = None
        self.node_1 = None

        self.voltage_node = None

    def assign_start_index(self, index):
        self.start_index = index
        self.node_1 = index + 1
        self.node_2 = index + 2
        # self.node_3 = index + 3
        self.node_4 = index + 3
        self.output_nodes.append(self.node_2)
        self.output_nodes.append(self.node_4)

    def assign_voltage_node(self, node):
        self.voltage_node = node
        self.node_3 = self.voltage_node

    def assign_input_node(self, node):
        self.input_nodes.append(node)

    def build_matrix(self):
        componentList = []
        input_node_1 = self.input_nodes[0]
        input_node_2 = self.input_nodes[1]
        NMOS_1 = NMOS(node_g=input_node_1, node_s=self.node_1, node_d=self.node_2)
        NMOS_2 = NMOS(node_g=input_node_2, node_s=self.node_1, node_d=self.node_4)
        Resistor_1 = Resistor(node_1=self.node_2, node_2=self.node_3)
        Resistor_2 = Resistor(node_1=self.node_4, node_2=self.node_3)
        Iss = Is(node_1=self.node_3, node_2=0)

        # start to assign the internal index for the devices with internal nodes
        internal_index = self.node_4
        internal_index = NMOS_1.assign_index(internal_index)
        internal_index = NMOS_2.assign_index(internal_index)

        componentList.extend(
            [NMOS_1, NMOS_2, Resistor_1, Resistor_2, Iss])

        # Build a matrix and b matrix
        for component in componentList:
            self.a_matrix += component.assign_matrix()[0]
            self.b_matrix += component.assign_matrix()[1]

        return [self.a_matrix, self.b_matrix]


class CommonSourceAmp:
    number_of_internalNodes = 2 + 4 * 1 - 1
    number_of_inputs = 1
    number_of_outputs = 1
    is_voltage_node = True

    def __init__(self, id):
        self.id = id
        self.start_index = None  # the start index to assign the internal nodes
        self.input_nodes = []
        self.output_nodes = []
        self.a_matrix = []
        self.b_matrix = []

        self.node_2 = None  # Voltage nodes
        self.node_1 = None

        self.voltage_node = None

    def assign_start_index(self, index):
        self.start_index = index
        self.node_1 = index + 1
        # self.node_2 = index + 2   # voltage node
        self.output_nodes.append(self.node_1)

    def assign_voltage_node(self, node):
        self.voltage_node = node
        self.node_2 = node

    def assign_input_node(self, node):
        self.input_nodes.append(node)

    def build_matrix(self):
        componentList = []
        input_node_1 = self.input_nodes[0]
        NMOS_1 = NMOS(node_g=input_node_1, node_s=0, node_d=self.node_1)
        Rd = Resistor(node_1=self.node_1, node_2=self.node_2)

        # start to assign the internal index for the devices with internal nodes
        internal_index = self.node_1
        internal_index = NMOS_1.assign_index(internal_index)

        componentList.extend(
            [NMOS_1, Rd])

        # Build a matrix and b matrix
        for component in componentList:
            self.a_matrix += component.assign_matrix()[0]
            self.b_matrix += component.assign_matrix()[1]

        return [self.a_matrix, self.b_matrix]


class CurrentMirror:
    number_of_internalNodes = 3 + 4 * 4
    number_of_inputs = 1
    number_of_outputs = 1
    is_voltage_node = False

    def __init__(self, id):
        self.id = id
        self.start_index = None  # the start index to assign the internal nodes
        self.input_nodes = []
        self.output_nodes = []
        self.a_matrix = []
        self.b_matrix = []
        self.node_3 = None
        self.node_2 = None
        self.node_1 = None

    def assign_start_index(self, index):
        self.start_index = index
        self.node_1 = index + 1
        self.node_2 = index + 2
        self.node_3 = index + 3
        self.output_nodes.append(self.node_2)

    def assign_input_node(self, node):
        self.input_nodes.append(node)

    def build_matrix(self):
        componentList = []
        input_node_1 = self.input_nodes[0]
        NMOS_1 = NMOS(node_g=input_node_1, node_s=self.node_1, node_d=input_node_1)
        NMOS_2 = NMOS(node_g=input_node_1, node_s=self.node_3, node_d=self.node_2)
        NMOS_3 = NMOS(node_g=input_node_1, node_s=0, node_d=self.node_1)
        NMOS_4 = NMOS(node_g=input_node_1, node_s=0, node_d=self.node_3)

        # start to assign the internal index for the devices with internal nodes
        internal_index = self.node_3
        internal_index = NMOS_1.assign_index(internal_index)
        internal_index = NMOS_2.assign_index(internal_index)
        internal_index = NMOS_3.assign_index(internal_index)
        internal_index = NMOS_4.assign_index(internal_index)

        componentList.extend(
            [NMOS_1, NMOS_2, NMOS_3, NMOS_4])

        # Build a matrix and b matrix
        for component in componentList:
            self.a_matrix += component.assign_matrix()[0]
            self.b_matrix += component.assign_matrix()[1]

        return [self.a_matrix, self.b_matrix]


class InstrumentalAmp:
    number_of_internalNodes = 7 + 3
    number_of_inputs = 2
    number_of_outputs = 1
    is_voltage_node = False

    def __init__(self, id):
        self.id = id
        self.start_index = None
        self.input_nodes = []
        self.output_nodes = []
        self.a_matrix = []
        self.b_matrix = []
        self.node_7 = None
        self.node_6 = None
        self.node_5 = None
        self.node_4 = None
        self.node_3 = None
        self.node_2 = None
        self.node_1 = None

    def assign_start_index(self, index):
        self.start_index = index
        self.node_1 = index + 1
        self.node_2 = index + 2
        self.node_3 = index + 3
        self.node_4 = index + 4
        self.node_5 = index + 5
        self.node_6 = index + 6
        self.node_7 = index + 7
        self.output_nodes.append(self.node_7)

    def assign_input_node(self, node):
        self.input_nodes.append(node)

    def build_matrix(self):
        componentList = []
        input_node_1 = self.input_nodes[0]
        input_node_2 = self.input_nodes[1]

        opamp_1 = OpAmp(node_1=input_node_1, node_2=self.node_1, node_3=self.node_2)
        opamp_2 = OpAmp(node_1=input_node_2, node_2=self.node_3, node_3=self.node_4)
        opamp_3 = OpAmp(node_1=self.node_5, node_2=self.node_6, node_3=self.node_7)
        R1_1 = Resistor(node_1=self.node_2, node_2=self.node_1)
        R1_2 = Resistor(node_1=self.node_4, node_2=self.node_3)
        Rg = Resistor(node_1=self.node_1, node_2=self.node_3)
        R2_1 = Resistor(node_1=self.node_2, node_2=self.node_6)
        R2_2 = Resistor(node_1=self.node_4, node_2=self.node_5)
        R3_1 = Resistor(node_1=self.node_6, node_2=self.node_7)
        R3_2 = Resistor(node_1=self.node_5, node_2=0)

        # start to assign the internal index for the devices with internal nodes
        internal_index = self.node_7
        internal_index = opamp_1.assign_index(internal_index)
        internal_index = opamp_2.assign_index(internal_index)
        internal_index = opamp_3.assign_index(internal_index)

        componentList.extend(
            [opamp_1, opamp_2, opamp_3, R1_1, R1_2, Rg, R2_1, R2_2, R3_1, R3_2])

        # Build a matrix and b matrix
        for component in componentList:
            self.a_matrix += component.assign_matrix()[0]
            self.b_matrix += component.assign_matrix()[1]

        return [self.a_matrix, self.b_matrix]


class Inverter:
    number_of_internalNodes = 2 + 4 * 2 - 1
    number_of_inputs = 1
    number_of_outputs = 1
    is_voltage_node = True

    def __init__(self, id):
        self.id = id
        self.start_index = None
        self.input_nodes = []
        self.output_nodes = []
        self.a_matrix = []
        self.b_matrix = []
        self.node_2 = None
        self.node_1 = None  # Voltage node

        self.voltage_node = None

    def assign_start_index(self, index):
        self.start_index = index
        # self.node_1 = index + 1  # Voltage node
        self.node_2 = index + 1
        self.output_nodes.append(self.node_2)

    def assign_input_node(self, node):
        self.input_nodes.append(node)

    def assign_voltage_node(self, node):
        self.voltage_node = node
        self.node_1 = node

    def build_matrix(self):
        componentList = []
        input_node_1 = self.input_nodes[0]

        NMOS_1 = NMOS(node_g=input_node_1, node_s=0, node_d=self.node_2)
        PMOS_1 = NMOS(node_g=input_node_1, node_s=self.node_1, node_d=self.node_2)
        # Vdd = Vs(node_1=self.node_1, node_2=0)

        # start to assign the internal index for the devices with internal nodes
        internal_index = self.node_2
        internal_index = NMOS_1.assign_index(internal_index)
        internal_index = PMOS_1.assign_index(internal_index)
        # internal_index = Vdd.assign_index(internal_index)

        componentList.extend(
            [NMOS_1, PMOS_1])

        # Build a matrix and b matrix
        for component in componentList:
            self.a_matrix += component.assign_matrix()[0]
            self.b_matrix += component.assign_matrix()[1]

        return [self.a_matrix, self.b_matrix]