import math
import random


class Is:
    numberofNode = 1

    def __init__(self, node_1, node_2, current=0):
        self.node_1 = node_1
        self.node_2 = node_2
        self.a_matrix = []
        self.b_matrix = []
        if current == 0:
            self.current = random.uniform(1e-9, 5)  # randomly generate a current between 1nA to 5A
        else:
            self.current = current

    def assign_matrix(self):
        # if node_1 or node_2 is 0, then assign the non-zero node to the a matrix and b matrix
        if self.node_1 == 0:
            self.b_matrix.append({"x": self.node_2, "value": -self.current})
        elif self.node_2 == 0:
            self.b_matrix.append({"x": self.node_1, "value": -self.current})
        else:
            self.b_matrix.append({"x": self.node_1, "value": -self.current})
            self.b_matrix.append({"x": self.node_2, "value": self.current})

        return [self.a_matrix, self.b_matrix]


class Resistor:
    numberofNode = 1

    def __init__(self, node_1, node_2, resistance=0):
        self.node_1 = node_1
        self.node_2 = node_2
        self.a_matrix = []
        self.b_matrix = []
        if resistance == 0:
            self.resistance = random.uniform(0.1, 10000)
        else:
            self.resistance = resistance

    def assign_matrix(self):
        # if node_1 or node_2 is 0, then assign the non-zero node to the a matrix and b matrix
        if self.node_1 == 0:
            self.a_matrix.append({"x": self.node_2, "y": self.node_2, "value": 1 / self.resistance})
        elif self.node_2 == 0:
            self.a_matrix.append({"x": self.node_1, "y": self.node_1, "value": 1 / self.resistance})
        else:
            self.a_matrix.append({"x": self.node_1, "y": self.node_1, "value": 1 / self.resistance})
            self.a_matrix.append({"x": self.node_2, "y": self.node_2, "value": 1 / self.resistance})
            self.a_matrix.append({"x": self.node_1, "y": self.node_2, "value": -1 / self.resistance})
            self.a_matrix.append({"x": self.node_2, "y": self.node_1, "value": -1 / self.resistance})
        return [self.a_matrix, self.b_matrix]


class Capacitor:
    numberofNode = 1

    def __init__(self, node_1, node_2, capacitance=0, Ieq=0):
        self.node_1 = node_1
        self.node_2 = node_2
        self.a_matrix = []
        self.b_matrix = []
        if capacitance == 0:
            self.capacitance = random.uniform(1, 100) * 1e-6
        else:
            self.capacitance = capacitance

        if Ieq == 0:
            self.Ieq = random.uniform(1, 100) * 1e-6
        else:
            self.Ieq = Ieq

        self.Is = Is(self.node_1, self.node_2, current=self.Ieq)

    def assign_matrix(self):
        # if node_1 or node_2 is 0, then assign the non-zero node to the a matrix and b matrix
        if self.node_1 == 0:
            self.a_matrix.append({"x": self.node_2, "y": self.node_2, "value": self.capacitance})
        elif self.node_2 == 0:
            self.a_matrix.append({"x": self.node_1, "y": self.node_1, "value": self.capacitance})
        else:
            self.a_matrix.append({"x": self.node_1, "y": self.node_1, "value": self.capacitance})
            self.a_matrix.append({"x": self.node_2, "y": self.node_2, "value": self.capacitance})
            self.a_matrix.append({"x": self.node_1, "y": self.node_2, "value": -self.capacitance})
            self.a_matrix.append({"x": self.node_2, "y": self.node_1, "value": -self.capacitance})

        # assign a matrix and b matrix from the Is class
        self.a_matrix += self.Is.assign_matrix()[0]
        self.b_matrix += self.Is.assign_matrix()[1]

        return [self.a_matrix, self.b_matrix]


class Diode:
    numberofNode = 1

    def __init__(self, node_1, node_2):
        self.node_1 = node_1
        self.node_2 = node_2
        self.a_matrix = []
        self.b_matrix = []
        self.Is = Is(self.node_1, self.node_2)
        self.gep = Resistor(self.node_1, self.node_2)

    def assign_matrix(self):
        self.a_matrix += self.Is.assign_matrix()[0]
        self.b_matrix += self.Is.assign_matrix()[1]
        self.a_matrix += self.gep.assign_matrix()[0]
        self.b_matrix += self.gep.assign_matrix()[1]

        return [self.a_matrix, self.b_matrix]


class Vs:
    numberofNode = 1

    def __init__(self, node_1, node_2, voltage=0, internal_index=0):
        self.node_1 = node_1
        self.node_2 = node_2
        self.a_matrix = []
        self.b_matrix = []
        if voltage == 0:
            self.voltage = random.uniform(1e-6, 100)  # randomly generate a voltage between 1uV to 100V
        else:
            self.voltage = voltage
        self.internal_index = internal_index

    def assign_index(self, internal_index):
        self.internal_index = internal_index
        return internal_index + self.numberofNode

    def assign_matrix(self):
        # if node_1 or node_2 is 0, then assign the non-zero node to a matrix and b matrix
        if self.node_1 == 0:
            self.b_matrix.append({"x": self.internal_index + 1, "value": self.voltage})
            self.a_matrix.append({"x": self.internal_index + 1, "y": self.node_2, "value": 1})
            self.a_matrix.append({"x": self.node_2, "y": self.internal_index + 1, "value": 1})
        elif self.node_2 == 0:
            self.b_matrix.append({"x": self.internal_index + 1, "value": self.voltage})
            self.a_matrix.append({"x": self.internal_index + 1, "y": self.node_1, "value": 1})
            self.a_matrix.append({"x": self.node_1, "y": self.internal_index + 1, "value": 1})
        else:
            self.b_matrix.append({"x": self.internal_index + 1, "value": self.voltage})
            self.a_matrix.append({"x": self.internal_index + 1, "y": self.node_1, "value": 1})
            self.a_matrix.append({"x": self.node_1, "y": self.internal_index + 1, "value": 1})
            self.a_matrix.append({"x": self.internal_index + 1, "y": self.node_2, "value": -1})
            self.a_matrix.append({"x": self.node_2, "y": self.internal_index + 1, "value": -1})

        return [self.a_matrix, self.b_matrix]


class VCCS:

    def __init__(self, node_1, node_2, node_3, node_4, gm=0):
        self.node_1 = node_1
        self.node_2 = node_2
        self.node_3 = node_3
        self.node_4 = node_4
        self.a_matrix = []
        self.b_matrix = []
        if gm == 0:
            self.gm = random.uniform(1e-3, 100e-3)  # randomly generate a gm between 1 to 100 millisiemens

    def assign_matrix(self):
        if self.node_1 == 0:
            if self.node_3 == 0:
                if self.node_4 != 0:
                    self.a_matrix.append({"x": self.node_2, "y": self.node_4, "value": self.gm})
            elif self.node_4 == 0:
                if self.node_3 != 0:
                    self.a_matrix.append({"x": self.node_2, "y": self.node_3, "value": -self.gm})
            else:
                self.a_matrix.append({"x": self.node_2, "y": self.node_3, "value": -self.gm})
                self.a_matrix.append({"x": self.node_2, "y": self.node_4, "value": self.gm})
        elif self.node_2 == 0:
            if self.node_3 == 0:
                if self.node_4 != 0:
                    self.a_matrix.append({"x": self.node_1, "y": self.node_4, "value": -self.gm})
            elif self.node_4 == 0:
                if self.node_3 != 0:
                    self.a_matrix.append({"x": self.node_1, "y": self.node_3, "value": self.gm})
            else:
                self.a_matrix.append({"x": self.node_1, "y": self.node_3, "value": self.gm})
                self.a_matrix.append({"x": self.node_1, "y": self.node_4, "value": -self.gm})
        else:
            if self.node_3 == 0:
                if self.node_4 != 0:
                    self.a_matrix.append({"x": self.node_1, "y": self.node_4, "value": -self.gm})
                    self.a_matrix.append({"x": self.node_2, "y": self.node_4, "value": self.gm})
            elif self.node_4 == 0:
                if self.node_3 != 0:
                    self.a_matrix.append({"x": self.node_1, "y": self.node_3, "value": self.gm})
                    self.a_matrix.append({"x": self.node_2, "y": self.node_3, "value": -self.gm})
            else:
                self.a_matrix.append({"x": self.node_1, "y": self.node_3, "value": self.gm})
                self.a_matrix.append({"x": self.node_2, "y": self.node_3, "value": -self.gm})
                self.a_matrix.append({"x": self.node_1, "y": self.node_4, "value": -self.gm})
                self.a_matrix.append({"x": self.node_2, "y": self.node_4, "value": self.gm})

        return [self.a_matrix, self.b_matrix]


class PMOS:
    # Create 4 internal nodes
    numberofNode = 4

    def __init__(self, node_g, node_s, node_d, node_b=1, internal_index=0):
        self.node_g = node_g
        self.node_s = node_s
        self.node_d = node_d
        if node_b == 1:
            self.node_b = node_s
        else:
            self.node_b = node_b
        self.a_matrix = []
        self.b_matrix = []
        self.internal_index = internal_index

    def get_node_g(self):
        return self.node_g

    def get_node_s(self):
        return self.node_s

    def get_node_d(self):
        return self.node_d

    def assign_index(self, internal_index):
        self.internal_index = internal_index
        return internal_index + self.numberofNode

    def assign_matrix(self):
        component_list = []
        Rg = Resistor(self.node_g, self.internal_index + 2)
        Rd = Resistor(self.node_d, self.internal_index + 3)
        Rb = Resistor(self.node_b, self.internal_index + 4)
        Rs = Resistor(self.node_s, self.internal_index + 1)
        DiodeBD = Diode(self.internal_index + 4, self.internal_index + 3)
        DiodeBS = Diode(self.internal_index + 4, self.internal_index + 1)
        C_BD = Capacitor(self.internal_index + 4, self.internal_index + 3)
        C_BS = Capacitor(self.internal_index + 4, self.internal_index + 1)
        C_GD = Capacitor(self.internal_index + 2, self.internal_index + 3)
        C_GS = Capacitor(self.internal_index + 1, self.internal_index + 2)
        C_GB = Capacitor(self.internal_index + 2, self.internal_index + 4)

        Is_gmb = VCCS(self.node_s, self.node_d, self.node_s, self.node_b)
        Is_gm = VCCS(self.node_s, self.node_d, self.node_s, self.node_g)
        R_ds = Resistor(self.node_d, self.node_s)

        # add all the components to the component list
        component_list.extend([Rg, Rd, Rb, Rs, DiodeBD, DiodeBS, C_BD, C_BS, C_GD, C_GS, C_GB, Is_gmb, Is_gm, R_ds])

        # Assign a matrix and b matrix for each component
        for component in component_list:
            self.a_matrix += component.assign_matrix()[0]
            self.b_matrix += component.assign_matrix()[1]

        return [self.a_matrix, self.b_matrix]


class NMOS:
    # Create 4 internal nodes
    numberofNode = 4

    def __init__(self, node_g, node_s, node_d, node_b=1, internal_index=0):
        self.node_g = node_g
        self.node_s = node_s
        self.node_d = node_d
        if node_b == 1:
            self.node_b = node_s
        else:
            self.node_b = node_b
        self.a_matrix = []
        self.b_matrix = []
        self.internal_index = internal_index

    def get_node_g(self):
        return self.node_g

    def get_node_s(self):
        return self.node_s

    def get_node_d(self):
        return self.node_d

    def assign_index(self, internal_index):
        self.internal_index = internal_index
        return internal_index + self.numberofNode

    def assign_matrix(self):
        component_list = []
        Rg = Resistor(self.node_g, self.internal_index + 2)
        Rd = Resistor(self.node_d, self.internal_index + 3)
        Rb = Resistor(self.node_b, self.internal_index + 4)
        Rs = Resistor(self.node_s, self.internal_index + 1)
        DiodeBD = Diode(self.internal_index + 4, self.internal_index + 3)
        DiodeBS = Diode(self.internal_index + 4, self.internal_index + 1)
        C_BD = Capacitor(self.internal_index + 4, self.internal_index + 3)
        C_BS = Capacitor(self.internal_index + 4, self.internal_index + 1)
        C_GD = Capacitor(self.internal_index + 2, self.internal_index + 3)
        C_GS = Capacitor(self.internal_index + 1, self.internal_index + 2)
        C_GB = Capacitor(self.internal_index + 2, self.internal_index + 4)

        Is_gmb = VCCS(self.node_s, self.node_d, self.node_s, self.node_b)
        Is_gm = VCCS(self.node_s, self.node_d, self.node_s, self.node_g)
        R_ds = Resistor(self.node_d, self.node_s)

        # add all the components to the component list
        component_list.extend([Rg, Rd, Rb, Rs, DiodeBD, DiodeBS, C_BD, C_BS, C_GD, C_GS, C_GB, Is_gmb, Is_gm, R_ds])

        # Assign a matrix and b matrix for each component
        for component in component_list:
            self.a_matrix += component.assign_matrix()[0]
            self.b_matrix += component.assign_matrix()[1]

        return [self.a_matrix, self.b_matrix]


# ideal op amp
class OpAmp:
    # Create 4 internal nodes
    numberofNode = 1

    def __init__(self, node_1, node_2, node_3, gain=0, internal_index=0):
        self.node_1 = node_1
        self.node_2 = node_2
        self.node_3 = node_3
        self.gain = gain
        self.a_matrix = []
        self.b_matrix = []
        self.internal_index = internal_index

        # randomly assign a gain between 100 to 1e6 if gain is not specified
        if gain == 0:
            self.gain = random.randint(100, 1000000)

    def assign_index(self, internal_index):
        self.internal_index = internal_index
        return internal_index + self.numberofNode

    def assign_matrix(self):
        self.a_matrix.append({"x": self.internal_index + 1, "y": self.node_3, "value": 1 / self.gain})

        if self.node_1 != 0:
            self.a_matrix.append({"x": self.internal_index + 1, "y": self.node_1, "value": -1})

        if self.node_2 != 0:
            self.a_matrix.append({"x": self.internal_index + 1, "y": self.node_2, "value": 1})

        self.a_matrix.append({"x": self.node_3, "y": self.internal_index + 1, "value": 1})

        return [self.a_matrix, self.b_matrix]
