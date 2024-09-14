class MyNode:  # Rename Node class to MyNode
    def __init__(self, id):
        self.id = id
        self.predecessors = []  # list to store predecessor nodes
        self.successors = []  # list to store successor nodes
        self.a_matrix = []
        self.b_matrix = []

        self.start_index = None  # the start index to assign the internal nodes
        self.chosen_module = None  # the class that is chosen to represent this node
        self.my_module = None  # the class that is assigned to this node
        self.number_of_internalNodes = None

    def assign_start_index(self, index):
        self.start_index = index

    def assign_chosen_module(self, chosen_class):
        self.chosen_module = chosen_class
        self.number_of_internalNodes = chosen_class.number_of_internalNodes

    # Create an Instance according to the chosen class
    def create_modules(self):
        self.my_module = self.chosen_module(self.id)
        self.my_module.assign_start_index(self.start_index)

    # Cycle through the current module's output nodes and assign them as input nodes to the successor nodes
    def assign_nodes(self):
        for index, successor in enumerate(self.successors):
            successor.my_module.assign_input_node(self.my_module.output_nodes[index % len(self.my_module.output_nodes)])

    def get_matrix(self):
        return [self.a_matrix, self.b_matrix]

    def build_matrix(self):
        [self.a_matrix, self.b_matrix] = self.my_module.build_matrix()
        return [self.a_matrix, self.b_matrix]









