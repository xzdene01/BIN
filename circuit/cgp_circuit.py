import re
import torch

class Node:
    """
    Represents a node in the CGP circuit.
    Each node has an ID, two inputs and an operation code, which can be than mapped to a logical operation.
    """
    def __init__(self, core):
        """
        Initialize a new node from the core string.
        :param core: The core string from the CGP circuit
        """
        self.id = core[0]
        
        splitted = core[1].split(",")
        self.in_1 = int(splitted[0])
        self.in_2 = int(splitted[1])
        self.op_code = int(splitted[2])
    
    def __str__(self):
        """
        Return a string representation of the node for printing.
        :return: The string representation of the node
        """
        return f"ID {self.id}: {self.in_1}, {op_code_mapping[self.op_code]}, {self.in_2}"

class CGPCircuit:
    """
    Represents a CGP circuit.
    The circuit consists of a prefix, core and outputs.
    The prefix is a dictionary containing the following keys:
    - c_in:     Number of inputs
    - c_out:    Number of outputs
    - c_rows:   Number of rows
    - c_cols:   Number of columns
    - c_ni:     Number of inputs to each node
    - c_no:     Number of outputs from each node
    - c_lback:  Number of columns back
    The core is a list of nodes, where each node is represented by a Node object.
    The outputs is a list of integers representing the output.
    """
    def __init__(self, cgp_string: str = None, file: str = None):
        """
        Initialize a new CGP circuit.
        :param file: The file path to load the CGP circuit from
        """
        self.prefix = {}
        self.core = []
        self.outputs = []

        if cgp_string:
            self.load_from_string(cgp_string)
        elif file:
            self.load_from_file(file)
    
    def forward(self, inputs: torch.Tensor, device: str = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        if inputs.dtype != torch.bool:
            raise ValueError("Input tensor must be of type bool")
        
        if inputs.shape[0] != self.prefix["c_in"]:
            raise ValueError(f"Number of inputs ({inputs.shape[0]}) does not match the number of inputs in the CGP circuit ({self.prefix["c_in"]})")

        # Initialize the node values
        node_values = torch.zeros(len(self.core), dtype=torch.bool)

        # Move tensors to device
        inputs = inputs.to(device)
        node_values = node_values.to(device)

        return

        # Set the input values
        for i in range(self.prefix["c_in"]):
            node_values[i] = inputs[i]

        # Evaluate the nodes
        for i in range(self.prefix["c_in"], len(self.core)):
            node = self.core[i]
            in_1 = node_values[node.in_1]
            in_2 = node_values[node.in_2]
            op_code = node.op_code

            if op_code == 0:
                node_values[i] = in_1
            elif op_code == 1:
                node_values[i] = ~in_1
            elif op_code == 2:
                node_values[i] = in_1 & in_2
            elif op_code == 3:
                node_values[i] = in_1 | in_2
            elif op_code == 4:
                node_values[i] = in_1 ^ in_2
            elif op_code == 5:
                node_values[i] = ~(in_1 & in_2)
            elif op_code == 6:
                node_values[i] = ~(in_1 | in_2)
            elif op_code == 7:
                node_values[i] = ~(in_1 ^ in_2)
            elif op_code == 8:
                node_values[i] = True
            elif op_code == 9:
                node_values[i] = False

        pass
    
    def load_from_string(self, cgp_string: str):
        """
        Load the CGP circuit from a string.
        Parsing of the input string was heavily inspired by the ArithsGen library (https://github.com/ehw-fit/ariths-gen/tree/main).
        :param cgp_string: The string to load the CGP circuit
        """
        cgp_prefix, cgp_core, cgp_outputs = re.match(r"{(.*)}(.*)\(([^()]+)\)", cgp_string).groups()

        # Process prefix
        c_in, c_out, c_rows, c_cols, c_ni, c_no, c_lback = map(int, cgp_prefix.split(","))

        if c_ni != 2:
            raise ValueError("Only 2 inputs (per node) are supported")
        if c_no != 1:
            raise ValueError("Only 1 output (per node) is supported")
        
        self.prefix = {
            "c_in": c_in,
            "c_out": c_out,
            "c_rows": c_rows,
            "c_cols": c_cols,
            "c_ni": c_ni,
            "c_no": c_no,
            "c_lback": c_lback
        }

        # Process core/nodes
        core = re.findall(r"\[([^\]]+)\](\d+,\d+,\d+)", cgp_core)
        for x in core:
            node = Node(x)
            self.core.append(node)
        
        # Process outputs
        self.outputs = list(map(int, cgp_outputs.split(",")))

    def load_from_file(self, file_path: str):
        """
        Load the CGP circuit from a file.
        :param file_path: The file path to load the CGP circuit
        """
        with open(file_path, "r") as file:
            file_content = file.read()
        
        self.load_from_string(file_content)

    def save_to_file(self, file_path):
        """
        Save the CGP circuit to a file.
        :param file_path: The file path to save the CGP circuit
        """
        prefix_str = "{" + f"{self.prefix["c_in"]},{self.prefix["c_out"]},{self.prefix["c_rows"]},{self.prefix["c_cols"]},{self.prefix["c_ni"]},{self.prefix["c_no"]},{self.prefix["c_lback"]}" + "}"
        core_str = ""
        for node in self.core:
            core_str += f"([{node.id}]{node.in_1},{node.in_2},{node.op_code})"
        outputs_str = ",".join(map(str, self.outputs))

        with open(file_path, "w") as file:
            file.write(f"{prefix_str}{core_str}({outputs_str})")
    
    def __str__(self):
        """
        Return a string representation of the CGP circuit for printing.
        :return: The string representation of the CGP circuit
        """
        nodes_str = "  " + "\n  ".join(str(node) for node in self.core)
        return f"Prefix: {self.prefix}\nCore:\n{nodes_str}\nOutputs: {self.outputs}"

# Mapping of operation codes to logical operations
op_code_mapping = {
    0: "IDENTITY",
    1: "NOT",
    2: "AND",
    3: "OR",
    4: "XOR",
    5: "NAND",
    6: "NOR",
    7: "XNOR",
    8: "TRUE",
    9: "FALSE"
}