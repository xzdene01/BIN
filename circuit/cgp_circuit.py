import re
import torch

from .mappings import opcode_to_str, opcode_to_func


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
        return f"ID {self.id}: {self.in_1}, {opcode_to_str[self.op_code]}, {self.in_2}"


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
        :param cgp_string: The string to load the CGP circuit from
        :param file: The file path to load the CGP circuit from
        """
        self.prefix = {}
        self.core = []
        self.outputs = []

        if cgp_string:
            self.load_from_string(cgp_string)
        elif file:
            self.load_from_file(file)

    def forward(self, inputs: torch.Tensor, device: str = "cpu"):
        return CGPCircuit.forward_static(inputs, self.prefix["c_in"], self.core, self.outputs, device)

    def forward_batch(self, inputs: torch.Tensor, device: str):
        c_in = self.prefix["c_in"]

        if inputs.dtype != torch.bool:
            raise ValueError("Input tensor must be of type bool")
        if inputs.shape[1] != c_in:
            raise ValueError(f"Number of inputs ({inputs.shape[0]}) does not match the number of inputs in the CGP circuit ({c_in})")
        if inputs.dim() != 2:
            raise ValueError("Expected inputs to be a 2D tensor of shape (batch_size, c_in)")

        total_length = 2 + c_in + len(self.core)
        batch_size = inputs.shape[0]

        values = torch.zeros((batch_size, total_length), dtype=torch.bool, device=device)

        # Set implicit 0 and 1 as constants
        values[:, 0] = False
        values[:, 1] = True

        # Load primary inputs starting from index 2
        values[:, 2:2 + c_in] = inputs

        for node in self.core:
            a = values[:, node.in_1]
            b = values[:, node.in_2]
            result = opcode_to_func[node.op_code](a, b)
            values[:, int(node.id)] = result

        output_indices = torch.tensor(self.outputs, dtype=torch.long, device=device)
        final_output = values[:, output_indices]
        return final_output
    
    @staticmethod
    def forward_static(inputs: torch.Tensor, c_in: int, cgp_core: list, cgp_out: list, device: str):
        if inputs.dtype != torch.bool:
            raise ValueError("Input tensor must be of type bool")
        if inputs.shape[0] != c_in:
            raise ValueError(f"Number of inputs ({inputs.shape[0]}) does not match the number of inputs in the CGP circuit ({c_in})")

        total_len = 2 + c_in + len(cgp_core)
        values = torch.empty(total_len, dtype=torch.bool, device=device)

        # Set implicit 0 and 1 as constants
        values[0] = False
        values[1] = True

        # Load primary inputs starting from index 2
        values[2:2 + c_in] = inputs

        for node in cgp_core:
            in_1 = values[node.in_1]
            in_2 = values[node.in_2]
            result = opcode_to_func[node.op_code](in_1, in_2)
            values[int(node.id)] = result

        final_output = values[torch.tensor(cgp_out, dtype=torch.long, device=device)]
        return final_output

    def get_active_mask(self):
        mask = torch.zeros(len(self.core))
        for node in self.core:
            mask[node.in_1 - self.prefix["c_in"] - 2] = True
            mask[node.in_2 - self.prefix["c_in"] - 2] = True

        for output in self.outputs:
            mask[output - self.prefix["c_in"] - 2] = True
        
        return mask
    
    def get_active_idxs(self):
        return torch.nonzero(self.get_active_mask()).flatten()
    
    def __len__(self):
        return int(self.get_active_mask().sum().item())

    @staticmethod
    def get_active_mask_static(c_in, core, outputs):
        mask = torch.zeros(len(core))
        for node in core:
            mask[node.in_1 - c_in - 2] = True
            mask[node.in_2 - c_in - 2] = True

        for output in outputs:
            mask[output - c_in - 2] = True

        return mask

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
