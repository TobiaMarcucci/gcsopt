import numpy as np
import cvxpy as cp
from gcsopt.programs import ConicProgram, ConvexProgram

class ConicEdge(ConicProgram):

    def __init__(self, tail, head, size, id_to_range=None, binary_variable=None):

        # Check inputs.
        super().__init__(size, id_to_range, binary_variable)
        self.slack_size = size - tail.size - head.size
        if self.slack_size < 0:
            raise ValueError(
                f"Size mismatch: edge.size = {size}, tail.size = {tail.size}, "
                f"head.size = {head.size}. Size of the edge must be larger "
                "than the sum of tail and head sizes.")

        # Store inputs.
        self.tail = tail
        self.head = head
        self.name = (tail.name, head.name)

    def _check_vector_sizes(self, xv, xw, xe):
        sizes = (xv.size, xw.size, xe.size)
        expected_sizes = (self.tail.size, self.head.size, self.slack_size)
        if sizes != expected_sizes:
            raise ValueError(
                f"Size mismatch. Got vectors of size {sizes}. Expected vectors "
                f"of size {expected_sizes}.")
                
    def _concatenate(self, xv, xw, xe):
        if self.slack_size == 0:
            return cp.hstack((xv, xw))
        else:
            return cp.hstack((xv, xw, xe))

    def cost_homogenization(self, xv, xw, xe, y):
        self._check_vector_sizes(xv, xw, xe)
        x = self._concatenate(xv, xw, xe)
        return super().cost_homogenization(x, y)
        
    def constraint_homogenization(self, xv, xw, xe, y):
        self._check_vector_sizes(xv, xw, xe)
        x = self._concatenate(xv, xw, xe)
        return super().constraint_homogenization(x, y)
    
class ConvexEdge(ConvexProgram):

    def __init__(self, tail, head):
        super().__init__()
        self.tail = tail
        self.head = head
        self.name = (self.tail.name, self.head.name)

    def to_conic(self, conic_tail, conic_head):

        # Include tail and head variables in id_to_range. Variable order is
        # x_tail, x_head, a dn then x_edge. Start with copy of tail dictionary.
        id_to_range = conic_tail.id_to_range.copy()

        # Add copy of the head dictionary shifted by the size of the tail.
        offset = conic_tail.size
        for id, r in conic_head.id_to_range.items():
            id_to_range[id] = range(r.start + offset, r.stop + offset)

        # Add to dictionary variables that are associated with this edge if they
        # are not in the tail or head dictionary yet.
        conic_program = super().to_conic()
        offset = conic_tail.size + conic_head.size
        for id, r in conic_program.id_to_range.items():
            if not id in id_to_range:
                id_to_range[id] = range(r.start + offset, r.stop + offset)

        # Initialize empty edge program.
        size = max([r.stop for r in id_to_range.values()])
        conic_edge = ConicEdge(
            conic_tail,
            conic_head,
            size,
            id_to_range,
            conic_program.binary_variable)

        # Reorder matrices and extend them with zeros according to the new
        # id_to_range dictionary.
        c = np.zeros(size)
        A = np.zeros((conic_program.A.shape[0], size))
        for id, r in conic_program.id_to_range.items():
            c[id_to_range[id]] = conic_program.c[r]
            A[:, id_to_range[id]] = conic_program.A[:, r]

        # Assemble conic edge.
        conic_edge.add_cost(c, conic_program.d)
        conic_edge.add_constraints(A, conic_program.b, conic_program.K)
        return conic_edge

    def _check_variables_are_defined(self, variables):
        defined_variables = self.variables + self.tail.variables + self.head.variables
        super()._check_variables_are_defined(variables, defined_variables)
