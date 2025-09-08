import numpy as np
import cvxpy as cp
from numbers import Number

class ConicProgram:

    def __init__(self, size, id_to_range=None, binary_variable=None):

        # Matrices of conic program.
        self.size = size
        self.c = np.zeros(size)
        self.d = 0
        self.A = np.zeros((0, size))
        self.b = np.zeros(0)
        self.K = []

        # Variables and data from the convex program.
        self.id_to_range = id_to_range
        self.x = cp.Variable(size) if size > 0 else None
        self.binary_variable = cp.Variable() if binary_variable is None else binary_variable

    def add_cost(self, ci, di):
        if not isinstance(di, Number):
            raise ValueError(
                f"Argument di must be a number, got type(di) = {type(di)}.")
        self.c += ci
        self.d += di

    def add_constraint(self, Ai, bi, Ki):
        if len({Ai.shape[0], bi.size, Ki[1]}) != 1:
            raise ValueError(
                "Matrix Ai, vector bi, and cone Ki must have coherent size. "
                f"Got Ai of shape {Ai.shape}, bi of size {bi.size}, and K of "
                f"size {Ki[1]}.")
        self.A = np.vstack((self.A, Ai))
        self.b = np.concatenate((self.b, bi))
        self.K.append(Ki)

    def add_constraints(self, A, b, K):
        K_size = sum([Ki[1] for Ki in K])
        if len({A.shape[0], b.size, K_size}) != 1:
            raise ValueError(
                "Matrix A, vector b, and cones K must have coherent size. Got "
                f"A of shape {A.shape}, b of size {b.size}, K of total size "
                f"{K_size}.")
        self.A = np.vstack((self.A, A))
        self.b = np.concatenate((self.b, b))
        self.K.extend(K)

    def cost_homogenization(self, x, y):
        value = self.d * y
        if self.size > 0:
            value += self.c @ x
        return value

    def constraint_homogenization(self, x, y):
        constraints = []
        if self.size == 0 or self.A.shape[0] == 0: # Unconstrained problem.
            return constraints
        z = self.A @ x + self.b * y
        start = 0
        for cone_type, cone_size in self.K:
            stop = start + cone_size
            constraint = self._constrain_in_cone(z[start:stop], cone_type)
            constraints.append(constraint)
            start = stop
        return constraints
    
    @staticmethod
    def _constrain_in_cone(z, K):

        # Linear constraints.
        if K in [cp.Zero, cp.NonNeg]:
            return K(z)
        elif K == cp.NonPos: # NonPos will be deprecated.
            return cp.NonNeg(- z)
        
        # Second order cone constraint.
        elif K == cp.SOC:
            return K(z[0], z[1:])
        
        # Semidefinite constraint.
        elif K == cp.PSD:
            n = round(z.size ** .5)
            z_mat = cp.reshape(z, (n, n), order='F')
            return K(z_mat)
        
        # Exponential cone constraint.
        elif K == cp.ExpCone:
            z_mat = z.reshape((3, -1), order='C')
            return K(*z_mat)
        
        # TODO: support all cone constraints.
        else:
            raise NotImplementedError
        
    def get_convex_variable_value(self, convex_variable, x=None):

        # Retrieve value.
        if x is None:
            x = self.x.value
        if x is None:
            return None
        value = x[self.id_to_range[convex_variable.id]]

        # One dimensional vector.
        if len(convex_variable.shape) == 1:
            return value
        
        # Asymmetric matrix.
        if not convex_variable.is_symmetric():
            return value.reshape(convex_variable.shape, order='F')
        
        # Symmetric matrix.
        n = convex_variable.shape[0]
        mat_value = np.zeros((n, n))
        mat_value[np.triu_indices(n)] = value
        mat_value.T[np.triu_indices(n)] = value
        return mat_value
    
    def solve(self, **kwargs):
        """
        This method is only used for testing, it is not used in the library.
        """
        cost = self.cost_homogenization(self.x, 1)
        constraints = self.constraint_homogenization(self.x, 1)
        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(**kwargs)
        return prob.value

class ConvexProgram:

    supported_attributes = ["nonneg", "nonpos", "symmetric", "PSD", "NSD"]

    def __init__(self):
        self.variables = []
        self.cost = 0
        self.constraints = []
        self.binary_variable = cp.Variable()

    def add_variable(self, shape, **kwargs):
        for attribute in kwargs:
            if not attribute in self.supported_attributes:
                raise ValueError(f"Attribute {attribute} is not supported.")
        variable = cp.Variable(shape, **kwargs)
        self.variables.append(variable)
        return variable
    
    def _check_variables_are_defined(self, variables, defined_variables=None):
        if defined_variables is None:
            defined_variables = self.variables
        defined_ids = [variable.id for variable in defined_variables]
        for variable in variables:
            if variable.id not in defined_ids:
                raise ValueError(f"Variable {variable} is not defined.")
    
    def add_cost(self, cost):
        if not isinstance(cost, Number):
            self._check_variables_are_defined(cost.variables())
        self.cost += cost

    def add_constraint(self, constraint):
        self._check_variables_are_defined(constraint.variables())
        self.constraints.append(constraint)

    def add_constraints(self, constraints):
        for constraint in constraints:
            self.add_constraint(constraint)

    def _ensure_variable_usage(self):
        """
        Ensure that all the variables are included in the conic program. If one
        variable is not used, add a dummy cost of zero times the variable.
        """
        used_ids = [v.id for c in self.constraints for v in c.variables()]
        if not isinstance(self.cost, Number):
            used_ids += [v.id for v in self.cost.variables()]
        for v in self.variables:
            if not v.id in used_ids:
                self.add_cost(0 * v.flatten(order="F")[0])

    def to_conic(self):
        """
        Converts this ConvexProgram into an equivalent ConicProgram, using the
        reduction chain in cvxpy.
        """

        # Ensure that all the variables are used in the conic program.
        self._ensure_variable_usage()
        
        # Deal with corner case with constant cost and no constraints.
        if isinstance(self.cost, Number) and len(self.constraints) == 0:
            conic_program = ConicProgram(0, binary_variable=self.binary_variable)
            conic_program.add_cost([], self.cost)
            return conic_program

        # Apply cvxpy reductions to get conic program.
        cp_convex = cp.Problem(cp.Minimize(self.cost), self.constraints)
        if not cp_convex.is_dcp():
            raise ValueError(f"Convex program is not DCP.")
        solver_opts = {"use_quad_obj": False}
        chain = cp_convex._construct_chain(solver_opts=solver_opts)
        chain.reductions = chain.reductions[:-1]
        cp_conic = chain.apply(cp_convex)[0]

        # Dictionary that maps the id of a variable in the cost and constraints
        # to the corresponding columns in the in the conic program.
        id_to_range = {}
        for variable in cp_conic.variables:
            start = cp_conic.var_id_to_col[variable.id]
            stop = start + variable.size
            id_to_range[variable.id] = range(start, stop)

        # Initialize empty conic program.
        conic_program = ConicProgram(cp_conic.x.size, id_to_range, self.binary_variable)

        # Define cost of conic program.
        cd = cp_conic.c.toarray().flatten()
        conic_program.add_cost(cd[:-1], cd[-1])

        # Define constraints of conic program. Sparse matrices are converted to
        # dense arrays, since keeping them sparse seems to make things slower.
        cols = cp_conic.c.shape[0]
        Ab = cp_conic.A.toarray().reshape((-1, cols), order='F')
        K = [(type(c), c.size) for c in cp_conic.constraints]
        conic_program.add_constraints(Ab[:, :-1], Ab[:, -1], K)

        return conic_program

    def solve(self, **kwargs):
        """
        This method is only used for testing, it is not used in the library.
        """
        prob = cp.Problem(cp.Minimize(self.cost), self.constraints)
        prob.solve(**kwargs)
        return prob.value