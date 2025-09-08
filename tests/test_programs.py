import unittest
import numpy as np
import cvxpy as cp
from gcsopt.programs import ConicProgram, ConvexProgram

class TestConicProgram(unittest.TestCase):

    def setUp(self):

        # Linear program
        #   minimize x1 + 2 * x2 + 3 * x3 + 4 * x4 + 1
        # subject to xi >= 1,  i = 1, 2, 3, 4
        #            x1 + x2 + x3 + x4 <= 6
        #            x4 = 2
        self.lp = ConicProgram(4)
        c = np.arange(4) + 1
        d = 1
        self.lp.add_cost(c, d)
        I = np.eye(4)
        A = np.vstack((I, np.ones((1, 4)), I[-1:]))
        b = - np.array([1, 1, 1, 1, 6, 2])
        K = [(cp.constraints.NonNeg, 4), (cp.constraints.NonPos, 1), (cp.constraints.Zero, 1)]
        self.lp.add_constraints(A, b, K)

        # Second order cone program
        #   minimize sqrt(2) * x1 + 2
        # subject to x1^2 >= x1^2 + x2^2
        #            x1 >= x2
        #            x2 >= x3
        #            x3 >= 1
        self.socp = ConicProgram(3)
        c = np.array([np.sqrt(2), 0, 0])
        d = 2
        self.socp.add_cost(c, d)
        A = np.vstack((
            np.eye(3),
            np.array([[1, -1, 0], [0, 1, -1], [0, 0, 1]])
            ))
        b = np.array([0] * 5 + [-1])
        K = [(cp.constraints.SOC, 3), (cp.constraints.NonNeg, 3)]
        self.socp.add_constraints(A, b, K)

        # Semidefinite program equal to the previous SOCP. For translation, see
        # https://people.eecs.berkeley.edu/~elghaoui/Teaching/EE227A/lecture6.pdf
        #   minimize sqrt(2) * x1 + 2
        # subject to [x1, x2, x3]
        #            [x2, x1,  0] >> 0
        #            [x3,  0, x1]
        #            x1 >= x2
        #            x2 >= x3
        #            x3 >= 1
        self.sdp = ConicProgram(3)
        self.sdp.add_cost(c, d)
        A = np.array([
            [1, 0, 0], # entry 00
            [0, 1, 0], # entry 10
            [0, 0, 1], # entry 20
            [0, 1, 0], # entry 01
            [1, 0, 0], # entry 11
            [0, 0, 0], # entry 21
            [0, 0, 1], # entry 02
            [0, 0, 0], # entry 12
            [1, 0, 0], # entry 22
            [1, -1, 0],
            [0, 1, -1],
            [0, 0, 1],
            ])
        b = np.array([0] * 11 + [-1])
        K = [(cp.constraints.PSD, 9), (cp.constraints.NonNeg, 3)]
        self.sdp.add_constraints(A, b, K)

    def test_add_cost(self):

        # Create empty program.
        size = 4
        prog = ConicProgram(size)

        # Add a valid cost.
        c = np.ones(4)
        d = 3
        prog.add_cost(c, d)
        np.testing.assert_array_almost_equal(prog.c, c)
        self.assertAlmostEqual(prog.d, d)

        # Vector c has wrong size.
        c = np.ones(2)
        d = 3
        self.assertRaises(ValueError, prog.add_cost, c, d)

        # d is not a scalar.
        c = np.ones(4)
        d = np.ones(3)
        self.assertRaises(ValueError, prog.add_cost, c, d)

    def test_add_constraint(self):

        # Create empty program.
        size = 4
        prog = ConicProgram(size)

        # Add a valid constraint.
        A = np.eye(4)
        b = np.ones(4)
        K = (cp.constraints.Zero, 4)
        prog.add_constraint(A, b, K)
        np.testing.assert_array_almost_equal(prog.A, A)
        np.testing.assert_array_almost_equal(prog.b, b)
        self.assertEqual(len(prog.K), 1)
        self.assertEqual(prog.K[0], K)

        # Add valid constraints.
        new_K = [(cp.constraints.Zero, 2), (cp.constraints.Zero, 2)]
        prog.add_constraints(A, b, new_K)
        np.testing.assert_array_almost_equal(prog.A, np.vstack((A, A)))
        np.testing.assert_array_almost_equal(prog.b, np.concatenate((b, b)))
        self.assertEqual(len(prog.K), 3)
        self.assertEqual(prog.K[0], K)
        self.assertEqual(prog.K[1], new_K[0])
        self.assertEqual(prog.K[2], new_K[1])

        # Matrix A has incorrect number of columns.
        A = np.eye(8)
        b = np.ones(8)
        K = (cp.constraints.Zero, 8)
        self.assertRaises(ValueError, prog.add_constraint, A, b, K)

        # A, b, and K have incoherent sizes.
        A = np.eye(4)
        b = np.ones(5)
        K = (cp.constraints.Zero, 4)
        Ks = [(cp.constraints.Zero, 2), (cp.constraints.Zero, 2)]
        self.assertRaises(ValueError, prog.add_constraint, A, b, K)
        self.assertRaises(ValueError, prog.add_constraints, A, b, Ks)
        b = np.ones(4)
        K = (cp.constraints.Zero, 5)
        Ks = [(cp.constraints.Zero, 2), (cp.constraints.Zero, 3)]
        self.assertRaises(ValueError, prog.add_constraint, A, b, K)
        self.assertRaises(ValueError, prog.add_constraints, A, b, Ks)
        K = (cp.constraints.Zero, 4)
        Ks = [(cp.constraints.Zero, 2), (cp.constraints.Zero, 2)]
        A = np.eye(5)
        self.assertRaises(ValueError, prog.add_constraint, A, b, K)
        self.assertRaises(ValueError, prog.add_constraints, A, b, Ks)

    def test_cost_homogenization(self):

        # Tested only for linear program since unaffected by cones.
        x = [np.zeros(4), np.ones(4), np.arange(4)]
        costs = [1, 11, 21]
        for xi, cost in zip(x, costs):
            self.assertAlmostEqual(self.lp.cost_homogenization(xi, 1), cost)

    def test_solve(self):

        # linear program
        cost = self.lp.solve()
        self.assertAlmostEqual(cost, 15, places=4)
        np.testing.assert_array_almost_equal(self.lp.x.value, [1, 1, 1, 2], decimal=4)

        # second order cone program
        cost = self.socp.solve()
        self.assertAlmostEqual(cost, 4, places=4)
        np.testing.assert_array_almost_equal(self.socp.x.value, [np.sqrt(2), 1, 1], decimal=4)

        # semidefinite program
        cost = self.sdp.solve()
        self.assertAlmostEqual(cost, 4, places=4)
        np.testing.assert_array_almost_equal(self.sdp.x.value, [np.sqrt(2), 1, 1], decimal=4)

        # Program with no variables.
        prog = ConicProgram(0)
        prog.add_cost(np.zeros(0), 1.33)
        cost = prog.solve()
        self.assertAlmostEqual(cost, 1.33, places=4)
        self.assertIsNone(prog.x.value)
        self.assertEqual(prog.x.size, 0)
        self.assertEqual(prog.x.shape, (0,))

class TestConvexProgram(unittest.TestCase):

    def setUp(self):

        # Linear program:
        #   minimize x1 + 2 * x2 + 3 * x3 + 4 * x4 + 1
        # subject to xi >= 1,  i = 1, 2, 3, 4
        #            x1 + x2 + x3 + x4 <= 6
        #            x4 = 2
        self.lp = ConvexProgram()
        x = self.lp.add_variable(4)
        self.lp.add_cost(x[0] + 2 * x[1] + 3 * x[2] + 4 * x[3] + 1)
        self.lp.add_constraint(x >= 1)
        self.lp.add_constraints([cp.sum(x) <= 6, x[3] == 2])

        # Second order cone program:
        #   minimize sqrt(2) * x1 + 2
        # subject to x1^2 >= x1^2 + x2^2
        #            x1 >= x2
        #            x2 >= x3
        #            x3 >= 1
        self.socp = ConvexProgram()
        x = self.socp.add_variable(3)
        self.socp.add_cost(np.sqrt(2) * x[0] + 2)
        self.socp.add_constraint(cp.SOC(x[0], x[1:]))
        self.socp.add_constraints([x[0] >= x[1], x[1] >= x[2], x[2] >= 1])

        # Semidefinite program:
        #   minimize sqrt(2) * x1 + 2
        # subject to [x1, x2, x3]
        #            [x2, x1,  0] >> 0
        #            [x3,  0, x1]
        #            x1 >= x2
        #            x2 >= x3
        #            x3 >= 1
        self.sdp = ConvexProgram()
        x = self.sdp.add_variable(3)
        X = self.sdp.add_variable((3, 3), PSD=True)
        self.sdp.add_cost(np.sqrt(2) * x[0] + 2)
        self.sdp.add_constraints([x[0] >= x[1], x[1] >= x[2], x[2] >= 1])
        self.sdp.add_constraints([X[i, i] == x[0] for i in range(3)])
        self.sdp.add_constraints([X[1,0] == x[1], X[2,0] == x[2], X[2,1] == 0])

    def test_add_variable(self):

        # Define some variables.
        prog = ConvexProgram()
        prog.add_variable(3, nonneg=True)
        prog.add_variable(3, nonpos=True)
        prog.add_variable((3, 3), symmetric=True)
        prog.add_variable((3, 3), PSD=True)
        prog.add_variable((3, 3), NSD=True)
        self.assertEqual(len(prog.variables), 5)

        # Test shapes.
        self.assertEqual(prog.variables[0].shape, (3,))
        self.assertEqual(prog.variables[1].shape, (3,))
        self.assertEqual(prog.variables[2].shape, (3, 3))
        self.assertEqual(prog.variables[3].shape, (3, 3))
        self.assertEqual(prog.variables[4].shape, (3, 3))

        # Test attributes.
        self.assertTrue(prog.variables[0].is_nonneg())
        self.assertTrue(prog.variables[1].is_nonpos())
        self.assertTrue(prog.variables[2].is_symmetric())
        self.assertTrue(prog.variables[3].is_psd())
        self.assertTrue(prog.variables[4].is_nsd())

    def test_add_cost(self):
        
        # Define some costs.
        prog = ConvexProgram()
        x = prog.add_variable(3)
        X = prog.add_variable((3, 3), PSD=True)
        prog.add_cost(3)
        prog.add_cost(cp.norm2(x) - cp.log_det(X))

        # Raise error if external variable.
        y = cp.Variable(3)
        with self.assertRaises(ValueError):
            prog.add_cost(cp.norm2(y))

    def test_add_constraint(self):

        # Define some constraints.
        prog = ConvexProgram()
        x = prog.add_variable(3)
        X = prog.add_variable((3, 3), PSD=True)
        prog.add_constraint(x >= 0)
        prog.add_constraint(np.ones(3) @ x == 0)
        prog.add_constraint(cp.log_det(X) >= 3)
        prog.add_constraints([x >= 3, x <= 11])

        # Raise error if external variable.
        y = cp.Variable(3)
        with self.assertRaises(ValueError):
            prog.add_constraint(y >= 0)

    def test_solve(self):

        # Linear program.
        program_value = self.lp.solve()
        variable_values = [variable.value for variable in self.lp.variables]
        self.assertAlmostEqual(program_value, 15, places=4)
        np.testing.assert_array_almost_equal(variable_values[0], [1, 1, 1, 2], decimal=4)

        # Second order cone program.
        program_value = self.socp.solve()
        variable_values = [variable.value for variable in self.socp.variables]
        self.assertAlmostEqual(program_value, 4, places=4)
        np.testing.assert_array_almost_equal(variable_values[0], [np.sqrt(2), 1, 1], decimal=4)

        # Semidefinite program.
        program_value = self.sdp.solve()
        variable_values = [variable.value for variable in self.sdp.variables]
        self.assertAlmostEqual(program_value, 4, places=4)
        np.testing.assert_array_almost_equal(variable_values[0], [np.sqrt(2), 1, 1], decimal=4)
        X_opt = np.array([[np.sqrt(2), 1, 1], [1, np.sqrt(2), 0], [1, 0, np.sqrt(2)]])
        np.testing.assert_array_almost_equal(variable_values[1], X_opt, decimal=4)

    def test_to_conic(self):

        # List of programs to be tested.
        programs = [self.lp, self.socp, self.sdp]

        # Linear program with matrix variable
        #   minimize x11 + 2 * x12 + x13 + 3 * x21 + 4 * x22 + 5 * x23 + 1
        # subject to xij >= i,  i = 1, 2, j = 1, 2, 3
        #            x11 + x12 + x21 + x22 <= 6
        #            x22 = 2
        prog = ConvexProgram()
        X = prog.add_variable((2, 3))
        prog.add_cost(X[0,0] + 2 * X[0,1] + X[0,2] + 3 * X[1,0] + 4 * X[1,1] + 5 * X[1,2] + 1)
        prog.add_constraints([X[0,0] >= 1, X[0,1] >= 1, X[0,2] >= 1,
                                     X[1,0] >= 2, X[1,1] >= 2, X[1,2] >= 2,
                                     X[0,0] + X[0,1] + X[1,0] + X[1,1] <= 6,
                                     X[1,1] == 2])
        programs.append(prog)
        
        # Miscellaneous program.
        prog = ConvexProgram()
        x = prog.add_variable(3, nonneg=True)
        y = prog.add_variable(2, nonneg=True)
        X = prog.add_variable((3, 3), PSD=True)
        Y = prog.add_variable((3, 4))
        prog.add_cost(cp.norm2(x + 2) + cp.sum_squares(y + 3) + 3)
        prog.add_cost(- cp.log_det(X) + X[0, 2] + cp.max(Y[0]))
        prog.add_constraints([x >= -3, y <= 3, X[0, 1] == 1, Y == np.ones(Y.shape)])
        programs.append(prog)

        # Infeasible program.
        prog = ConvexProgram()
        x = prog.add_variable(4, nonneg=True)
        prog.add_cost(cp.sum(x))
        prog.add_constraint(x <= -1)
        programs.append(prog)

        # Infeasible program with multiple variables.
        prog = ConvexProgram()
        x = prog.add_variable(4, nonneg=True)
        y = prog.add_variable(2)
        prog.add_cost(cp.sum(x) + cp.sum(y))
        prog.add_constraint(x <= -1)
        programs.append(prog)

        # Unbounded program.
        prog = ConvexProgram()
        x = prog.add_variable(4)
        prog.add_cost(cp.sum(x))
        prog.add_constraint(x <= -1)
        programs.append(prog)

        # Trivial sdp.
        prog = ConvexProgram()
        X = prog.add_variable((1, 1), PSD=True)
        prog.add_cost(- cp.log_det(X))
        prog.add_constraint(X[0, 0] == 1)
        programs.append(prog)

        # Sdp for inimum volume ellipsoid.
        for d in range(2, 6):
            points = np.vstack((np.zeros((1, d)), np.eye(d)))
            prog = ConvexProgram()
            A = prog.add_variable((d, d), PSD=True)
            b = prog.add_variable(d)
            prog.add_cost(- cp.log_det(A))
            for point in points:
                prog.add_constraint(cp.norm2(A @ point + b) <= 1)
            programs.append(prog)

        # Check that solving convex program is equal to solve conic program.
        for convex_prog in programs:
            conic_prog = convex_prog.to_conic()

            # Same objectives.
            convex_value = convex_prog.solve()
            conic_value = conic_prog.solve()
            self.assertAlmostEqual(convex_value, conic_value, places=4)

            # Same optimal solutions.
            for variable in convex_prog.variables:
                if variable.value is None:
                    self.assertIsNone(conic_prog.x.value)
                else:
                    conic_value = conic_prog.get_convex_variable_value(variable)
                    np.testing.assert_array_almost_equal(variable.value, conic_value, decimal=4)

    def test_to_conic_corner_cases(self):

        # Program with constant cost.
        convex_prog = ConvexProgram()
        x = convex_prog.add_variable(4)
        convex_prog.add_cost(1.55)
        convex_prog.add_constraint(x >= 0)
        conic_prog = convex_prog.to_conic()
        convex_value = convex_prog.solve()
        conic_value = conic_prog.solve()
        self.assertAlmostEqual(convex_value, 1.55, places=4)
        self.assertAlmostEqual(conic_value, 1.55, places=4)

        # Program with constant cost and no variables.
        convex_prog = ConvexProgram()
        convex_prog.add_cost(1.55)
        conic_prog = convex_prog.to_conic()
        convex_value = convex_prog.solve()
        conic_value = conic_prog.solve()
        self.assertAlmostEqual(convex_value, 1.55, places=4)
        self.assertAlmostEqual(conic_value, 1.55, places=4)

        # Program with free variable.
        prog = ConvexProgram()
        x = prog.add_variable(4, nonneg=True)
        prog.add_cost(1.55)
        conic_prog = convex_prog.to_conic()
        convex_value = convex_prog.solve()
        conic_value = conic_prog.solve()
        self.assertAlmostEqual(convex_value, 1.55, places=4)
        self.assertAlmostEqual(conic_value, 1.55, places=4)

        # Program with no constraints.
        convex_prog = ConvexProgram()
        x = convex_prog.add_variable(3)
        convex_prog.add_cost(cp.norm_inf(x))
        conic_prog = convex_prog.to_conic()
        convex_value = convex_prog.solve()
        conic_value = conic_prog.solve()
        self.assertAlmostEqual(convex_value, 0, places=4)
        self.assertAlmostEqual(conic_value, 0, places=4)

if __name__ == '__main__':
    unittest.main()
