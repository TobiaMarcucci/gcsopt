import cvxpy as cp

class EmptyVariable():

    def __init__(self):
        self.value = None
        self.size = 0
        self.shape = (0,)

def safe_variable(size):
    if size > 0:
        return cp.Variable(size)
    elif size == 0:
        return EmptyVariable()
    else:
        raise ValueError("Variable size must be nonnegative.")