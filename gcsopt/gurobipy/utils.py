def has_gurobi():
    import importlib.util
    return importlib.util.find_spec("gurobipy")
