import time 
import z3
import dreal
import lyznet
import sympy as sp


def z3_global_quadratic_verifier(system, eps=1e-5):
    print('_' * 50)
    print("Verifying global stability using quadratic Lyapunov function "
          "(with Z3): ")

    z3_x = [z3.Real(f"x{i}") for i in range(1, len(system.symbolic_vars) + 1)]

    dreal_x = [
        dreal.Variable(f"x{i}") 
        for i in range(1, len(system.symbolic_vars) + 1)
        ]
    dreal_V = dreal.Expression(0)
    for i in range(len(dreal_x)):
        for j in range(len(dreal_x)):
            dreal_V += dreal_x[i] * system.P[i][j] * dreal_x[j]

    dreal_f = [
        lyznet.utils.sympy_to_dreal(
            expr, dict(zip(system.symbolic_vars, dreal_x))
            )
        for expr in system.symbolic_f
        ]

    lie_derivative_of_V_dreal = dreal.Expression(0)
    for i in range(len(dreal_x)):
        lie_derivative_of_V_dreal += dreal_f[i] * dreal_V.Differentiate(
            dreal_x[i])
    # print(lie_derivative_of_V_dreal)

    norm_x_squared = sum([x**2 for x in z3_x])

    solver = z3.Solver()

    lie_derivative_of_V_z3 = lyznet.utils.dreal_to_z3(
        lie_derivative_of_V_dreal, z3_x)
    print("DV*f: ", lie_derivative_of_V_z3)

    solver.add(lie_derivative_of_V_z3 > -eps*norm_x_squared)
    
    result = solver.check()

    if result == z3.unsat:
        print("Verified: The EP is globally asymptotically stable.")
    else:
        print("Cannot verify global asymptotic stability. "
              "Counterexample: ")
        print(solver.model())


def z3_quadratic_verifier(system, c_max=100, eps=1e-5, accuracy=1e-4):
    z3_x = [z3.Real(f"x{i}") for i in range(1, len(system.symbolic_vars) + 1)]
    norm_x_squared = sum([x**2 for x in z3_x])

    xPx = sum([z3_x[i] * sum([system.P[i][j] * z3_x[j] 
               for j in range(len(z3_x))]) for i in range(len(z3_x))])

    norm_x_squared = sum([x**2 for x in z3_x])

    dreal_x = [
        dreal.Variable(f"x{i}") 
        for i in range(1, len(system.symbolic_vars) + 1)
        ]
    dreal_V = dreal.Expression(0)
    for i in range(len(dreal_x)):
        for j in range(len(dreal_x)):
            dreal_V += dreal_x[i] * system.P[i][j] * dreal_x[j]

    dreal_f = [
        lyznet.utils.sympy_to_dreal(
            expr, dict(zip(system.symbolic_vars, dreal_x))
            )
        for expr in system.symbolic_f
        ]

    lie_derivative_of_V_dreal = dreal.Expression(0)
    for i in range(len(dreal_x)):
        lie_derivative_of_V_dreal += dreal_f[i] * dreal_V.Differentiate(
            dreal_x[i])

    lie_derivative_of_V_z3 = lyznet.utils.dreal_to_z3(
        lie_derivative_of_V_dreal, z3_x)
 
    def verify_level_c(c):
        solver = z3.Solver()
        solver.add(z3.And(xPx <= c, 
                          lie_derivative_of_V_z3 + eps * norm_x_squared > 0))

        result = solver.check()
        if result == z3.unsat:
            return None
        else:
            return solver.model()

    lyznet.tik()
    c = lyznet.utils.bisection_glb(verify_level_c, 0, c_max, accuracy=accuracy)
    print(f"Region of attraction verified for x^TPx<={c}.")
    lyznet.tok()
    return c


def extract_sympy_PolyNet(system, net):
    x = sp.Matrix(system.symbolic_vars)
    layers = len(net.layers)
    
    weights = [layer.weight.data.cpu().numpy() for layer in net.layers]
    biases = [layer.bias.data.cpu().numpy() for layer in net.layers]
    final_layer_weight = net.final_layer.weight.data.cpu().numpy()
    final_layer_bias = net.final_layer.bias.data.cpu().numpy()

    h = x
    for i in range(layers):
        z = sp.Matrix([sum(h[j] * weights[i][k][j] for j in range(h.shape[0])) + biases[i][k] for k in range(len(biases[i]))])
        h = sp.Matrix([z[k]**net.deg for k in range(z.shape[0])])

    V_net = sum(h[k] * final_layer_weight[0][k] for k in range(h.shape[0])) + final_layer_bias[0]    
    return V_net


def z3_clf_verifier(system, net, c_max=100, accuracy=1e-2):
    V_net = extract_sympy_PolyNet(system, net)

    print('_' * 50)
    print("Verifying control Lyapunov function (with Z3):")    
    print("V = ", sp.simplify(V_net))

    V_net_matrix = sp.Matrix([V_net])
    
    # Symbolic variables and expressions for gradient and dynamics
    x = sp.Matrix(system.symbolic_vars)
    grad_V = V_net_matrix.jacobian(x)  # Compute Jacobian of V_net
    LfV = grad_V * system.symbolic_f
    LgV = grad_V * system.symbolic_g

    # Generate Z3 variables based on the system's symbolic variables
    z3_x = [z3.Real(f"x{i+1}") for i in range(len(system.symbolic_vars))]
    subs = {system.symbolic_vars[i]: z3_x[i] for i in range(len(system.symbolic_vars))}
    
    # Convert the SymPy expressions to Z3
    func_subs = {}
    extra_constraints = []
    V_net_z3 = lyznet.utils.sympy_to_z3(V_net, subs, func_subs, extra_constraints)

    LfV_z3 = lyznet.utils.sympy_to_z3(LfV[0], subs, func_subs, extra_constraints)
    LgV_z3_list = [lyznet.utils.sympy_to_z3(expr, subs, func_subs, extra_constraints) for expr in LgV]

    # Construct the conditions
    LgV_zero_z3 = z3.And(*[expr == 0 for expr in LgV_z3_list])
    z3_x_non_zero = z3.Or(*[x != 0 for x in z3_x]) 
    condition = z3.And(LgV_zero_z3, z3_x_non_zero)

    V_positive_condition = z3.Implies(z3_x_non_zero, V_net_z3 > 0)
    
    def verify_level_c(c2_P):
        solver = z3.Solver()
        
        # Construct the condition for V(x) <= c2_P
        c_condition = z3.And(V_net_z3 <= c2_P)
        
        # Final condition: LfV < 0 given the conditions for LgV, x, and V(x)
        clf_condition = z3.Implies(z3.And(condition, c_condition), LfV_z3 < 0)

        clf_condition = z3.And(clf_condition, V_positive_condition)
        
        # Add extra constraints for sin/cos variables
        solver.add(extra_constraints)
        solver.add(z3.Not(clf_condition))  # Negation of the CLF condition
        
        # Check for satisfiability
        result = solver.check()
        
        if result == z3.unsat:
            # print(f"level {c2_P} verified")
            return None
        else:
            return solver.model()

    # Perform bisection to find the largest c2_P that satisfies the CLF condition
    lyznet.tik()
    c_optimal = lyznet.utils.bisection_glb(verify_level_c, 0, c_max, accuracy=accuracy)
    print(f"CLF condition verified for x^T P x <= {c_optimal}.")
    lyznet.tok()

    return c_optimal


def z3_global_clf_verifier(system, net, c2_P=None):
    """Global CLF verification using Z3 for a neural network-based Lyapunov function."""
    
    # Extract the neural network as a SymPy expression
    V_net = extract_sympy_PolyNet(system, net)
    print('_' * 50)
    print("Verifying global control Lyapunov function (with Z3):")    
    print("V = ", sp.simplify(V_net))

    # Symbolic variables and expressions for gradient and dynamics

    V_net_matrix = sp.Matrix([V_net])
    
    # Symbolic variables and expressions for gradient and dynamics
    x = sp.Matrix(system.symbolic_vars)
    grad_V = V_net_matrix.jacobian(x)  # Compute Jacobian of V_net
    LfV = grad_V * system.symbolic_f
    LgV = grad_V * system.symbolic_g

    # Generate Z3 variables based on the system's symbolic variables
    z3_x = [z3.Real(f"x{i+1}") for i in range(len(system.symbolic_vars))]
    subs = {system.symbolic_vars[i]: z3_x[i] for i in range(len(system.symbolic_vars))}
    
    # Convert the SymPy expressions to Z3
    func_subs = {}
    extra_constraints = []
    V_net_z3 = lyznet.utils.sympy_to_z3(V_net, subs, func_subs, extra_constraints)
    LfV_z3 = lyznet.utils.sympy_to_z3(LfV[0], subs, func_subs, extra_constraints)
    LgV_z3_list = [lyznet.utils.sympy_to_z3(expr, subs, func_subs, extra_constraints) for expr in LgV]

    # Construct the conditions
    LgV_zero_z3 = z3.And(*[expr == 0 for expr in LgV_z3_list])
    z3_x_non_zero = z3.Or(*[x != 0 for x in z3_x])  # At least one variable is non-zero
    condition = z3.And(LgV_zero_z3, z3_x_non_zero)

    if c2_P is not None:
        # Add the constraint V(x) >= c2_P
        condition = z3.And(condition, V_net_z3 >= c2_P)

    clf_condition = z3.Implies(condition, LfV_z3 < 0)

    z3_x_non_zero = z3.Or(*[x != 0 for x in z3_x])  
    V_positive_condition = z3.Implies(z3_x_non_zero, V_net_z3 > 0)
    clf_condition = z3.And(clf_condition, V_positive_condition)

    # Initialize the solver
    solver = z3.Solver()
    solver.add(extra_constraints)  
    solver.add(z3.Not(clf_condition))  

    lyznet.tik() 
    if solver.check() == z3.unsat:
        print("Global CLF condition is verified!")
        success = True

    else:
        print("The global CLF condition is not valid. Counterexample:")
        model = solver.model()
        for i, var in enumerate(z3_x):
            print(f"x{i+1} = {model[var]}")
        for func_expr, z3_var in func_subs.items():
            print(f"{z3_var} = {model[z3_var]} (represents {func_expr})")
    
        # Evaluate the conditions using the counterexample model
        LgV_zero_val = model.eval(LgV_zero_z3, model_completion=True)
        z3_x_non_zero_val = model.eval(z3_x_non_zero, model_completion=True)
        LfV_val = model.eval(LfV_z3 < 0, model_completion=True)
        V_net_val = model.eval(V_net_z3 > 0, model_completion=True)  # Check if V(x) > 0

        print(f"LgV=zero evaluates to: {LgV_zero_val}")
        print(f"x_non_zero evaluates to: {z3_x_non_zero_val}")
        print(f"LfV < 0 evaluates to: {LfV_val}")
        print(f"V > 0 evaluates to: {V_net_val}") 

        success = False
        
    lyznet.tok() 

    return success


def z3_global_quadratic_clf_verifier(system, c2_P=None):
    x = sp.Matrix(system.symbolic_vars)    
    V = x.T * sp.Matrix(system.P) * x
    grad_V = V.jacobian(x)
    LfV = grad_V * system.symbolic_f
    LgV = grad_V * system.symbolic_g

    print('_' * 50)
    print("Verifying global quadratic control Lyapunov function (with Z3):")
    print("V = ", sp.simplify(V[0]))

    # Create Z3 variables for state variables
    num_vars = len(system.symbolic_vars)
    z3_x = [z3.Real(f"x{i+1}") for i in range(num_vars)]
    subs = {system.symbolic_vars[i]: z3_x[i] for i in range(num_vars)}

    # Dictionaries to hold function substitutions and extra constraints
    func_subs = {}
    extra_constraints = []

    # Convert SymPy expressions to Z3 expressions
    V_z3 = lyznet.utils.sympy_to_z3(V[0], subs, func_subs, extra_constraints)
    LfV_z3 = lyznet.utils.sympy_to_z3(LfV[0], subs, func_subs, extra_constraints)
    LgV_z3_list = [lyznet.utils.sympy_to_z3(expr, subs, func_subs, extra_constraints) for expr in LgV]

    # Construct the conditions
    LgV_zero_z3 = z3.And(*[expr == 0 for expr in LgV_z3_list])
    z3_x_non_zero = z3.Or(*[x != 0 for x in z3_x])  # At least one variable is non-zero
    condition = z3.And(LgV_zero_z3, z3_x_non_zero)

    if c2_P is not None:
        # Add the constraint V(x) >= c2_P
        condition = z3.And(condition, V_z3 >= c2_P)
    clf_condition = z3.Implies(condition, LfV_z3 < 0)

    # Initialize the solver
    solver = z3.Solver()
    solver.add(extra_constraints)  # Add constraints for sin and cos variables
    # print("extra_constraints: ", extra_constraints)

    solver.add(z3.Not(clf_condition))  # Add the negation of the CLF condition
    # print("clf_condition: ", clf_condition)

    lyznet.tik()
    # Check for satisfiability
    if solver.check() == z3.unsat:
        print("Global CLF condition is verified!")
    else:
        print("The global CLF condition is not valid. Counterexample:")
        model = solver.model()
        for i, var in enumerate(z3_x):
            print(f"x{i+1} = {model[var]}")
        # Print the values of the function variables
        for func_expr, z3_var in func_subs.items():
            print(f"{z3_var} = {model[z3_var]} (represents {func_expr})")
        
        # Evaluate the conditions using the counterexample model
        LgV_zero_val = model.eval(LgV_zero_z3, model_completion=True)
        z3_x_non_zero_val = model.eval(z3_x_non_zero, model_completion=True)
        LfV_val = model.eval(LfV_z3 < 0, model_completion=True)
        V_val = model.eval(V_z3 > 0, model_completion=True)  # Check if V(x) > 0

        print(f"LgV=zero evaluates to: {LgV_zero_val}")
        print(f"x_non_zero evaluates to: {z3_x_non_zero_val}")
        print(f"LfV < 0 evaluates to: {LfV_val}")
        print(f"V > 0 evaluates to: {V_val}") 

    lyznet.tok()

def z3_quadratic_clf_verifier(system, c_max=100, accuracy=1e-3):
    x = sp.Matrix(system.symbolic_vars)    
    V = x.T * sp.Matrix(system.P) * x
    grad_V = V.jacobian(x)    
    LfV = grad_V * system.symbolic_f
    LgV = grad_V * system.symbolic_g
    
    print('_' * 50)
    print("Verifying quadratic control Lyapunov function (with Z3):")
    print("V = ", sp.simplify(V[0]))

    # Create Z3 variables for state variables
    num_vars = len(system.symbolic_vars)
    z3_x = [z3.Real(f"x{i+1}") for i in range(num_vars)]
    subs = {system.symbolic_vars[i]: z3_x[i] for i in range(num_vars)}

    # Dictionaries to hold function substitutions and extra constraints
    func_subs = {}
    extra_constraints = []

    # Convert SymPy expressions to Z3 expressions
    V_z3 = lyznet.utils.sympy_to_z3(V[0], subs, func_subs, extra_constraints)
    LfV_z3 = lyznet.utils.sympy_to_z3(LfV[0], subs, func_subs, extra_constraints)
    LgV_z3_list = [lyznet.utils.sympy_to_z3(expr, subs, func_subs, extra_constraints) for expr in LgV]

    # Construct the conditions
    LgV_zero_z3 = z3.And(*[expr == 0 for expr in LgV_z3_list])
    z3_x_non_zero = z3.Or(*[x != 0 for x in z3_x])  
    condition = z3.And(LgV_zero_z3, z3_x_non_zero)

    def verify_level_c(c2_P):
        solver = z3.Solver()
        c_condition = z3.And(V_z3 <= c2_P)
        clf_condition = z3.Implies(z3.And(condition, c_condition), LfV_z3 < 0)        
        # Add extra constraints for sin/cos variables
        solver.add(extra_constraints)
        solver.add(z3.Not(clf_condition))  # Negation of the CLF condition
        result = solver.check()
        
        if result == z3.unsat:
            return None
        else:
            return solver.model()

    lyznet.tik()
    c_optimal = lyznet.utils.bisection_glb(verify_level_c, 0, c_max, accuracy=accuracy)
    print(f"CLF condition verified for x^T P x <= {c_optimal}.")
    lyznet.tok()

    return c_optimal
