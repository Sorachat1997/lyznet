import time 

import dreal
import lyznet

import sympy
from fractions import Fraction

def reach_verifier_dreal(system, x, V, f, c1, V_f_x=None, c_max=100, 
                         tol=1e-4, accuracy=1e-4, number_of_jobs=32):
    config = lyznet.utils.config_dReal(number_of_jobs=number_of_jobs, tol=tol)
    xlim = system.domain
    epsilon = 1e-4

    if system.system_type == "DiscreteDynamicalSystem":
        if V_f_x is None:
            f = [lyznet.utils.simplify_dreal_expression(expr, x) for expr in f]
            V_f_x = lyznet.utils.compose_dreal_expressions(V, x, f, x)
        lyap_condition = V_f_x - V 

    else:        
        lie_derivative_of_V = dreal.Expression(0)
        for i in range(len(x)):
            lie_derivative_of_V += f[i] * V.Differentiate(x[i])
        lyap_condition = lie_derivative_of_V

    def Check_reachability(c2):    
        x_bound = lyznet.utils.get_bound(x, xlim, V, c1_V=c1, c2_V=c2)
        x_boundary = dreal.logical_or(x[0] == xlim[0][0], x[0] == xlim[0][1])
        for i in range(1, len(x)):
            x_boundary = dreal.logical_or(x[i] == xlim[i][0], x_boundary)
            x_boundary = dreal.logical_or(x[i] == xlim[i][1], x_boundary)
        set_inclusion = dreal.logical_imply(
            x_bound, dreal.logical_not(x_boundary)
            )
        reach = dreal.logical_imply(x_bound, lyap_condition <= -epsilon)
        condition = dreal.logical_and(reach, set_inclusion)
        return dreal.CheckSatisfiability(dreal.logical_not(condition), config)

    c_best = lyznet.utils.bisection_glb(Check_reachability, 
                                        c1, c_max, accuracy)
    return c_best


def clf_reach_verifier_dreal(system, x, V, c1, c_max=100, tol=1e-4, 
                             accuracy=1e-4, number_of_jobs=32):
    config = lyznet.utils.config_dReal(number_of_jobs=number_of_jobs, tol=tol)
    xlim = system.domain

    f = [
        lyznet.utils.sympy_to_dreal(
            expr, dict(zip(system.symbolic_vars, x))
            )
        for expr in system.symbolic_f
        ]

    g = [
        [
            lyznet.utils.sympy_to_dreal(expr, dict(zip(system.symbolic_vars, x)))
            for expr in row
        ]
        for row in system.symbolic_g.tolist()
    ]

    LfV = dreal.Expression(0)
    for i in range(len(x)):
            LfV += f[i] * V.Differentiate(x[i])
    grad_V = [[V.Differentiate(x[i]) for i in range(len(x))]]
    LgV = lyznet.utils.matrix_multiply_dreal(grad_V, g)
    LgV_zero = dreal.And(*(expr == dreal.Expression(0) for row in LgV 
                                     for expr in row))

    def Check_clf_condition(c2):    
        x_bound = lyznet.utils.get_bound(x, xlim, V, c1_V=c1, c2_V=c2)
        x_boundary = dreal.logical_or(x[0] == xlim[0][0], x[0] == xlim[0][1])
        for i in range(1, len(x)):
            x_boundary = dreal.logical_or(x[i] == xlim[i][0], x_boundary)
            x_boundary = dreal.logical_or(x[i] == xlim[i][1], x_boundary)
        set_inclusion = dreal.logical_imply(
            x_bound, dreal.logical_not(x_boundary)
            )
        clf_condition = dreal.logical_imply(dreal.And(x_bound, LgV_zero), 
                                            LfV < 0)
        condition = dreal.logical_and(clf_condition, set_inclusion)
        return dreal.CheckSatisfiability(dreal.logical_not(condition), config)
    
    c_best = lyznet.utils.bisection_glb(Check_clf_condition, 
                                        c1, c_max, accuracy)

    return c_best


def quadratic_reach_verifier(system, c1_P, tol=1e-4, accuracy=1e-4,
                             number_of_jobs=32, c_max=100):
    # Create dReal variables based on the number of symbolic variables
    x = [
        dreal.Variable(f"x{i}") 
        for i in range(1, len(system.symbolic_vars) + 1)
        ]
    V = dreal.Expression(0)
    for i in range(len(x)):
        for j in range(len(x)):
            V += x[i] * system.P[i][j] * x[j]
    print('_' * 50)
    print("Verifying ROA using quadratic Lyapunov function:")
    print("x^TPx = ", V)

    # Create dReal expressions for f based on the symbolic expressions
    f = [
        lyznet.utils.sympy_to_dreal(
            expr, dict(zip(system.symbolic_vars, x))
            )
        for expr in system.symbolic_f
        ]

    # for discrete-time systems, V(f(x)) is passed to the verifier
    V_f_x = None
    if system.system_type == "DiscreteDynamicalSystem":
        V_f_x = dreal.Expression(0)
        for i in range(len(f)):
            for j in range(len(f)):
                V_f_x += f[i] * system.P[i][j] * f[j]
        V_f_x = lyznet.utils.simplify_dreal_expression(V_f_x, x)

    start_time = time.time()
    c2_P = reach_verifier_dreal(system, x, V, f, c1_P, V_f_x, c_max=c_max, 
                                tol=tol, accuracy=accuracy,
                                number_of_jobs=number_of_jobs)

    if c2_P is None:
        c2_P = c1_P

    end_time = time.time()
    if c2_P > c1_P:     
        print(f"Largest level set x^T*P*x <= {c2_P} verified by reach & stay.")
    else:
        print(f"Largest level set x^T*P*x <= {c2_P} remains the same.")
    print(f"Time taken for verification: {end_time - start_time} seconds.\n")
    return c2_P


def quadratic_CM_verifier(system, c1_P, c2_P, tol=1e-4, accuracy=1e-4,
                          number_of_jobs=32):
    print('_' * 50)
    print("Verifying quadratic contraction metric...")

    config = lyznet.utils.config_dReal(number_of_jobs=number_of_jobs, tol=tol)
    xlim = system.domain
    d = len(system.symbolic_vars)

    # Create dReal variables based on the number of symbolic variables
    x = [dreal.Variable(f"x{i}") 
         for i in range(1, len(system.symbolic_vars) + 1)]
    V = dreal.Expression(0)
    for i in range(len(x)):
        for j in range(len(x)):
            V += x[i] * system.P[i][j] * x[j]

    f = [
        lyznet.utils.sympy_to_dreal(
            expr, dict(zip(system.symbolic_vars, x))
            )
        for expr in system.symbolic_f
        ]

    Df_x = [[fi.Differentiate(xi) for xi in x] for fi in f]
    Df_x_T = list(map(list, zip(*Df_x)))  # transpose of Df_x

    M_prod1 = lyznet.utils.matrix_multiply_dreal(Df_x_T, system.P) 
    M_prod2 = list(map(list, zip(*M_prod1))) 

    CM_derivative = [[M_prod1[i][j] + M_prod2[i][j] 
                      for j in range(d)] for i in range(d)]

    # verifying negative definiteness of CM_derivative
    constraints = []
    for n in range(1, d + 1):
        sub_matrix = [[-CM_derivative[i][j] for j in range(n)] for i in range(n)]
        det_sub_matrix = lyznet.utils.compute_determinant_dreal(sub_matrix)
        constraints.append(det_sub_matrix >= tol)

    negative_definiteness = dreal.And(*constraints)

    def Check_contraction(c2):    
        x_bound = lyznet.utils.get_bound(x, xlim, V, c2_V=c2)
        condition = dreal.logical_imply(x_bound, negative_definiteness)    
        result = dreal.CheckSatisfiability(
            dreal.logical_not(condition), config
            )
        return result

    start_time = time.time()
    c_best = lyznet.utils.bisection_glb(Check_contraction, 
                                        c1_P, c2_P, accuracy)
    end_time = time.time()

    print(f"Largest level set x^T*P*x <= {c_best} verified " 
          "for P to be a contraction metric.")
    print(f"Time taken for verification: " 
          f"{end_time - start_time} seconds.\n")

    return c_best

def quadratic_clf_verifier(system, c1_P, c_max=100, tol=1e-4, accuracy=1e-5,
                           number_of_jobs=32):
    if system.P is None:
            return
    config = lyznet.utils.config_dReal(number_of_jobs=number_of_jobs, tol=tol)
    xlim = system.domain

    x = [
        dreal.Variable(f"x{i}") 
        for i in range(1, len(system.symbolic_vars) + 1)
        ]

    V = dreal.Expression(0)
    for i in range(len(x)):
        for j in range(len(x)):
            V += x[i] * system.P[i][j] * x[j]
    print('_' * 50)
    print("Verifying quadratic control Lyapunov function:")
    print("x^TPx = ", V)

    start_time = time.time()
    c2_P = clf_reach_verifier_dreal(system, x, V, c1_P, c_max=c_max, 
                                    tol=tol, accuracy=accuracy,
                                    number_of_jobs=number_of_jobs)    
    end_time = time.time()

    if c2_P is None:
        c2_P = c1_P
    if c2_P > c1_P:     
        print(f"Largest level set x^T*P*x <= {c2_P} verified for CLF condition.")
    else:
        print(f"Largest level set x^T*P*x <= {c2_P} remains the same.")
    print(f"Time taken for verification: {end_time - start_time} seconds.\n")
    
    return c2_P
