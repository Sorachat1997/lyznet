import time 
import dreal 
import lyznet


def sos_reach_verifier(system, sos_V_sympy, c2_P, tol=1e-4, accuracy=1e-2,
                       number_of_jobs=32, c_max=1):
    config = lyznet.utils.config_dReal(number_of_jobs=number_of_jobs, tol=tol)
    xlim = system.domain

    print('_' * 50)
    print("Verifying SOS Lyapunov function:")

    # Create dReal variables based on the number of symbolic variables
    x = [dreal.Variable(f"x{i}") 
         for i in range(1, len(system.symbolic_vars) + 1)]

    f = [
        lyznet.utils.sympy_to_dreal(
            expr, dict(zip(system.symbolic_vars, x))
            )
        for expr in system.symbolic_f
        ]

    sos_V_dreal = lyznet.utils.sympy_to_dreal(
                    sos_V_sympy, dict(zip(system.symbolic_vars, x))
                    )
    print("V = ", sos_V_dreal.Expand())

    lie_derivative_of_V = dreal.Expression(0)
    for i in range(len(x)):
        lie_derivative_of_V += f[i] * sos_V_dreal.Differentiate(x[i])

    quad_V = dreal.Expression(0)
    for i in range(len(x)):
        for j in range(len(x)):
            quad_V += x[i] * system.P[i][j] * x[j]
    target = quad_V <= c2_P

    start_time = time.time()

    def Check_inclusion(c1):
        x_bound = lyznet.utils.get_bound(x, xlim, sos_V_dreal, c2_V=c1)
        condition = dreal.logical_imply(x_bound, target)
        return dreal.CheckSatisfiability(dreal.logical_not(condition), config)
 
    c1_V = lyznet.utils.bisection_glb(Check_inclusion, 0, 1, accuracy)
    print(f"Verified V<={c1_V} is contained in x^TPx<={c2_P}.")

    c2_V = lyznet.reach_verifier_dreal(system, x, sos_V_dreal, f, c1_V, 
                                       c_max=c_max, tol=tol, accuracy=accuracy,
                                       number_of_jobs=number_of_jobs)
    print(f"Verified V<={c2_V} will reach V<={c1_V} and hence x^TPx<={c2_P}.")
    end_time = time.time()
    print(f"Time taken for verifying SOS Lyapunov function of {system.name}: " 
          f"{end_time - start_time} seconds.\n")

    return c1_V, c2_V
