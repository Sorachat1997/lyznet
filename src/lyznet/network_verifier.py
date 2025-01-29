import time 

import dreal 
import lyznet
import numpy as np


def decompose_system(system, topology):
    subsystems = []
    interconnections = []

    if len(topology) == 1:
        raise ValueError("Topology contains only one subsystem! "
                         "A partition of at least two subsystems required.")
    
    subsystems = []
    interconnections = []
    
    all_elements = []
    for sublist in topology:
        for elem in sublist:
            if elem in all_elements:
                raise ValueError(f"Overlap found: node {elem} appears "
                                 "in more than one subsystem.")
            else:
                all_elements.append(elem)
    
    for group in topology:
        group = sorted(group)
        group_f = [system.symbolic_f[i-1] for i in group]
        group_domain = [system.domain[i-1] for i in group]
        
        internal_dynamics = []
        external_dynamics = []        
        for eq in group_f:
            terms = eq.as_ordered_terms()
            group_vars = [system.symbolic_vars[i-1] for i in group]

            internal_terms = []
            for term in terms:
                if all(var in group_vars for var in term.free_symbols):
                    internal_terms.append(term)

            external_terms = []
            for term in terms:
                if any(var not in group_vars for var in term.free_symbols):
                    external_terms.append(term)            

            internal_dynamics.append(sum(internal_terms))
            external_dynamics.append(sum(external_terms))
        subsystem = lyznet.DynamicalSystem(internal_dynamics, group_domain, 
                                           name=f"subsystem_{group}",
                                           symbolic_vars=group_vars)
        subsystems.append(subsystem)
        interconnections.append(external_dynamics)

    return subsystems, interconnections


def subsys_quadratic_verifier(systems):
    print("_" * 50)
    print("Verifying stability and ROA for subsystems "
          "using quadratic Lyapunov functions...")
    c_P = []
    start_time = time.time()
    for idx, subsystem in enumerate(systems):
        print(f"Processing subsystem {idx+1}...")
        c1 = lyznet.local_stability_verifier(subsystem)
        c2 = lyznet.quadratic_reach_verifier(subsystem, c1)
        c_P.append((c1, c2))
    end_time = time.time()
    exec_time = end_time - start_time
    print(f"Time taken for verifying stability and ROA for subsystems \n"
          f"using quadratic Lyapunov functions: {exec_time} seconds.\n")
    return c_P


def verify_network_local_stability(systems, interconnections, c_P, 
                                   accuracy=1e-4):
    print("_" * 50)
    print("Verifying network local stability... \n")
    num_of_subsystems = len(systems)
    gain_matrix = np.zeros((num_of_subsystems, num_of_subsystems))
    c1_P = np.array([item[0] for item in c_P])

    start_time = time.time()

    def target_stable_for_scale(scale):
        # print("-" * 50)
        # print(f"Verifying scaling factor {scale} for local stability... ")
        for i, system in enumerate(systems):
            eig_min_P = np.min(np.linalg.eigvals(systems[i].P))
            eig_max_P = np.max(np.linalg.eigvals(systems[i].P))
            rii = lyznet.local_self_gain_verifier(
                                system, c1_P[i] * scale
                            )
            if rii is None:
                print("Verification failed for scaling factor: ", scale)
                return True               
            gain_matrix[i, i] = (
                - 1 / eig_max_P 
                + 2 * rii / eig_min_P
            )

        for i, interconnect in enumerate(interconnections):
            eig_min_Pi = np.min(np.linalg.eigvals(systems[i].P))
            for k, row in enumerate(interconnect):
                # print("row: ", row)
                if row != 0:
                    terms = row.as_ordered_terms()
                    # print("terms: ", terms)
                    for term in terms:
                        # print("term: ", term)
                        # print("k: ", k)
                        for j, subsystem in enumerate(systems):
                            if j != i: 
                                eig_min_Pj = np.min(
                                    np.linalg.eigvals(subsystem.P)
                                    )
                                # print(f"subsystem_{j} symbolic_vars: ", subsystem.symbolic_vars)
                                if any(var in subsystem.symbolic_vars 
                                       for var in term.free_symbols):
                                    # print("j", j)
                                    rii, rij = lyznet.local_gain_verifier(
                                        systems[i], c1_P[i]*scale, subsystem, 
                                        c1_P[j]*scale, term, k
                                    )
                                    # rij = lyznet.local_gain_verifier(
                                    #     systems[i], c1_P[i]*scale, subsystem, 
                                    #     c1_P[j]*scale, term, k
                                    # )
                                    if rij is None and rii is not None:
                                        print("Verification failed for "
                                              "scaling factor: ", scale)
                                        return True                                       
                                    # print(f"r_{i}{j}: ", rij)
                                    gain_matrix[i, j] = rij / eig_min_Pj
                                    gain_matrix[i, i] += (
                                        2*rii + rij
                                        ) / eig_min_Pi
                                    # gain_matrix[i, i] += 2*rij / eig_min_Pi

        # np.set_printoptions(precision=3)
        # print("Local gain matrix R:\n ", gain_matrix)
        eigenvalues = np.linalg.eigvals(gain_matrix)
        stability = all(eig.real < 0 for eig in eigenvalues)
        # print("All eigenvalues of gain matrix have negative real parts?", 
        #       stability)

        verified_c1_P = c1_P * scale
        # print("Currently verified scale: ", scale)
        # print("Currently verified levels: ", verified_c1_P)
        x = gain_matrix @ verified_c1_P
        invariance = all(element < 0 for element in x)
        # print("All entries of gain matrix * scaled c1_P are negative?", 
        #       invariance)

        if stability and invariance:
            print("Verification successful for scaling factor: ", scale)
            return None  # For the bisection routine; ``None'' means successful 
        else: 
            print("Verification failed for scaling factor: ", scale)
            return True

    best_scale = lyznet.utils.bisection_glb(target_stable_for_scale, 0, 1, 
                                            accuracy)
    end_time = time.time()
    exec_time = end_time - start_time

    print("\n")
    print("Largest scaling factor for which local stability is verified: ", 
          best_scale)    
    print("Level sets verified for network local stability:  \n", 
          c1_P * best_scale)    
    print(f"Time taken for verification: {exec_time} seconds.\n")

    return c1_P * best_scale


def local_self_gain_verifier(system, c, tol=1e-4, accuracy=1e-4):
    config = lyznet.utils.config_dReal(number_of_jobs=32, tol=tol)
    xlim = system.domain
    x = [
        dreal.Variable(f"x{i}") 
        for i in range(1, len(system.symbolic_vars) + 1)
        ]
    V = dreal.Expression(0)
    for i in range(len(x)):
        for j in range(len(x)):
            V += x[i] * system.P[i][j] * x[j]
    # Create dReal expressions for f based on the symbolic expressions
    f = [
        lyznet.utils.sympy_to_dreal(
            expr, dict(zip(system.symbolic_vars, x))
            )
        for expr in system.symbolic_f
        ]
    g = [
        f[i] - sum(system.A[i][j] * x[j] for j in range(len(x))) 
        for i in range(len(x))
        ]
    Dg = [[None for _ in range(len(x))] for _ in range(len(g))]
    for i in range(len(g)):
        for j in range(len(x)):
            Dg[i][j] = g[i].Differentiate(x[j])
    P_Dg = np.dot(system.P, Dg)
    frobenius_norm_sq = sum(sum(m_ij**2 for m_ij in row) for row in P_Dg)
    x_bound = lyznet.utils.get_bound(x, xlim, V, c2_V=c)

    def Check_self_gain(r):
        h = frobenius_norm_sq <= r**2
        condition = dreal.logical_imply(x_bound, h)
        return dreal.CheckSatisfiability(dreal.logical_not(condition), config)
    r_best = lyznet.utils.bisection_lub(Check_self_gain, 0, 0.5, accuracy)
    return r_best


def local_gain_verifier(system1, c1, system2, c2, interconnection_term, k, 
                        tol=1e-4, accuracy=1e-4):
    config = lyznet.utils.config_dReal(number_of_jobs=32, tol=tol)
    xlim = system1.domain
    ylim = system2.domain

    x = [
        dreal.Variable(f"x{i}") 
        for i in range(1, len(system1.symbolic_vars) + 1)
        ]
    y = [
        dreal.Variable(f"y{i}") 
        for i in range(1, len(system2.symbolic_vars) + 1)
        ]

    V1 = dreal.Expression(0)
    for i in range(len(x)):
        for j in range(len(x)):
            V1 += x[i] * system1.P[i][j] * x[j]

    V2 = dreal.Expression(0)
    for i in range(len(y)):
        for j in range(len(y)):
            V2 += y[i] * system2.P[i][j] * y[j]

    map_vars_to_x_y = {**dict(zip(system1.symbolic_vars, x)), 
                       **dict(zip(system2.symbolic_vars, y))}

    g_term = lyznet.utils.sympy_to_dreal(interconnection_term, 
                                         map_vars_to_x_y)

    g = [dreal.Expression(0.0) for _ in range(len(system1.symbolic_vars))]
    g[k] = g_term

    Dg_x = [[None for _ in range(len(x))] for _ in range(len(g))]
    for i in range(len(g)):
        for j in range(len(x)):
            Dg_x[i][j] = g[i].Differentiate(x[j])
    P_Dg_x = np.dot(system1.P, Dg_x)

    Dg_y = [[None for _ in range(len(y))] for _ in range(len(g))]
    for i in range(len(g)):
        for j in range(len(y)):
            Dg_y[i][j] = g[i].Differentiate(y[j])
    P_Dg_y = np.dot(system1.P, Dg_y)

    frobenius_norm_sq_P_Dg_x = sum(sum(m_ij**2 for m_ij in row) 
                                   for row in P_Dg_x)

    frobenius_norm_sq_P_Dg_y = sum(sum(m_ij**2 for m_ij in row) 
                                   for row in P_Dg_y)

    x_bound = lyznet.utils.get_bound(x, xlim, V1, c2_V=c1)
    y_bound = lyznet.utils.get_bound(y, ylim, V2, c2_V=c2)

    # frobenius_norm_sq_P_Dg = (frobenius_norm_sq_P_Dg_y 
    #                           + frobenius_norm_sq_P_Dg_x)

    # def Check_local_self_gain(r):
    #     h = frobenius_norm_sq_P_Dg <= r**2
    #     condition = dreal.logical_imply(dreal.logical_and(x_bound, y_bound), h)
    #     # print("condition: ", condition)
    #     return dreal.CheckSatisfiability(dreal.logical_not(condition), config)

    # r = lyznet.utils.bisection_lub(Check_local_self_gain, 0, 100, accuracy)

    # return r

    def Check_local_self_gain(r):
        h = frobenius_norm_sq_P_Dg_x <= r**2
        condition = dreal.logical_imply(dreal.logical_and(x_bound, y_bound), h)
        return dreal.CheckSatisfiability(dreal.logical_not(condition), config)

    r1 = lyznet.utils.bisection_lub(Check_local_self_gain, 0, 100, accuracy)
    
    def Check_local_gain(r):
        h = frobenius_norm_sq_P_Dg_y <= r**2
        condition = dreal.logical_imply(dreal.logical_and(x_bound, y_bound), h)
        return dreal.CheckSatisfiability(dreal.logical_not(condition), config)

    r2 = lyznet.utils.bisection_lub(Check_local_gain, 0, 100, accuracy)

    return r1, r2


def quadratic_self_gain_verifier(system, c2_V, total_gain, 
                                 tol=1e-4, accuracy=1e-4):
    epsilon = -1e-4
    config = lyznet.utils.config_dReal(number_of_jobs=32, tol=tol)
    xlim = system.domain
    x = [
        dreal.Variable(f"x{i}") 
        for i in range(1, len(system.symbolic_vars) + 1)
        ]
    f = [
        lyznet.utils.sympy_to_dreal(
            expr, dict(zip(system.symbolic_vars, x))
            )
        for expr in system.symbolic_f
        ]
    V = dreal.Expression(0)
    for i in range(len(x)):
        for j in range(len(x)):
            V += x[i] * system.P[i][j] * x[j]
    lie_derivative_of_V = dreal.Expression(0)
    for i in range(len(x)):
        lie_derivative_of_V += f[i] * V.Differentiate(x[i])

    def Check_self_gain(c):
        x_bound = lyznet.utils.get_bound(x, xlim, V, c1_V=c, c2_V=c2_V)
        condition = dreal.logical_imply(
            x_bound, lie_derivative_of_V <= -total_gain + epsilon
            )
        return dreal.CheckSatisfiability(dreal.logical_not(condition), config)
    best_c = lyznet.utils.bisection_lub(Check_self_gain, 0, c2_V, accuracy)
    return best_c


def quadratic_gain_verifier(system1, c1, system2, c2, interconnection_term, k, 
                            tol=1e-4, accuracy=1e-4):
    xlim = system1.domain
    ylim = system2.domain
    config = lyznet.utils.config_dReal(number_of_jobs=32, tol=tol)

    x = [
        dreal.Variable(f"x{i}") 
        for i in range(1, len(system1.symbolic_vars) + 1)
        ]
    y = [
        dreal.Variable(f"y{i}") 
        for i in range(1, len(system2.symbolic_vars) + 1)
        ]

    V1 = dreal.Expression(0)
    for i in range(len(x)):
        for j in range(len(x)):
            V1 += x[i] * system1.P[i][j] * x[j]

    V2 = dreal.Expression(0)
    for i in range(len(y)):
        for j in range(len(y)):
            V2 += y[i] * system2.P[i][j] * y[j]

    x_bound = lyznet.utils.get_bound(x, xlim, V1, c2_V=c1)
    y_bound = lyznet.utils.get_bound(y, ylim, V2, c2_V=c2)

    map_vars_to_x_y = {**dict(zip(system1.symbolic_vars, x)), 
                       **dict(zip(system2.symbolic_vars, y))}

    term_dreal = lyznet.utils.sympy_to_dreal(
                    interconnection_term, map_vars_to_x_y
                    )

    def Check_upper_bound(r):
        condition = dreal.logical_imply(
            dreal.logical_and(x_bound, y_bound), 
            V1.Differentiate(x[k]) * term_dreal <= r
            )
        return dreal.CheckSatisfiability(dreal.logical_not(condition), config)

    best_r = lyznet.utils.bisection_lub(Check_upper_bound, 0, 100, accuracy)
    return best_r


def verify_network_quadratic_reach(systems, interconnections, c1_P, c_P): 
    print("_" * 50)
    print("Verifying reachability using quadratic Lyapunov functions...\n")
    num_of_subsystems = len(systems)
    # gain_matrix = [[0.0 for _ in range(num_of_subsystems)] 
    #                for _ in range(num_of_subsystems)]
    total_gains = [0.0 for _ in range(len(systems))]

    c2_P = np.array([item[1] for item in c_P])
    target_levels = c1_P
    
    MAX_ITERATIONS = 20
    start_time = time.time()

    def cannot_reach_target(scale):
        # print(f"Verifying reachability for scaling factor:  {scale}")
        c2_V = c2_P * scale
        # print("Initial levels: \n", c2_V)
        # print("Target levels: \n", target_levels)
        c2_V_updated = [True for _ in range(num_of_subsystems)]
        # print("_" * 25)
        for iteration in range(MAX_ITERATIONS):
            # print(f"Iteration: {iteration}")
            all_below_target = True
            for i, interconnect in enumerate(interconnections):
                total_gain = 0.0
                for k, row in enumerate(interconnect):
                    # print("row: ", row)
                    if row != 0:
                        terms = row.as_ordered_terms()
                        # print("terms: ", terms)
                        for term in terms:
                            # print("term: ", term)
                            # print("k: ", k)
                            for j, subsystem in enumerate(systems):
                                if j != i: 
                                    if any(var in subsystem.symbolic_vars 
                                           for var in term.free_symbols):
                                        # print("j", j)
                                        term_gain = lyznet.quadratic_gain_verifier(
                                            systems[i], c2_V[i], 
                                            subsystem, c2_V[j], term, k
                                        )
                                        if term_gain is None: 
                                            print("Verification failed "
                                                  f"for scaling factor: {scale}")
                                        else:    
                                            total_gain += term_gain
                if (total_gain == total_gains[i]) and iteration != 0:
                    # print(f"Total gain for subsystem {i+1} remained "
                    #       " the same. Skipped self gain verification.")
                    c2_V_updated[i] = False                
                else:
                    # print(f"Total gain for subsystem {i+1}: ", total_gain)
                    total_gains[i] = total_gain
                    new_c2_V = quadratic_self_gain_verifier(
                                    systems[i], c2_V[i], total_gain
                                    )   
                    if new_c2_V is not None:
                        c2_V[i] = new_c2_V
                        # print(f"Invariance of V<={c2_V[i]} verified "
                        #       f"for subsystem {i+1}.")
                        c2_V_updated[i] = True
                    elif iteration == 0:
                        # print("Invariance cannot be verified "
                        #       f"for subsystem {i+1} at initial step! ")
                        print("Verification failed "
                              f"for scaling factor: {scale}")
                        return True
                    else:
                        c2_V_updated[i] = False

                if c2_V[i] >= target_levels[i]:
                    all_below_target = False  

            # print("Updated verifiable level sets for Lyapunov functions " 
            #       f"after iteration {iteration+1}: \n", c2_V)
            # print("Subsystems updated: ", c2_V_updated)
            if all_below_target: 
                # print("Verified reachable level sets contained "
                #       f"in target set after iteration {iteration+1}. ")
                print("Verification successful "
                      f"for scaling factor: {scale}")                 
                return None
            elif not any(c2_V_updated):
                # print("Verified reachable level sets cannot be improved" 
                #       f"after iteration {iteration+1}. ")
                print("Verification failed "
                      f"for scaling factor: {scale}")
                return True

    best_scale = lyznet.utils.bisection_glb(
                    cannot_reach_target, 0, 1, accuracy=1e-4
                    )
    end_time = time.time()

    print("\n")
    print("Largest scaling factor for which reachability is verified: ", 
          best_scale)    
    print("Level sets verified for network reachability:  \n", 
          c2_P * best_scale)    
    print("Time taken for reachability verification: "
          f"{end_time - start_time} seconds.\n")
            
    return c2_P * best_scale
