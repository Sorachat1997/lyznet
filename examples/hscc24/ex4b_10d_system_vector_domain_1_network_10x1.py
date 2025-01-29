import sympy as sp
import lyznet

x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = sp.symbols('x1:11')

f = [
    -x1 + 0.5 * x2 - 0.1 * x9**2,
    -0.5 * x1 - x2,
    -x3 + 0.5 * x4 - 0.1 * x1**2,
    -0.5 * x3 - x4,
    -x5 + 0.5 * x6 + 0.1 * x7**2,
    -0.5 * x5 - x6,
    -x7 + 0.5 * x8,
    -0.5 * x7 - x8,
    -x9 + 0.5 * x10,
    -0.5 * x9 - x10 + 0.1 * x2**2
    ]

domain = [[-1, 1]] * 10
sys_name = "ex4b_10d_system_vector"
system = lyznet.DynamicalSystem(f, domain, sys_name)

print("System dynamics: x' = ", system.symbolic_f)
print("Domain: ", system.domain)

topology = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]

subsystems, interconnections = lyznet.decompose_system(system, topology)

c_P = lyznet.subsys_quadratic_verifier(subsystems)
c1_P = lyznet.verify_network_local_stability(subsystems, interconnections, c_P)
