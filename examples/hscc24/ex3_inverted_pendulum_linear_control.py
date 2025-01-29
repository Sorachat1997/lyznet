import sympy 
import lyznet

x1, x2 = sympy.symbols('x1 x2')
k = [4.4142, 2.3163]
f = [x2, sympy.sin(x1) - x2 - (k[0]*x1 + k[1]*x2)]
domain = [[-4.0, 4.0], [-10.0, 10.0]]
sys_name = "ex3_pendulum_linear_control"
system = lyznet.DynamicalSystem(f, domain, sys_name)

print("System dynamics: x' = ", system.symbolic_f)
print("Domain: ", system.domain)

# Call the local stability verifier
c1_P = lyznet.local_stability_verifier(system)
