import sympy as sp
import numpy as np
import lyznet
import math

lyznet.utils.set_random_seed(seed=123)

x1, x2 = sp.symbols('x1 x2')

theta = 1e-2
sine = math.sin(theta)
cosine = math.cos(theta)

# Substitute control law into dynamics
f = [
    (- 2*sine*x1*(x1**4+2*x1**2*x2**2-x2**4)
     - 2*cosine*x2*(-x1**4+2*x1**2*x2**2+x2**4)),
    (2*cosine*x1*(x1**4+2*x1**2*x2**2-x2**4)
     - 2*sine*x2*(-x1**4+2*x1**2*x2**2+x2**4))
]

domain = [[-100, 100]] * 2
sys_name = f"deg_5_homo_poly_{theta}"
system = lyznet.DynamicalSystem(f, domain, sys_name)

print("System dynamics: x' = ", system.symbolic_f)
print("Eigenvalues of linearization: ", np.linalg.eigvals(system.A))

c1_P = lyznet.local_stability_verifier(system)

net, model_path = lyznet.neural_learner(
    system, lr=0.001, num_colloc_pts=300000, max_epoch=100, layer=1, width=10, 
    loss_mode="Homo_Lyapunov", net_type="Homo")


c1_V, c2_V = lyznet.neural_verifier(system, net, 
                                    net_type="Homo", verifier="Homo")


lyznet.plot_V(system, net, model_path, 
              phase_portrait=True, c1_V=10, c2_V=100)
