import sympy as sp
import numpy as np
import lyznet

lyznet.utils.set_random_seed(seed=417)

x1, x2 = sp.symbols('x1 x2')

f = [
    (-1/3*(8*x1**11 + 22*x1**10*x2 + 93*x1**9*x2**2 + 126*x1**8*x2**3 
     + 483*x1**7*x2**4 + 186*x1**6*x2**5 + 445*x1**5*x2**6 
     - 1189*x1**4*x2**7 + 234*x1**3*x2**8 - 713*x1**2*x2**9 
     + 1772*x1*x2**10 - 414*x2**11)*sp.sqrt(2*x1**6 + 8*x1**5*x2 
     + 22*x1**4*x2**2 + 18*x1**3*x2**3 + 26*x1**2*x2**4 
     - 16*x1*x2**5 + 13*x2**6)/(2*x1**4 - 4*x1**3*x2 + 5*x1**2*x2**2 
     + 22*x1*x2**3 + 14*x2**4)),

    (1/3*(6*x1**11 + 16*x1**10*x2 + 74*x1**9*x2**2 
     + 123*x1**8*x2**3 + 180*x1**7*x2**4 - 267*x1**6*x2**5 
     - 1083*x1**5*x2**6 - 1729*x1**4*x2**7 - 1580*x1**3*x2**8 
     + 687*x1**2*x2**9 + 50*x1*x2**10 - 194*x2**11)*sp.sqrt(2*x1**6 
     + 8*x1**5*x2 + 22*x1**4*x2**2 + 18*x1**3*x2**3 + 26*x1**2*x2**4 
     - 16*x1*x2**5 + 13*x2**6)/(2*x1**4 - 4*x1**3*x2 + 5*x1**2*x2**2 
     + 22*x1*x2**3 + 14*x2**4))
]

domain = [[-1, 1]] * 2
# domain = [[-100, 100]] * 2

sys_name = "irrational_nn_lyap"
system = lyznet.DynamicalSystem(f, domain, sys_name)

print("System dynamics: x' = ", system.symbolic_f)
print("Eigenvalues of linearization: ", np.linalg.eigvals(system.A))

c1_P = lyznet.local_stability_verifier(system)

net, model_path = lyznet.neural_learner(
    system, lr=0.001, num_colloc_pts=300000, max_epoch=5, layer=1, width=1, 
    loss_mode="Homo_Lyapunov", net_type="Homo")


c1_V, c2_V = lyznet.neural_verifier(system, net, 
                                    net_type="Homo", verifier="Homo")


lyznet.plot_V(system, net, model_path, c1_V=0.1, c2_V=0.7, phase_portrait=True)
# lyznet.plot_V(system, net, model_path, c1_V=10, c2_V=70, phase_portrait=True)
