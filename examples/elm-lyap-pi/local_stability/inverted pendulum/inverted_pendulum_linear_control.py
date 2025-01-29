import sympy 
import lyznet

lyznet.utils.set_random_seed(47)

x1, x2 = sympy.symbols('x1 x2')
k = [4.4142, 2.3163]
f = [x2, sympy.sin(x1) - x2 - (k[0]*x1 + k[1]*x2)]
domain = [[-16.0, 16.0]] * 2
sys_name = "ex5_pendulum_linear_control"
system = lyznet.DynamicalSystem(f, domain, sys_name)

print("System dynamics: x' = ", system.symbolic_f)
# c1_P = lyznet.local_stability_verifier(system)
# c2_P = lyznet.quadratic_reach_verifier(system, c1_P)

W, b, beta, model_path = lyznet.numpy_elm_learner(
    system, num_hidden_units=3200, num_colloc_pts=24000,
    lambda_reg=0.0, test=True
    )
