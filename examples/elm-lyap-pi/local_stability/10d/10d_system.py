import sympy as sp
import lyznet

lyznet.utils.set_random_seed()

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

domain = [[-0.1, 0.1]] * 10
sys_name = "10d_system"
system = lyznet.DynamicalSystem(f, domain, sys_name)

print("System dynamics: x' = ", system.symbolic_f)

# c1_P = lyznet.local_stability_verifier(system)


W, b, beta, model_path = lyznet.numpy_elm_learner(
    system, num_hidden_units=6400, num_colloc_pts=12000,
    lambda_reg=0.0, test=True
    )


lyznet.plot_V(system, elm_model=[W, b, beta], model_path=model_path, 
              phase_portrait=True)
