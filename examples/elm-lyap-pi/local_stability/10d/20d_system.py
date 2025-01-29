import sympy as sp
import lyznet

lyznet.utils.set_random_seed()

# Define symbols for all 30 variables
x = sp.symbols('x1:21')

# Define the 30-dimensional system as three decoupled 10-dimensional systems
f = [
    # First system (x1 to x10)
    -x[0] + 0.5 * x[1] - 0.1 * x[8]**2,
    -0.5 * x[0] - x[1],
    -x[2] + 0.5 * x[3] - 0.1 * x[0]**2,
    -0.5 * x[2] - x[3],
    -x[4] + 0.5 * x[5] + 0.1 * x[6]**2,
    -0.5 * x[4] - x[5],
    -x[6] + 0.5 * x[7],
    -0.5 * x[6] - x[7],
    -x[8] + 0.5 * x[9],
    -0.5 * x[8] - x[9] + 0.1 * x[1]**2,
    
    # Second system (x11 to x20)
    -x[10] + 0.5 * x[11] - 0.1 * x[18]**2,
    -0.5 * x[10] - x[11],
    -x[12] + 0.5 * x[13] - 0.1 * x[10]**2,
    -0.5 * x[12] - x[13],
    -x[14] + 0.5 * x[15] + 0.1 * x[16]**2,
    -0.5 * x[14] - x[15],
    -x[16] + 0.5 * x[17],
    -0.5 * x[16] - x[17],
    -x[18] + 0.5 * x[19],
    -0.5 * x[18] - x[19] + 0.1 * x[11]**2
]

# Define the domain for all 30 variables
domain = [[-0.1, 0.1]] * 20

# Name for the system
sys_name = "30d_system_decoupled"

system = lyznet.DynamicalSystem(f, domain, sys_name)

print("System dynamics: x' = ", system.symbolic_f)

# c1_P = lyznet.local_stability_verifier(system)


W, b, beta, model_path = lyznet.numpy_elm_learner(
    system, num_hidden_units=6400, num_colloc_pts=12000,
    lambda_reg=0.0, test=True
    )


lyznet.plot_V(system, elm_model=[W, b, beta], model_path=model_path, 
              phase_portrait=True)

# net, model_path = lyznet.neural_learner(system, lr=0.001, 
#                                         num_colloc_pts=300000, max_epoch=20, 
#                                         layer=1, width=10, 
#                                         loss_mode="Lyapunov_PDE")
