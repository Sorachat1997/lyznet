import sympy as sp
import numpy as np
import torch
import lyznet

# Symbolic variables for the F-8 longitudinal dynamics
x1, x2, x3 = sp.symbols('x1 x2 x3')

# Drift dynamics f(x)
f_expr = sp.Matrix([
    -0.878 * x1 + x3
    - x1**2 * x3 - 0.0896 * x1 * x3
    - 0.019 * x2**2 + 0.473 * x1**2
    + 3.813 * x1**3,
    x3,
    -4.209 * x1 - 0.396 * x3
    - 0.408 * x1**2 - 2.137 * x1**3,
])

# Control effectiveness g(x)
g_expr = sp.Matrix([[-0.216], [0.0], [-20.991]])

# Baseline control law u0(x)
u0_expr = (
    -0.60357 * x1 + 0.51918 * x2 + 2.6414 * x3
    + 1.1853 * x1**2 - 0.00390102 * x2**2 + 0.15242 * x3**2
    - 0.2542 * x1 * x2 - 0.58084 * x1 * x3 + 0.15914 * x2 * x3
    + 8.0966 * x3 + 0.07795 * x3**3 + 0.25808 * x1**3
    - 1.9417 * x2**2 - 4.2947 * x2 * x3
    + 0.5879 * x1 * x2**2 + 0.34048 * x1 * x3**2
    + 0.077199 * x2**2 * x3 - 0.13242 * x3**2
    + 0.69357 * x1 * x2 * x3
)

u0_sym = sp.Matrix([u0_expr])

# Construct closed-loop dynamics f_u(x) = f(x) + g(x)u0(x)
f_u = lyznet.get_closed_loop_f_expr(f_expr, g_expr, u0_sym, (x1, x2, x3))

# Domain for the state variables
domain = [[-4, 4], [-4, 4], [-4, 4]]

# Closed-loop system instance
closed_loop = lyznet.DynamicalSystem(f_u, domain, "F8_closed_loop", symbolic_vars=(x1, x2, x3))

# Quadratic Lyapunov-based estimate of the region of attraction
c1_P = lyznet.local_stability_verifier(closed_loop)
c2_P = lyznet.quadratic_reach_verifier(closed_loop, c1_P)
print(f"Quadratic RoA level: {c2_P:.4g}")

# Train a neural Lyapunov function via Zubov's PDE
net, model_path = lyznet.neural_learner(
    closed_loop,
    data=None,
    lr=0.001,
    layer=2,
    width=30,
    num_colloc_pts=300000,
    max_epoch=20,
    loss_mode="Zubov",
)

# Verify the neural Lyapunov function
c1_V, c2_V = lyznet.neural_verifier(closed_loop, net, c2_P)
print(f"Neural RoA level: {c2_V:.4g}")

# Plot Lyapunov level sets and phase portrait
lyznet.plot_V(
    closed_loop,
    net,
    model_path,
    c2_V=c2_V,
    c2_P=c2_P,
    phase_portrait=True,
)

# Estimate volume coverage of the verified region
test_data = lyznet.generate_data(closed_loop, n_samples=10000)
volume_percent = lyznet.utils.test_volume(closed_loop, net, c2_V, test_data)
print(f"Verified ROA covers {volume_percent * 100:.2f}% of sampled volume.")
