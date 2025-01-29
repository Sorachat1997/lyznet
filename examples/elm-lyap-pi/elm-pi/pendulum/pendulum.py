import sympy as sp
import lyznet

lyznet.utils.set_random_seed(123)

x1, x2 = sp.symbols('x1 x2')
symbolic_vars = (x1, x2)

f = sp.Matrix([x2, 19.6*sp.sin(x1) - 4.0*x2])
g = sp.Matrix([0, 40.0])

domain = [[-1, 1]] * 2
sys_name = f"pendulum_elm_pi_{domain[0]}"

R = 2.0*sp.eye(1)

initial_u = sp.Matrix([-(1/40)*x1 - (19.6/40)*sp.sin(x1)])

system = lyznet.ControlAffineSystem(f, g, domain, sys_name, R=R)

lyznet.elm_pi(system, initial_u=initial_u, num_of_iters=10, width=200, 
              num_colloc_pts=800, final_plot=True, plot_each_iteration=True)
