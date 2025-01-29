import sympy 
import lyznet

lyznet.utils.set_random_seed(seed=123)

x1, x2 = sympy.symbols('x1 x2')
f = [x2, - sympy.sin(x1) - 0.1*x2]
domain = [[-3.5, 3.5], [-3.0, 3.0]]
sys_name = "elm_simple_pendulum_roa_zubov"
system = lyznet.DynamicalSystem(f, domain, sys_name)

print("System dynamics: x' = ", system.symbolic_f)

c1_P = lyznet.local_stability_verifier(system)
c2_P = lyznet.quadratic_reach_verifier(system, c1_P)

# data = lyznet.generate_data(system, n_samples=3000, transform="exp", v_max=1000)

W, b, beta, model_path = lyznet.numpy_elm_learner(
    system, num_hidden_units=100, num_colloc_pts=3000,
    # data=data, 
    loss_mode="Zubov", mu=0.02, 
    lambda_reg=0.1
    )

# c1_V, c2_V = lyznet.numpy_elm_verifier(system, W, b, beta, c2_P)

# # m=100
# c1_V = 0.42724609375 
# c2_V = 0.8648216724395752 

# m=200
c1_V = 0.5859375
c2_V = 0.9621381759643555


def sos_V(x1, x2):
    return (
        1.15551292992e-08 * x1 - 2.02890083799e-08 * x2 + 4.62709345897e-08 * x1**3
        + 1.49421428747 * x1**2 + 0.22545961041 * x1 * x2 + 1.63427196577 * x2**2
        - 0.819116270593 * x1**4 - 0.21723010504 * x1**3 * x2 - 1.20354641279e-07 * x1**2 * x2
        - 2.73895166255e-08 * x1 * x2**2 - 1.0893364881e-07 * x2**3 - 1.6820513799 * x1**2 * x2**2
        - 0.232299026707 * x1 * x2**3 - 0.916417665086 * x2**4 - 1.79294476203e-08 * x1**5
        + 5.79425666337e-08 * x1**4 * x2 - 6.3466569542e-09 * x1**3 * x2**2
        + 1.28001168765e-07 * x1**2 * x2**3 + 2.19687574495e-08 * x1 * x2**4
        + 6.66061613649e-08 * x2**5 + 0.160180339511 * x1**6 + 0.0560936542448 * x1**5 * x2
        + 0.478702259915 * x1**4 * x2**2 + 0.111552252399 * x1**3 * x2**3
        + 0.48509034403 * x1**2 * x2**4 + 0.0605216996554 * x1 * x2**5 + 0.1761658345 * x2**6
    )


sos_V_sympy = sos_V(x1, x2)
# c1_SOS, c2_SOS = lyznet.sos_reach_verifier(
#     system, sos_V_sympy, c2_P)

lyznet.plot_V(system, elm_model=[W, b, beta], model_path=model_path, 
              V_list=[sos_V], c_lists=[[0.99]], 
              c1_V=c1_V, c2_V=c2_V, c1_P=c1_P, c2_P=c2_P, phase_portrait=True)

test_data = lyznet.generate_data(system, n_samples=30000, transform="exp", 
                                 v_max=1000)
volume_percent = lyznet.utils.test_volume_elm(system, W, b, beta, c2_V, 
                                              test_data)
sos_volume_percent = lyznet.utils.test_volume_sos(system, sos_V, 0.99, 
                                                  test_data)
