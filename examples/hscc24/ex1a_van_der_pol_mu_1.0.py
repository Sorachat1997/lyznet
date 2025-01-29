import sympy 
import lyznet

lyznet.utils.set_random_seed()

# Define dynamics
mu = 1.0
x1, x2 = sympy.symbols('x1 x2')
f_vdp = [-x2, x1 - mu * (1 - x1**2) * x2]
domain_vdp = [[-2.5, 2.5], [-3.5, 3.5]]
sys_name = f"ex1a_van_der_pol_mu_{mu}.py"
vdp_system = lyznet.DynamicalSystem(f_vdp, domain_vdp, sys_name)

print("System dynamics: x' = ", vdp_system.symbolic_f)
print("Domain: ", vdp_system.domain)

# Call the local stability verifier
c1_P = lyznet.local_stability_verifier(vdp_system)
# Call the quadratic verifier
c2_P = lyznet.quadratic_reach_verifier(vdp_system, c1_P)

# # Generate data (needed for data-augmented learner)
data = lyznet.generate_data(vdp_system, n_samples=3000)

# # Call the neural lyapunov learner
net, model_path = lyznet.neural_learner(vdp_system, data=data, lr=0.001, 
                                        layer=2, width=30, 
                                        num_colloc_pts=300000, max_epoch=20,
                                        loss_mode="Zubov")

# Call the neural lyapunov verifier
c1_V, c2_V = lyznet.neural_verifier(vdp_system, net, c2_P)

# Compare verified ROA with SOS
def sos_V(x1, x2):
    return (3.61548500567e-05*x1 - 7.77553365207e-05*x2 + 0.395600610126*x1**2
            + 0.000179981896*x1**2*x2 - 0.317356182214*x1*x2
            + 0.268601490195*x2**2+0.0535151077505*x1**3*x2
            + 0.0181249699127*x1**2*x2**2 + 1.94145089767e-05*x1**3
            - 4.21804163477e-06*x1*x2**2 - 0.000278084011968*x2**3
            - 0.0703413857135*x1**4 - 0.00115556235815*x1*x2**3
            - 0.013387240107*x2**4 - 2.83127701242e-05*x1**5
            - 8.20599685031e-05*x1**4*x2 - 2.10810947625e-05*x1**3*x2**2
            + 0.000114514291267*x1**2*x2**3 - 6.97300263371e-05*x1*x2**4
            + 4.91479519387e-05*x2**5 + 0.0091235157231*x1**6
            + 0.0074425593303*x1**5*x2 - 0.00483129236538*x1**4*x2**2 
            - 0.00499315837856*x1**3*x2**3 + 0.00324105218028*x1**2*x2**4
            - 0.00147229445287*x1*x2**5 + 0.000695094927905*x2**6)


sos_V_sympy = sos_V(x1, x2)
c1_SOS, c2_SOS = lyznet.sos_reach_verifier(vdp_system, sos_V_sympy, c2_P)

lyznet.plot_V(vdp_system, net, model_path, 
              V_list=[sos_V], c_lists=[[c2_SOS]], c2_V=c2_V, c2_P=c2_P,
              phase_portrait=True)

test_data = lyznet.generate_data(vdp_system, n_samples=90000)
volume_percent = lyznet.utils.test_volume(vdp_system, net, c2_V, test_data)
sos_volume_percent = lyznet.utils.test_volume_sos(vdp_system, sos_V, c2_SOS, 
                                                  test_data)
