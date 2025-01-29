import sympy 
import lyznet

lyznet.utils.set_random_seed()

delta = 3.141592653589793/3
x1, x2 = sympy.symbols('x1 x2')
f = [x2, -0.5*x2 - (sympy.sin(x1+delta)-sympy.sin(delta))]
domain_vdp = [[-2.0, 3.0], [-3.0, 1.5]]
sys_name = "ex2_two_machine_power"
system = lyznet.DynamicalSystem(f, domain_vdp, sys_name)

print("System dynamics: x' = ", system.symbolic_f)
print("Domain: ", system.domain)

# Call the local stability verifier
c1_P = lyznet.local_stability_verifier(system)
# Call the quadratic verifier
c2_P = lyznet.quadratic_reach_verifier(system, c1_P)

# Generate data (needed for data-augmented learner)
data = lyznet.generate_data(system, n_samples=3000)

# Call the neural lyapunov learner
net, model_path = lyznet.neural_learner(system, data=data, lr=0.001, layer=2, 
                                        width=30, num_colloc_pts=300000, 
                                        max_epoch=20, loss_mode="Zubov")


# Call the neural lyapunov verifier
c1_V, c2_V = lyznet.neural_verifier(system, net, c2_P)

def sos_V(x1, x2):
    return (
        -2.92714753486e-06 * x1 + 3.57112799075e-05 * x2 
        + 0.224014657569 * x1**2 + 0.239486001719 * x1**3 
        + 0.383702288626 * x1 * x2 + 0.647698474548 * x2**2
        + 1.61830590112 * x1**4 + 1.30360629196 * x1**2 * x2 
        + 3.43974516993 * x1**3 * x2 + 1.24138244957 * x1 * x2**2 
        + 1.45349892327 * x2**3 + 5.26939565255 * x1**2 * x2**2
        + 0.937247722722 * x1 * x2**3 + 1.70047975291 * x2**4 
        - 2.11897969762 * x1**5 + 0.813028567437 * x1**4 * x2 
        - 6.42684925215 * x1**3 * x2**2 - 9.40985592626 * x1**2 * x2**3 
        + 1.35209051643 * x1 * x2**4 - 4.16794817141 * x2**5
        + 1.20703759055 * x1**6 - 2.6590919776 * x1**5 * x2 
        + 0.502922432246 * x1**4 * x2**2 + 5.25089806981 * x1**3 * x2**3 
        + 9.61728957998 * x1**2 * x2**4 - 2.66463321768 * x1 * x2**5 
        + 2.9122931959 * x2**6
    )


sos_V_sympy = sos_V(x1, x2)
c1_SOS, c2_SOS = lyznet.sos_reach_verifier(system, sos_V_sympy, c2_P)

lyznet.plot_V(system, net, model_path, 
              V_list=[sos_V], c_lists=[[c2_SOS]], c2_V=c2_V, c2_P=c2_P,
              phase_portrait=True)

test_data = lyznet.generate_data(system, n_samples=90000)
volume_percent = lyznet.utils.test_volume(system, net, c2_V, test_data)
sos_volume_percent = lyznet.utils.test_volume_sos(system, sos_V, c2_SOS, 
                                                  test_data)
