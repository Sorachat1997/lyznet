import sympy 
import lyznet

lyznet.utils.set_random_seed()

# Define dynamics
mu = 3.0
x1, x2 = sympy.symbols('x1 x2')
f_vdp = [-x2, x1 - mu * (1 - x1**2) * x2]
domain_vdp = [[-3.0, 3.0], [-6.0, 6.0]]
sys_name = f"ex1b_van_der_pol_mu_{mu}.py"
vdp_system = lyznet.DynamicalSystem(f_vdp, domain_vdp, sys_name)

print("System dynamics: x' = ", vdp_system.symbolic_f)
print("Domain: ", vdp_system.domain)

# Call the local stability verifier
c1_P = lyznet.local_stability_verifier(vdp_system)
# Call the quadratic verifier
c2_P = lyznet.quadratic_reach_verifier(vdp_system, c1_P)

# Generate data (needed for data-augmented learner)
data = lyznet.generate_data(vdp_system, n_samples=3000)

# Call the neural lyapunov learner
net, model_path = lyznet.neural_learner(
    vdp_system, data=data, lr=0.001, layer=2, width=30, num_colloc_pts=300000, 
    max_epoch=20, loss_mode="Zubov")

# Call the neural lyapunov verifier
c1_V, c2_V = lyznet.neural_verifier(vdp_system, net, c2_P)

# Compare verified ROA with SOS
def sos_V(x1, x2):
    return (-3.45170890476e-09*x1 + 1.18207436789e-09*x2 + 0.724166734819*x1**2 
            - 1.76558340904e-09*x1**2*x2 - 0.443120764123*x1*x2 
            + 0.100535553877*x2**2 - 0.0827925966603*x1**3*x2 
            + 0.113707192696*x1**2*x2**2 + 2.66977190991e-09*x1**3 
            - 1.96231083293e-09*x1*x2**2 + 2.46445852241e-09*x2**3 
            - 0.258471270203*x1**4 - 7.15465083743e-05*x1*x2**3 
            - 0.00868754707599*x2**4 - 7.06729359872e-10*x1**5 
            + 4.71809154412e-10*x1**4*x2 + 1.78626846573e-09*x1**3*x2**2 
            - 3.85499453562e-09*x1**2*x2**3 + 2.13640045899e-09*x1*x2**4
            - 6.48146577731e-10*x2**5 + 0.042169106804*x1**6 
            + 0.0659159674928*x1**5*x2 + 0.00064555582031*x1**4*x2**2
            - 0.0243141499204*x1**3*x2**3 + 0.0191263911108*x1**2*x2**4 
            - 0.00770278152905*x1*x2**5 + 0.00153689109462*x2**6)


sos_V_sympy = sos_V(x1, x2)
c1_SOS, c2_SOS = lyznet.sos_reach_verifier(vdp_system, sos_V_sympy, c2_P)

lyznet.plot_V(vdp_system, net, model_path, [sos_V], [[c2_SOS]], 
              c2_V=c2_V, c2_P=c2_P, phase_portrait=True)

test_data = lyznet.generate_data(vdp_system, n_samples=90000)
volume_percent = lyznet.utils.test_volume(vdp_system, net, c2_V, test_data)
sos_volume_percent = lyznet.utils.test_volume_sos(vdp_system, sos_V, c2_SOS, 
                                                  test_data)
