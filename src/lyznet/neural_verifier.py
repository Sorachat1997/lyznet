import time 

import numpy as np
import dreal 

import lyznet


def extract_dreal_Net(model, x):
    layers = len(model.layers) 
    weights = [layer.weight.data.cpu().numpy() for layer in model.layers]
    biases = [layer.bias.data.cpu().numpy() for layer in model.layers]

    final_layer_weight = model.final_layer.weight.data.cpu().numpy()
    final_layer_bias = model.final_layer.bias.data.cpu().numpy()

    h = x
    for i in range(layers):
        z = np.dot(h, weights[i].T) + biases[i]
        h = [dreal.tanh(z[j]) for j in range(len(weights[i]))]
    
    V_net = (np.dot(h, final_layer_weight.T) + final_layer_bias)[0]
    return V_net


def extract_dreal_CMNet(model, x):
    layers = len(model.layers) 
    weights = [layer.weight.data.cpu().numpy() for layer in model.layers]
    biases = [layer.bias.data.cpu().numpy() for layer in model.layers]

    final_layer_weight = model.final_layer.weight.data.cpu().numpy()
    final_layer_bias = model.final_layer.bias.data.cpu().numpy()

    h = x
    for i in range(layers):
        z = np.dot(h, weights[i].T) + biases[i]
        h = [dreal.tanh(z[j]) for j in range(len(weights[i]))]
    
    CM_net = (np.dot(h, final_layer_weight.T) + final_layer_bias)
    return CM_net


def extract_dreal_UNet(model, x):
    layers = len(model.layers) 
    weights = [layer.weight.data.cpu().numpy() for layer in model.layers]
    biases = [layer.bias.data.cpu().numpy() for layer in model.layers]

    final_layer_weight = model.final_layer.weight.data.cpu().numpy()
    final_layer_bias = model.final_layer.bias.data.cpu().numpy()

    h = x
    for i in range(layers):
        z = np.dot(h, weights[i].T) + biases[i]
        h = [dreal.tanh(z[j]) for j in range(len(weights[i]))]
    
    U_net = [np.dot(h, final_layer_weight[i]) + final_layer_bias[i] 
             for i in range(final_layer_weight.shape[0])]
    # U_net = np.dot(h, final_layer_weight.T) + final_layer_bias
    return U_net


def extract_dreal_PolyNet(model, x):
    layers = len(model.layers) 
    weights = [layer.weight.data.cpu().numpy() for layer in model.layers]
    biases = [layer.bias.data.cpu().numpy() for layer in model.layers]
    final_layer_weight = model.final_layer.weight.data.cpu().numpy()
    final_layer_bias = model.final_layer.bias.data.cpu().numpy()

    h = x
    for i in range(layers):
        z = np.dot(h, weights[i].T) + biases[i]
        h = [z[j]**2 for j in range(len(weights[i]))]
    
    V_net = (np.dot(h, final_layer_weight.T) + final_layer_bias)[0]
    return V_net


def extract_dreal_HomoNet(model, x):
    layers = len(model.layers) 
    weights = [layer.weight.data.cpu().numpy() for layer in model.layers]
    biases = [layer.bias.data.cpu().numpy() for layer in model.layers]

    final_layer_weight = model.final_layer.weight.data.cpu().numpy()
    final_layer_bias = model.final_layer.bias.data.cpu().numpy()

    norm = dreal.sqrt(sum(xi * xi for xi in x))
    h = [xi / norm for xi in x]
    for i in range(layers):
        z = np.dot(h, weights[i].T) + biases[i]
        h = [dreal.tanh(z[j]) for j in range(len(weights[i]))]
    
    V_net = (np.dot(h, final_layer_weight.T) + final_layer_bias)[0]
    
    input_layer_weight_norm = np.linalg.norm(weights[0])

    return V_net * (norm ** model.deg), input_layer_weight_norm


def extract_dreal_HomoUNet(model, x):
    layers = len(model.layers) 
    weights = [layer.weight.data.cpu().numpy() for layer in model.layers]
    biases = [layer.bias.data.cpu().numpy() for layer in model.layers]

    final_layer_weight = model.final_layer.weight.data.cpu().numpy()
    final_layer_bias = model.final_layer.bias.data.cpu().numpy()

    # Compute the norm of the input and normalize the inputs
    norm = dreal.sqrt(sum(xi * xi for xi in x))
    h = [xi / norm for xi in x]

    for i in range(layers):
        z = np.dot(h, weights[i].T) + biases[i]
        h = [dreal.tanh(z[j]) for j in range(len(weights[i]))]
    
    # Calculate U_net for each output neuron
    U_net = [(np.dot(h, final_layer_weight[i].T) + final_layer_bias[i]) * (norm ** model.deg)
             for i in range(final_layer_weight.shape[0])]
    
    input_layer_weight_norm = np.linalg.norm(weights[0])

    return U_net, input_layer_weight_norm


def extract_dreal_HomoPolyNet(model, x):
    layers = len(model.layers) 
    weights = [layer.weight.data.cpu().numpy() for layer in model.layers]
    biases = [layer.bias.data.cpu().numpy() for layer in model.layers]

    final_layer_weight = model.final_layer.weight.data.cpu().numpy()
    final_layer_bias = model.final_layer.bias.data.cpu().numpy()

    norm = dreal.sqrt(sum(xi * xi for xi in x))
    h = [xi / norm for xi in x]
    for i in range(layers):
        z = np.dot(h, weights[i].T) + biases[i]
        h = [z[j]**2 for j in range(len(weights[i]))]
    
    V_net = (np.dot(h, final_layer_weight.T) + final_layer_bias)[0]
    return V_net * (norm ** model.deg)


def extract_dreal_SimpleNet(model, x):
    d = len(model.initial_layers)    
    weights = [layer.weight.data.cpu().numpy() 
               for layer in model.initial_layers]
    biases = [layer.bias.data.cpu().numpy() for layer in model.initial_layers]
    
    h = []
    for i in range(d):
        xi = x[i]  
        z = xi * weights[i][0, 0] + biases[i][0]  
        h_i = dreal.tanh(z) 
        h.append(h_i)
    
    final_output = sum([h_i * h_i for h_i in h])    
    return final_output


def neural_CM_verifier(system, model, V_net=None, c2_V=None, 
                       tol=1e-4, number_of_jobs=32):
    print('_' * 50)
    
    if V_net is not None and c2_V is not None: 
        print(f"Verifying neural contraction metric on V<={c2_V}:")
    else:
        print(f"Verifying neural contraction metric on domain {system.domain}:")

    config = lyznet.utils.config_dReal(number_of_jobs=number_of_jobs, tol=tol)
    xlim = system.domain

    # Create dReal variables based on the number of symbolic variables
    x = [dreal.Variable(f"x{i}") 
         for i in range(1, len(system.symbolic_vars) + 1)]

    if V_net is not None and c2_V is not None: 
        V_learn = extract_dreal_Net(V_net, x)
        x_bound = lyznet.utils.get_bound(x, xlim, V_learn, c2_V=c2_V)
    else: 
        x_bound = lyznet.utils.get_x_bound(x, xlim)

    f = [
        lyznet.utils.sympy_to_dreal(
            expr, dict(zip(system.symbolic_vars, x))
            )
        for expr in system.symbolic_f
        ]
    CM_learn = extract_dreal_CMNet(model, x)
    # print("CM = ", CM_learn[0].Expand())

    d = len(system.symbolic_vars)
    M = [[dreal.Expression(0) for _ in range(d)] for _ in range(d)]

    k = 0
    for i in range(d):
        for j in range(i, d):
            M[i][j] = CM_learn[k]
            M[j][i] = CM_learn[k]  # Fill in symmetric terms
            k += 1

    print("Verifying positive definiteness...")

    constraints = []
    for n in range(1, d + 1):
        sub_matrix = [[M[i][j] for j in range(n)] for i in range(n)]
        det_sub_matrix = lyznet.utils.compute_determinant_dreal(sub_matrix)
        constraints.append(det_sub_matrix >= tol)

    positive_definiteness = dreal.And(*constraints)
    condition = dreal.logical_imply(x_bound, positive_definiteness)    
    start_time = time.time()
    result = dreal.CheckSatisfiability(
        dreal.logical_not(condition), config
        )
    if result is None:
        print("Positive definiteness verified!")
    else:
        print(result)
        print("Positive definiteness cannot be verified!")
    end_time = time.time()
    print(f"Time taken for verifying positive definiteness: " 
          f"{end_time - start_time} seconds.\n")

    print("Verifying contraction...")
    Df_x = [[fi.Differentiate(xi) for xi in x] for fi in f]
    Df_x_T = list(map(list, zip(*Df_x)))  # transpose of Df_x

    M_dot = [[dreal.Expression(0) for _ in range(d)] for _ in range(d)]

    for i in range(d):
        for j in range(d):
            lie_derivative_of_Mij = dreal.Expression(0)
            for k in range(len(x)):
                lie_derivative_of_Mij += f[k] * M[i][j].Differentiate(x[k])
            M_dot[i][j] = lie_derivative_of_Mij

    # print("M: ", M)

    M_prod1 = lyznet.utils.matrix_multiply_dreal(Df_x_T, M) 
    M_prod2 = list(map(list, zip(*M_prod1))) 
    # The following is supposed to be equivalent but some weird bugs exist with dReal
    # M_prod2 = lyznet.utils.matrix_multiply_dreal(M, Df_x)

    CM_derivative = [[M_prod1[i][j] + M_prod2[i][j] + M_dot[i][j] 
                      for j in range(d)] for i in range(d)]

    # verifying negative definiteness of CM_derivative
    constraints = []
    for n in range(1, d + 1):
        sub_matrix = [[-CM_derivative[i][j] for j in range(n)] for i in range(n)]
        det_sub_matrix = lyznet.utils.compute_determinant_dreal(sub_matrix)
        constraints.append(det_sub_matrix >= tol)

    negative_definiteness = dreal.And(*constraints)
    condition = dreal.logical_imply(x_bound, negative_definiteness)    
    start_time = time.time()
    result = dreal.CheckSatisfiability(
        dreal.logical_not(condition), config
        )
    if result is None:
        print("Negative definiteness verified!")
    else:
        print("Negative definiteness cannot be verified! Counterexample found: ")
        print(result)
    end_time = time.time()
    print(f"Time taken for verifying negative definiteness: " 
          f"{end_time - start_time} seconds.\n")

    return result


def neural_verifier(system, model, c2_P=None, c1_V=0.1, c2_V=1, 
                    tol=1e-4, accuracy=1e-2, 
                    net_type=None, number_of_jobs=32, verifier=None):
    # {x^TPx<=c2_P}: target quadratic-Lyapunov level set 
    # c1_V: target Lyapunov level set if c2_P is not specified
    # c2_V: maximal level to be verified

    config = lyznet.utils.config_dReal(number_of_jobs=number_of_jobs, tol=tol)
    xlim = system.domain

    # Create dReal variables based on the number of symbolic variables
    x = [dreal.Variable(f"x{i}") 
         for i in range(1, len(system.symbolic_vars) + 1)]

    f = [
        lyznet.utils.sympy_to_dreal(
            expr, dict(zip(system.symbolic_vars, x))
            )
        for expr in system.symbolic_f
        ]

    print('_' * 50)
    print("Verifying neural Lyapunov function:")

    if net_type == "Simple":
        V_learn = extract_dreal_SimpleNet(model, x)
    elif net_type == "Homo": 
        V_learn, norm_W = extract_dreal_HomoNet(model, x)        
    elif net_type == "Poly":
        V_learn = extract_dreal_PolyNet(model, x)
    elif net_type == "HomoPoly":
        V_learn = extract_dreal_HomoPolyNet(model, x)
    else:
        V_learn = extract_dreal_Net(model, x)
    print("V = ", V_learn.Expand())

    lie_derivative_of_V = dreal.Expression(0)
    for i in range(len(x)):
        lie_derivative_of_V += f[i] * V_learn.Differentiate(x[i])

    # If homogeneous verifier is called, do the following: 
    if verifier == "Homo": 
        # config = lyznet.utils.config_dReal(number_of_jobs=32, tol=1e-7)
        norm = dreal.sqrt(sum(xi * xi for xi in x)) 
        unit_sphere = (norm == 1)
        condition_V = dreal.logical_imply(unit_sphere, V_learn >= 1e-7)
        condition_dV = dreal.logical_imply(
            unit_sphere, lie_derivative_of_V <= -1e-7
            )
        condition = dreal.logical_and(condition_V, condition_dV)
        start_time = time.time()
        result = dreal.CheckSatisfiability(
            dreal.logical_not(condition), config
            )
        if result is None:
            print("Global stability verified for homogeneous vector field!")
            # print(f"The norm of the weight matrix is: {norm_W}")
        else:
            print(result)
            print("Stability cannot be verified for homogeneous vector field!")
        end_time = time.time()
        print(f"Time taken for verifying Lyapunov function of {system.name}: " 
              f"{end_time - start_time} seconds.\n")
        return 1, 1

    quad_V = dreal.Expression(0)
    for i in range(len(x)):
        for j in range(len(x)):
            quad_V += x[i] * system.P[i][j] * x[j]
    
    if c2_P is not None:
        target = quad_V <= c2_P

    start_time = time.time()

    def Check_inclusion(c1):
        x_bound = lyznet.utils.get_bound(x, xlim, V_learn, c2_V=c1)
        condition = dreal.logical_imply(x_bound, target)
        return dreal.CheckSatisfiability(dreal.logical_not(condition), config)
 
    if c2_P is not None:
        c1_V = lyznet.utils.bisection_glb(Check_inclusion, 0, 1, accuracy)
        print(f"Verified V<={c1_V} is contained in x^TPx<={c2_P}.")
    else:
        print(f"Target level set not specificed. Set it to be V<={c1_V}.")        
    c2_V = lyznet.reach_verifier_dreal(system, x, V_learn, f, c1_V, c_max=c2_V, 
                                       tol=tol, accuracy=accuracy,
                                       number_of_jobs=number_of_jobs)
    print(f"Verified V<={c2_V} will reach V<={c1_V}.")
    end_time = time.time()
    print(f"Time taken for verifying Lyapunov function of {system.name}: " 
          f"{end_time - start_time} seconds.\n")

    return c1_V, c2_V

def neural_clf_verifier(system, model, c2_P=None, c1_V=0.1, c2_V=1, 
                        tol=1e-4, accuracy=1e-2, number_of_jobs=32):

    config = lyznet.utils.config_dReal(number_of_jobs=number_of_jobs, tol=tol)
    xlim = system.domain

    x = [dreal.Variable(f"x{i}") 
         for i in range(1, len(system.symbolic_vars) + 1)]

    V_learn = extract_dreal_Net(model, x)
    print("V = ", V_learn.Expand())

    quad_V = dreal.Expression(0)
    for i in range(len(x)):
        for j in range(len(x)):
            quad_V += x[i] * system.P[i][j] * x[j]
    
    if c2_P is not None:
        target = quad_V <= c2_P

    start_time = time.time()
    def Check_inclusion(c1):
        x_bound = lyznet.utils.get_bound(x, xlim, V_learn, c2_V=c1)
        condition = dreal.logical_imply(x_bound, target)
        return dreal.CheckSatisfiability(dreal.logical_not(condition), config)
 
    print('_' * 50)
    print("Verifying neural Lyapunov function:")

    if c2_P is not None:
        c1_V = lyznet.utils.bisection_glb(Check_inclusion, 0, 1, accuracy)
        print(f"Verified V<={c1_V} is contained in x^TPx<={c2_P}.")
    else:
        print(f"Target level set not specificed. Set it to be V<={c1_V}.")        

    c2_V = lyznet.clf_reach_verifier_dreal(system, x, V_learn, c1_V, 
                                           c_max=c2_V, tol=tol, 
                                           accuracy=accuracy,
                                           number_of_jobs=number_of_jobs)    

    print(f"Verified V<={c2_V} will reach V<={c1_V}.")
    end_time = time.time()
    print(f"Time taken for verifying control Lyapunov function of {system.name}: " 
          f"{end_time - start_time} seconds.\n")

    return c1_V, c2_V