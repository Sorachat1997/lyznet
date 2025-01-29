import numpy as np
import sympy
import scipy
import torch
import lyznet


class DynamicalSystem:
    def __init__(self, f, domain, name, symbolic_vars=None, Q=None):
        self.f = f 
        self.domain = domain
        self.name = name
        self.system_type = "DynamicalSystem"

        self.symbolic_f = sympy.Matrix(f) if isinstance(f, list) else f

        if symbolic_vars is not None:
            self.symbolic_vars = list(symbolic_vars)
        else:
            n = len(self.symbolic_f)
            self.symbolic_vars = [sympy.symbols(f'x{i+1}') for i in range(n)]

        self.A = self.compute_linearization()  
        self.Q = Q if Q is not None else np.eye(len(self.symbolic_f))
        self.P = self.compute_quadratic_lyapunov_function() 

        self.f_numpy = sympy.lambdify(
            self.symbolic_vars, self.symbolic_f, modules=['numpy']
            )

        self.f_torch = [sympy.lambdify(self.symbolic_vars, f_i, 
                                       modules=[torch]) for f_i in self.f]

    def f_numpy_vectorized(self, samples):
        f_x = self.f_numpy(*samples.T)
        f_x = f_x.squeeze(axis=1).T
        return f_x

    def compute_quadratic_lyapunov_function(self):
        if not np.all(np.real(np.linalg.eigvals(self.A)) < 0):
            print("Skipping solving Lyapunov equation: "
                  "A is not Hurwitz.")
            return None
        if not np.all(np.linalg.eigvals(self.Q) > 0):
            print("Skipping solving Lyapunov equation: "
                  "Q is not positive definite.")
            return None   
        self.P = scipy.linalg.solve_continuous_lyapunov(self.A.T, -self.Q)
        return self.P

    # def compute_linearization(self):
    #     jacobian_np = np.array(
    #         lyznet.utils.compute_jacobian_np_dreal(
    #             self.symbolic_f, self.symbolic_vars
    #             )
    #         )
    #     return jacobian_np

    def compute_linearization(self, point=None):
        if point is None:
            point = [0] * len(self.symbolic_vars)
        jacobian_np = np.array(
            lyznet.utils.compute_jacobian_np_dreal(
                self.symbolic_f, self.symbolic_vars, point
            )
        )
        return jacobian_np


class ControlAffineSystem(DynamicalSystem):
    def __init__(self, f, g, domain, name, Q=None, R=None, u_func_numpy=None, 
                 u_expr=None):
        super().__init__(f, domain, name, Q=Q)
        self.system_type = "ControlAffineSystem"
        self.g = g
        self.symbolic_g = sympy.Matrix(g) 
        origin = {var: 0 for var in self.symbolic_vars}
        self.B = np.array(self.symbolic_g.subs(origin)).astype(np.float64)

        self.g_numpy = sympy.lambdify(
            self.symbolic_vars, self.symbolic_g, modules=['numpy']
            )

        self.g_torch = lambda x: lyznet.utils.numpy_to_torch(x, self.g_numpy)

        self.R = (np.array(R).astype(np.float64) if R is not None else 
                  np.eye(self.B.shape[1]))
        self.Q = (np.array(Q).astype(np.float64) if Q is not None else 
                  np.eye(self.A.shape[1]))

        self.P, self.K = self.compute_lqr_gain()

        # adding expressions/functions for controller (initilized to linear)
        if u_expr is None and self.K is not None:
            self.u_expr = sympy.Matrix(self.K) * sympy.Matrix(self.symbolic_vars)
        else:
            self.u_expr = u_expr

        if u_func_numpy is None:
            self.u_func_numpy = self.default_u_func_numpy
        else:
            self.u_func_numpy = u_func_numpy

        # use vectorized numpy functions to handle batch inputs
        self.closed_loop_f_numpy = lyznet.get_closed_loop_f_numpy(
                                self, self.f_numpy_vectorized, 
                                self.g_numpy_vectorized, 
                                self.u_func_numpy 
                                )

    def default_u_func_numpy(self, x):
        u_expr_numpy = sympy.lambdify(self.symbolic_vars, self.u_expr, 
                                      modules=['numpy'])
        x = np.atleast_2d(x)
        u_value_transposed = np.transpose(u_expr_numpy(*x.T))
        u_value = np.transpose(u_value_transposed, (0, 2, 1))
        output = np.squeeze(u_value, axis=-1)
        return output

    # def g_numpy_vectorized(self, samples):
    #     if samples.ndim == 1:
    #         samples = samples.reshape(1, -1)
    #     g_x = self.g_numpy(*samples.T)
    #     if g_x.ndim == 2:
    #         g_x = np.tile(g_x[np.newaxis, :, :], (samples.shape[0], 1, 1))
    #     return g_x

    def g_numpy_vectorized(self, samples):
        if samples.ndim == 1:
            samples = samples.reshape(1, -1)
        
        g_x = np.array([self.g_numpy(*sample) for sample in samples], 
                        dtype=np.float64)
        return g_x

    def compute_lqr_gain(self):
        if not np.all(np.linalg.eigvals(self.R) > 0):
            print("Skipping solving Lyapunov equation: "
                  "R is not positive definite.")
            return None, None   

        if lyznet.utils.is_controllable(self.A, self.B):
            print("The system is controllable.")
        elif lyznet.utils.is_stabilizable(self.A, self.B):
            print("The system is not controllable, but stabilizable.")
        else:
            print("The system is not stabilizable. Skipping lqr computation.")
            return None, None

        P = scipy.linalg.solve_continuous_are(self.A, self.B, self.Q, self.R)
        K = - np.linalg.inv(self.R) @ (self.B.T @ P)
        return P, K


class DiscreteDynamicalSystem(DynamicalSystem):
    def __init__(self, f, domain, name, symbolic_vars=None, Q=None):
        super().__init__(f, domain, name, symbolic_vars=symbolic_vars, Q=Q)
        self.system_type = "DiscreteDynamicalSystem"

    def compute_quadratic_lyapunov_function(self):
        if not np.all(np.abs(np.linalg.eigvals(self.A)) < 1):
            print("Skipping solving Lyapunov equation: "
                  "A is not Schur stable.")
            return None
        if not np.all(np.linalg.eigvals(self.Q) > 0):
            print("Skipping solving Lyapunov equation: "
                  "Q is not positive definite.")
            return None   
        self.P = scipy.linalg.solve_discrete_lyapunov(self.A.T, self.Q)
        return self.P
