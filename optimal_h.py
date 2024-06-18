import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')


def kernel(kernel_met, h, a, D):
    kernel_met_lower = kernel_met.lower()
    u = (D - a) / h
    if kernel_met_lower == 'gaussian':
        # kernel_eval = 1 / np.sqrt(2 * np.pi) * np.exp(-1/2 * ((D-a)/h)**2)
        kernel_eval = norm.pdf(u, 0, 1)
    elif kernel_met_lower == 'epanechnikov':
        kernel_eval = 3 / 4 * (1 - u**2) * (np.abs(u) <= 1)
    elif kernel_met_lower == 'uniform':
        kernel_eval = 1 / 2 * (np.abs(u) <= 1)
    elif kernel_met_lower == 'quartic':
        kernel_eval = 15 / 16 * (1 - u**2) ** 2 * (np.abs(u) <= 1)
    elif kernel_met_lower == 'triweight':
        kernel_eval = 35 / 32 * (1 - u**2) ** 3 * (np.abs(u) <= 1)
    elif kernel_met_lower == 'tricube':
        kernel_eval = 70 / 81 * (1 - np.abs(u)**3) ** 3 * (np.abs(u) <= 1)

    return 1 / h * kernel_eval

def ex_surface(h, treatment_val, Y_inv_test, est_Y_inv, est_pi, kernel_met, D, N):
    objective = 0
    a = treatment_val
    b = 2 * h
    num_Y_quantiles = Y_inv_test.shape[1]
    var_sum = 0
    B_sum = 0
    for s in range(num_Y_quantiles):
        # compute the kernel
        kernel_eval = kernel(kernel_met, b, a, D)
        kernel_ep_eval = kernel(kernel_met, 0.5*b, a, D)
        kernel_h_eval = kernel(kernel_met, h, a, D)

        Vhs = est_Y_inv[:, s].reshape(-1, 1) + kernel_h_eval.reshape(-1, 1) * \
              (Y_inv_test[:, s].reshape(-1, 1) - est_Y_inv[:, s].reshape(-1, 1)) / est_pi

        Vbs = est_Y_inv[:, s].reshape(-1, 1) + kernel_eval.reshape(-1, 1) * \
              (Y_inv_test[:, s].reshape(-1, 1) - est_Y_inv[:, s].reshape(-1, 1)) / est_pi

        Vebs = est_Y_inv[:, s].reshape(-1, 1) + kernel_ep_eval.reshape(-1, 1) * \
               (Y_inv_test[:, s].reshape(-1, 1) - est_Y_inv[:, s].reshape(-1, 1)) / est_pi

        var = np.var(Vhs) * h
        B_a = np.mean(Vbs - Vebs, axis=0)[0] / (0.75 * np.power(b, 2))

        var_sum += var
        B_sum += B_a ** 2

    return np.power((var_sum / (4*N*B_sum)), 0.2)

def obj_fun_surface(h, treatment_val, Y_inv_test, est_Y_inv, est_pi, kernel_met, D, N):
    objective = 0
    a = treatment_val
    b = 2 * h[0]
    num_Y_quantiles = Y_inv_test.shape[1]

    for s in range(num_Y_quantiles):
        # compute the kernel
        kernel_eval = kernel(kernel_met, b, a, D)
        kernel_ep_eval = kernel(kernel_met, 0.5*b, a, D)
        kernel_h_eval = kernel(kernel_met, h[0], a, D)

        Vhs = est_Y_inv[:, s].reshape(-1, 1) + kernel_h_eval.reshape(-1, 1) * \
              (Y_inv_test[:, s].reshape(-1, 1) - est_Y_inv[:, s].reshape(-1, 1)) / est_pi

        Vbs = est_Y_inv[:, s].reshape(-1, 1) + kernel_eval.reshape(-1, 1) * \
              (Y_inv_test[:, s].reshape(-1, 1) - est_Y_inv[:, s].reshape(-1, 1)) / est_pi

        Vebs = est_Y_inv[:, s].reshape(-1, 1) + kernel_ep_eval.reshape(-1, 1) * \
               (Y_inv_test[:, s].reshape(-1, 1) - est_Y_inv[:, s].reshape(-1, 1)) / est_pi

        var = np.var(Vhs)

        B_a = np.mean(Vbs - Vebs, axis=0) / (0.75 * np.power(b, 2))

        # add_term = var / (N * h[0]) + B_a * np.power(h[0], 2)
        # add_term = var / N + B_a * np.power(h[0], 2)
        add_term = var / N + np.power(B_a, 2) * np.power(h[0], 4)
        objective += add_term

    return objective + h[1]


def obj_fun_surface_IPW(h, treatment_val, Y_inv_test, est_Y_inv, est_pi, kernel_met, D, N):
    objective = 0
    a = treatment_val
    b = 2 * h[0]
    num_Y_quantiles = Y_inv_test.shape[1]

    for s in range(num_Y_quantiles):
        # compute the kernel
        kernel_eval = kernel(kernel_met, b, a, D)
        kernel_ep_eval = kernel(kernel_met, 0.5*b, a, D)
        kernel_h_eval = kernel(kernel_met, h[0], a, D)

        Vhs = kernel_h_eval.reshape(-1, 1) * Y_inv_test[:, s].reshape(-1, 1) / est_pi

        Vbs = kernel_eval.reshape(-1, 1) * Y_inv_test[:, s].reshape(-1, 1) / est_pi

        Vebs = kernel_ep_eval.reshape(-1, 1) * Y_inv_test[:, s].reshape(-1, 1) / est_pi

        var = np.var(Vhs)

        B_a = np.mean(Vbs - Vebs, axis=0) / (0.75 * np.power(b, 2))

        # add_term = var / (N * h[0]) + B_a * np.power(h[0], 2)
        # add_term = var / N + B_a * np.power(h[0], 2)
        add_term = var / N + np.power(B_a, 2) * np.power(h[0], 4)
        objective += add_term

    return objective + h[1]


'''optimal bandwidth for the sample covariance function'''
def optbandwidth_surface(treatment_val, est_pi, est_Y_inv, XD_test, Y_inv_test, kernel_met):
    D = XD_test[:, -1].reshape(-1, 1)
    c = 1
    N = len(XD_test)
    initial_guess = c * np.std(D) * np.power(N, -0.2)


    opt_bandwidth = minimize(obj_fun_surface_IPW, (initial_guess, 0), method='SLSQP', bounds=((0, None), (0, 0)),
                             args=(treatment_val, Y_inv_test, est_Y_inv, est_pi, kernel_met, D, N),
                             options={'xatol': 1e-7, 'disp': True})
   
    a1 = opt_bandwidth.x[0]
    '''
    a2 = ex_surface(initial_guess, treatment_val, Y_inv_test, est_Y_inv, est_pi, kernel_met, D, N)
    '''
    return a1


'''
def obj_fun_surface(h, treatment_val, Y_inv_test, est_Y_inv, est_pi, kernel_met, D, N, init_h):
    objective = 0
    a = treatment_val
    b = 2 * init_h
    num_Y_quantiles = Y_inv_test.shape[1]

    for s in range(num_Y_quantiles):
        # compute the kernel
        kernel_eval = kernel(kernel_met, b, a, D)
        kernel_ep_eval = kernel(kernel_met, 0.5*b, a, D)
        kernel_h_eval = kernel(kernel_met, init_h, a, D)

        Vhs = est_Y_inv[:, s].reshape(-1, 1) + kernel_h_eval.reshape(-1, 1) * \
              (Y_inv_test[:, s].reshape(-1, 1) - est_Y_inv[:, s].reshape(-1, 1)) / est_pi

        Vbs = est_Y_inv[:, s].reshape(-1, 1) + kernel_eval.reshape(-1, 1) * \
              (Y_inv_test[:, s].reshape(-1, 1) - est_Y_inv[:, s].reshape(-1, 1)) / est_pi

        Vebs = est_Y_inv[:, s].reshape(-1, 1) + kernel_ep_eval.reshape(-1, 1) * \
               (Y_inv_test[:, s].reshape(-1, 1) - est_Y_inv[:, s].reshape(-1, 1)) / est_pi

        var = np.var(Vhs) * init_h
        B_a = np.mean(Vbs - Vebs, axis=0) / (0.75 * np.power(b, 2))

        add_term = var / (N * h[0]) + np.power(B_a, 2) * np.power(h[0], 4)
        # add_term = var / N + B_a * np.power(h[0], 4)
        # add_term = var / N + np.power(B_a, 2) * np.power(h[0], 4)
        objective += add_term

    return objective
'''

