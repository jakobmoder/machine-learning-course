#!/usr/bin/env python

""" Python code submission file.

IMPORTANT:
- Do not include any additional python packages.
- Do not change the interface and return values of the task functions.
- Only insert your code between the Start/Stop of your code tags.
- Prior to your submission, check that the pdf showing your plots is generated.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.special import gamma, digamma

# import additional functions Scipy functions if need be.
from scipy.optimize import newton

def task1():

    """ Probability Distributions

        Requirements for the plots:
        - The plot should show the histogram of the given data x and the
          density functions of the Gaussian as well as the Student's t model
    """

    with np.load('./data.npz') as data:
        x = data['task1_data']

    fig, ax = plt.subplots(1, 1, figsize=(15,5))
    fig.suptitle('Task 1 - Probability Distributions', fontsize=16)

    ax.set_title(r'Histogram against PDFs (Gaussian and Student\'s t)')
    ax.set_xlabel(r'$x$')

    """ Start of your code
    """
    def f_norm(mu, var, x):
        return 1/np.sqrt(2*np.pi*var) * np.exp(-(x-mu)**2/(2*var))
    
    def f_st(mu, lam, nu, x):
        return gamma((nu+1)/2) / gamma(nu/2) * (lam/(np.pi * nu))**(1/2) * (1 + (lam * (x-mu)**2) / nu)**(-(nu+1)/2)
    
    def f_E_eta(mu, lam, nu, x):
        return (nu+1) / (nu+lam*(x-mu)**2)
    
    def f_E_log_eta(mu, lam, nu, x):
        return digamma((nu+1)/2) - np.log((nu+lam*(x-mu)**2))/2
    
    def f_mu_update(E_eta, x):
        return np.sum(x * E_eta) / np.sum(E_eta)
    
    def f_lam_update(E_eta, mu, x):
        return len(x) / np.sum((x-mu)**2 * E_eta)
    
    # update rule in zero form
    def f_nu_zero(nu, N, E_eta, E_log_eta):
        return -digamma(nu/2) + 1/2*np.log(nu/2) + 1/(2*N) * np.sum(E_log_eta - E_eta) + 1 # digamma(nu/2) (np.log(nu/2) - 1/nu)
        
        
    
    # normal-dist mean and variance
    mu_ml = np.sum(x) / len(x)
    var_ml = np.sum((x-mu_ml)**2) / len(x)
    print('normal-dist mean and variance: %2.3f ; %2.f' % (mu_ml, var_ml))
    
    # student's t dist initial values
    nu = 1.4 #constant parameter
    lam = 1/var_ml
    mu = mu_ml
    # number of steps to be computed
    k = 15
    for i in range(0, k):
        E_eta = f_E_eta(mu, lam, nu, x) 
        E_log_eta = f_E_log_eta(mu, lam, nu, x)
        # update mu and lam
        mu = f_mu_update(E_eta, x)
        lam = f_lam_update(E_eta, mu, x)
        
    print('students t dist mu and lam after %d iterations: %2.6f ; %5.1f' % (k, mu, lam))
    
    # domain
    xmin, xmax = -0.13, 0.03
    x_n = 200
    #dx = (xmax - xmin) / x_n
    xlin = np.linspace(xmin, xmax, x_n+1)
    
    # plot data
    ax.hist(x, density=True, bins=150)
    ax.plot(xlin, f_norm(mu_ml, var_ml, xlin))
    ax.plot(xlin, f_st(mu, lam, nu, xlin))
    
    # bonus task
    lam = 1/var_ml
    mu = mu_ml
    nu = 1.0 # init nu with 1 this time
    k = 18
    nus = []
    for i in range(0, k):
        E_eta = f_E_eta(mu, lam, nu, x) 
        E_log_eta = f_E_log_eta(mu, lam, nu, x)
        # update mu, lam and also nu
        mu = f_mu_update(E_eta, x)
        lam = f_lam_update(E_eta, mu, x)    
        nu = newton(f_nu_zero, nu, args=[len(x), E_eta, E_log_eta], maxiter=50)
        nus.append(nu)
    
    print('students t dist mu, lam, nu after %d iterations: %2.6f ; %5.1f ; %3.5f' % (k, mu, lam, nu))
    print('iterations of nu: %s' % nus)
    # plot dist with new params
    ax.plot(xlin, f_st(mu, lam, nu, xlin))
    
    """ End of your code
    """

    return fig


def task2():
    """ Bayesian Linear Regression

        Requirements for the plots:
        - ax[0] should show the data points with the computed predictive distribution
          (mean and 2\sigma interval) with the given parameters.
        - ax[1] should show the data points with the computed predictive distribution
          (mean and 2\sigma interval) with parameters of your choice.
    """

    # load the data
    with np.load('data.npz') as data:
        # x1 = data['task1_data']
        data = data['task2_data']
        x = data[0]
        t = data[1]

    fig, ax = plt.subplots(2, 1, figsize=(15,10))
    fig.suptitle('Task 2 - Bayesian Linear Regression', fontsize=16)

    ax[0].set_title('Inclination $\\xi$ (t) over Orbital Eccentricity $\\rho$ (x)- Given parameters')
    ax[0].set_xlabel('$x$')
    ax[0].set_ylabel('$t$')

    ax[1].set_title('Inclination $\\xi$ (t) over Orbital Eccentricity $\\rho$ (x) - Chosen parameters')
    ax[1].set_xlabel('$x$')
    ax[1].set_ylabel('$t$')

    """ Start of your code
    """
    # sigmoid function
    def sigm(a):
        return 1/(1+np.exp(-a))
    
    def phi(x, c, s):
        return sigm((x-c)/s)
    
    def f_cm(m, M):
        return 7/5 * m/M - 1/5
    
    def f_PHI(x):
        PHI = np.ones((len(x), M+1)) * 1e6 # init with bias term
        for n in range(0, len(x)):
            PHI[n, 1:] = np.array([phi(x[n], f_cm(m, M), sb) for m in range(1, M+1)])
        return PHI
    
    M = 50 # number of shape functions
    # N = len(x) # number of data points
    sb = 1e-2  # shape function parameter
    alpha = 1 # precision of prior
    beta = 1e3 # precision of likelihood
    # build PHI matrix
    # PHI = np.ones((N, M+1)) * 1e6 # init with bias term
    PHI = f_PHI(x)
    
    S = np.linalg.inv(alpha * np.eye(M+1) + beta * np.matmul(PHI.transpose(), PHI))
    m = beta * np.matmul(np.matmul(S, PHI.transpose()), t)  # equals w_map
    
    # domain for pred dist
    xi = np.linspace(0, 1, 301)
    # create updated PHI matrix
    PHI_test = f_PHI(xi)
    
    pred_mean = np.matmul(PHI_test, m)
    pred_var = 1/beta + np.matmul(np.matmul(PHI_test, S), PHI_test.transpose())  
    
    ax[0].scatter(x, t)
    ax[0].plot(xi, pred_mean)
    ax[0].fill_between(xi, pred_mean + np.sqrt(np.diag(pred_var)), pred_mean - np.sqrt(np.diag(pred_var)))
    
    sb = 0.015
    alpha = 10
    beta = 1e2
    PHI = f_PHI(x) 
    S = np.linalg.inv(alpha * np.eye(M+1) + beta * np.matmul(PHI.transpose(), PHI))
    m = beta * np.matmul(np.matmul(S, PHI.transpose()), t)  # equals w_map
    PHI_test = f_PHI(xi)
    pred_mean = np.matmul(PHI_test, m)
    pred_var = 1/beta + np.matmul(np.matmul(PHI_test, S), PHI_test.transpose())
    
    ax[1].scatter(x, t)
    ax[1].plot(xi, pred_mean)
    ax[1].fill_between(xi, pred_mean + np.sqrt(np.diag(pred_var)), pred_mean - np.sqrt(np.diag(pred_var)))
    
    # experiment with parameters
    # alpha describes how confident we are in our prior. in ranges where we have no data this precision becomes dominant in the result
    # beta is the likelihood precision. consequently this determines the precision where we have seen many data
    # sb defines the height of the sigmoidals. when this value is too high the shape functions cannot correctly approximate 
    # low gradients of the curve. when it is too small on the other hand the sigmoidal steps can be seen in the results.
    # to summarize a right combination of number of shape funtions and the parameter sb has to be selected to accomodata the data.
    
    
    # in the scond plot I increased alpha and therefore presummed higher confidence in the prior
    # on the other hand I lowered the value of beta and therefore implied lower trust in the likelihood 
    
    """ End of your code
    """

    return fig

if __name__ == '__main__':
    # save_data()
    tasks = [task1, task2]

    pdf = PdfPages('figures.pdf')
    for task in tasks:
        f = task()
        pdf.savefig(f)

    pdf.close()
