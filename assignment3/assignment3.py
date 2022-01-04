#!/usr/bin/env python

""" Python code submission file.

IMPORTANT:
- Do not include any additional python packages.
- Do not change the existing interface and return values of the task functions.
- Prior to your submission, check that the pdf showing your plots is generated.
"""

from operator import inv, matmul
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from numpy import linalg
from numpy.core.fromnumeric import trace, transpose
from numpy.linalg.linalg import norm
from scipy.optimize import approx_fprime
from scipy.optimize.minpack import check_gradient


def f_k(xn, xm, theta):
    assert len(theta) == 4
    xnr = xn.reshape(-1, 1) # column 
    xmr = xm.reshape(1, -1) # row
    Knm = theta[0] * np.exp(-theta[1]/2 * (xnr - xmr)**2) + theta[2] + theta[3] * np.dot(xnr, xmr) 
    return Knm

def task1():

    """ Gaussian Processes

        Requirements for the plots:
            - ax[0] The combined predictive plot with the given parameters
            - ax[1] The combined predictive plot with the chosen parameters
    """

    fig, ax = plt.subplots(2, 1, figsize=(15,10))
    fig.suptitle('Task 1 - Gaussian Processes on Toy Dataset', fontsize=16)

    ax[0].set_title('Prediction over $\mathcal{X}$ - Given parameters')
    ax[0].set_xlabel('$x$')
    ax[0].set_ylabel('$t$')

    ax[1].set_title('Prediction over $\mathcal{X}$ - Chosen parameters')
    ax[1].set_xlabel('$x$')
    ax[1].set_ylabel('$t$')

    """ Start of your code
    """ 
    N = 30 # training data points
    x = np.random.uniform(0.0, 1.0, N)
    y = np.sin(2*np.pi*x)
    std = 0.15
    t = y + np.random.normal(0, std, N)

    P = 300 # testing data points
    xp = np.linspace(0, 1.3, P)

    # parameters
    beta_inv = std**2 # variance
    theta = [1, 3, 0, 1]

    # build covariance matrix
    Cnn = f_k(x, x, theta) + np.eye(N)*beta_inv
    Cnp = f_k(x, xp, theta)
    Cpp = f_k(xp, xp, theta) + np.eye(P)*beta_inv

    # mean and covariance of predictive dist
    mu_pgn = np.matmul(Cnp.transpose(), np.matmul(np.linalg.inv(Cnn), t) )
    C_pgn = Cpp - np.matmul(Cnp.transpose(), np.matmul(np.linalg.inv(Cnn), Cnp) )

    yp = np.sin(2*np.pi*xp) 
    ax[0].plot(xp, yp, label='y(x)')
    ax[0].plot(xp, mu_pgn, label='mean of predictive-dist')
    ax[0].scatter(x, t, label='training data points')
    # draw samples from predictive-dist
    ns = 5
    pgn_samples = np.random.multivariate_normal(mu_pgn, C_pgn, ns)
    for i in range(0, ns):
        ax[0].plot(xp, pgn_samples[i, :], alpha=0.25)

    diff = 1.96 * np.sqrt(np.diag(C_pgn))
    ax[0].fill_between(xp, mu_pgn+diff, mu_pgn-diff, alpha=0.35)
    ax[0].legend()

    # new beta and theta
    std = 0.4
    t = y + np.random.normal(0, std, N)
    beta_inv = std**2 # variance
    theta = [3, 120, 3, 2] # theta = [0, 30, 3, 1]
    Cnn = f_k(x, x, theta) + np.eye(N)*beta_inv
    Cnp = f_k(x, xp, theta)
    Cpp = f_k(xp, xp, theta) + np.eye(P)*beta_inv
    mu_pgn = np.matmul(Cnp.transpose(), np.matmul(np.linalg.inv(Cnn), t) )
    C_pgn = Cpp - np.matmul(Cnp.transpose(), np.matmul(np.linalg.inv(Cnn), Cnp) )

    ax[1].plot(xp, yp, label='y(x)')
    ax[1].plot(xp, mu_pgn, label='mean of predictive-dist')
    ax[1].scatter(x, t, label='training data points')
    diff = 1.96 * np.sqrt(np.diag(C_pgn))
    ax[1].fill_between(xp, mu_pgn+diff, mu_pgn-diff, alpha=0.35)
    ax[1].legend()

    """ End of your code
    """
    return fig


def f_sdiff(xn, xm):
    xnr = xn.reshape(-1, 1) # column 
    xmr = xm.reshape(1, -1) # row
    return (xnr - xmr)**2

def f_mult(xn, xm):
    xnr = xn.reshape(-1, 1) # column 
    xmr = xm.reshape(1, -1) # row
    return np.dot(xnr, xmr) 


# negative log-likelihood function of p(t|theta)
def f_nlog(theta, x, t, beta_inv):
        Cn = f_k(x, x, theta) + np.eye(len(x)) * beta_inv
        s, logdet = np.linalg.slogdet(Cn)
        return 1/2 * s*logdet  + 1/2 * np.matmul(np.matmul(t.transpose(), np.linalg.inv(Cn)), t) + len(x)/2 * np.log(2*np.pi)

def task2():

    """ Learning the parameters

        Requirements for the plots:
            - ax[0] The combined predictive plot with the optimal parameters
            - ax[1] Negative log likelihood over k returned from gradient_check
            - ax[2] Comparison between the numerical and the analytical gradient
                    as a bar plot
    """

    fig = plt.figure(figsize=(15,10), constrained_layout=True)
    fig.suptitle('Task 2 - Learning the Parameters on Toy Dataset', fontsize=16)
    ax = [None, None, None]
    g = fig.add_gridspec(6, 5)
    ax[0] = fig.add_subplot(g[1:4:, :])
    ax[0].set_title('Prediction over $\mathcal{X}$ - Optimal parameters')
    ax[0].set_xlabel('$x$')
    ax[0].set_ylabel('$t$')

    ax[1] = fig.add_subplot(g[4:, 0:3])
    ax[1].set_title('Negative log likelihood over $k$')
    ax[1].set_xlabel('$k$')

    ax[2] = fig.add_subplot(g[4:, 3:5])
    ax[2].set_title('Numerical vs. Analytical Gradient')
    ax[2].set_xticks(np.arange(4))
    ax[2].set_xticklabels((r'$\frac{\partial}{\partial \theta_0}$', r'$\frac{\partial}{\partial \theta_1}$', r'$\frac{\partial}{\partial \theta_2}$', r'$\frac{\partial}{\partial \theta_3}$'))

    """ Start of your code
    """
    N = 30 # training data points
    x = np.random.uniform(0.0, 1.0, N)
    y = np.sin(2*np.pi*x)
    std = 0.15
    t = y + np.random.normal(0, std, N)
    grad, num_grad = gradient_check(x, t, std**2)
    #print('analytical gradient: %s' % grad)
    #print('numerical gradient: %s' % num_grad)
    ax[2].bar(np.arange(4), grad, width=0.4, label='analytical gradient')
    ax[2].bar(np.arange(4) + 0.5, num_grad, width=0.4, label='numerical gradient', color='r')
    ax[2].legend()

    # parameters
    beta_inv = std**2 # variance
    theta, nll = compute_optimal_parameters(x, t, beta_inv)
    #print('Optimal gradients: %s' % theta)
    ax[1].plot(np.arange(100), nll)

    P = 300 # testing data points
    xp = np.linspace(0, 1.3, P)
    # build covariance matrix
    Cnn = f_k(x, x, theta) + np.eye(N)*beta_inv
    Cnp = f_k(x, xp, theta)
    Cpp = f_k(xp, xp, theta) + np.eye(P)*beta_inv
    # mean and covariance of predictive dist
    mu_pgn = np.matmul(Cnp.transpose(), np.matmul(np.linalg.inv(Cnn), t) )
    C_pgn = Cpp - np.matmul(Cnp.transpose(), np.matmul(np.linalg.inv(Cnn), Cnp) )
    yp = np.sin(2*np.pi*xp) 
    ax[0].plot(xp, yp, label='y(x)')
    ax[0].plot(xp, mu_pgn, label='mean of predictive-dist')
    ax[0].scatter(x, t, label='training data points')
    # draw samples from predictive-dist
    ns = 5
    pgn_samples = np.random.multivariate_normal(mu_pgn, C_pgn, ns)
    for i in range(0, ns):
        ax[0].plot(xp, pgn_samples[i, :], alpha=0.25)

    diff = 1.96 * np.sqrt(np.diag(C_pgn))
    ax[0].fill_between(xp, mu_pgn+diff, mu_pgn-diff, alpha=0.35)
    ax[0].legend()

    """ End of your code
    """
    return fig


def f_gradient(Ci, t, Gr):
    return 1/2 * np.trace(np.matmul(Ci, Gr)) - 1/2 * np.matmul(np.matmul(t.transpose(), Ci), np.matmul(Gr, np.matmul(Ci, t)))

def compute_gradients(x, t, theta, param):
    Cnn = f_k(x, x, theta) + np.eye(len(x)) * param
    S = f_sdiff(x, x)
    Cnni = np.linalg.inv(Cnn)
    # grad[0] = 1/2 * np.trace(np.matmul(Cnni, np.exp(-theta[0]/2 * S))) - 1/2 * t.transpose()*Cnni * np.exp(-theta[0]/2*S) * Cnni * t
    grad = np.zeros(4)
    grad[0] = f_gradient(Cnni, t, np.exp(-theta[1]/2 * S))
    grad[1] = f_gradient(Cnni, t, -theta[0]*1/2 * S * np.exp(-theta[1]/2 *S)) # no np.matmul!
    grad[2] = f_gradient(Cnni, t, np.ones( (len(x), len(x)) ))
    grad[3] = f_gradient(Cnni, t, f_mult(x, x))
    return grad

def gradient_check(x, t, param): 
    theta = np.random.rand(4)
    grad = np.zeros(4)
    num_grad = np.zeros(4)

    """ Start of your code
    """
    grad = compute_gradients(x, t, theta, param)
    beta_inv = param
    eps = (np.finfo(float).eps)**(1/2)
    num_grad = approx_fprime(theta, f_nlog, eps, x, t, beta_inv)

    """ End of your code
    """

    return grad, num_grad


def compute_optimal_parameters(x, t, param):

    theta = np.random.rand(4)
    nll = np.zeros(100)
    """ Start of your code
    """
    alpha = 1
    K = 100
    T = 10
    for k in range(0, K):
        E = f_nlog(theta, x, t, param)
        nll[k] = E
        Egr = compute_gradients(x, t, theta, param)
        for ti in range(0, T):
            theta_new = theta - alpha * Egr
            theta_new0 = np.c_[theta_new, np.zeros(len(theta))]
            theta_new = np.max(theta_new0, axis=1)
            #print(theta_new)
            q = E + np.dot(Egr.transpose(), (theta_new - theta)) + 1/(2*alpha) * np.linalg.norm(theta_new-theta, 2)
            E_new = f_nlog(theta_new, x, t, param)
            if E_new <= q:
                #theta = theta_new
                alpha = 2 * alpha
                break
            else:
                alpha = 0.5 * alpha
        theta = theta_new    

    """ End of your code
    """
    return theta, nll


def task3():

    """ Gaussian Process on COVID-19

        Requirements for the plots:
            - ax[0] The combined predictive plot with the optimal parameters
                    on the given COVID 19 dataset
    """
    with np.load('./data.npz') as data:
        date = data['date']
        t = data['cases']
        s_t = t.max()
        t /= s_t
        s_x = len(t)
        x = np.linspace(0, 1, len(t))  

    fig, ax = plt.subplots(1, 1, figsize=(15,10))
    fig.suptitle('Task 3 - Gaussian Processes on COVID-19 Dataset', fontsize=16)

    ax.set_title(r'Guassian process model prediction $\approx 136$ days into the future')
    ax.set_xlabel(f'Days since {date[0]}')

    """ Start of your code
    """
    N = len(x)
    # parameters
    std = 0.03
    beta_inv = std**2 # variance
    theta, _ = compute_optimal_parameters(x, t, beta_inv)

    P = 2000 # testing data points
    xp = np.linspace(0, 1.3, P) # *1.3 is about 590 days
    # build covariance matrix
    Cnn = f_k(x, x, theta) + np.eye(N)*beta_inv
    Cnp = f_k(x, xp, theta)
    Cpp = f_k(xp, xp, theta) + np.eye(P)*beta_inv
    # mean and covariance of predictive dist
    mu_pgn = np.matmul(Cnp.transpose(), np.matmul(np.linalg.inv(Cnn), t) )
    C_pgn = Cpp - np.matmul(Cnp.transpose(), np.matmul(np.linalg.inv(Cnn), Cnp) )

    ax.plot(xp*s_x, mu_pgn*s_t, label='mean of predictive-dist')
    # draw samples from predictive-dist
    ns = 5
    pgn_samples = np.random.multivariate_normal(mu_pgn, C_pgn, ns)
    for i in range(0, ns):
        ax.plot(xp*s_x, pgn_samples[i, :]*s_t, alpha=0.15)

    diff = 1.96 * np.sqrt(np.diag(C_pgn))
    ax.fill_between(xp*s_x, (mu_pgn+diff)*s_t, (mu_pgn-diff)*s_t, alpha=0.35)
    ax.legend()

    ax.scatter(x * s_x, t * s_t, s=20, label='New cases / day')
    ax.legend()

    """ End of your code
    """

    return fig


if __name__ == '__main__':
    tasks = [task1, task2, task3]

    pdf = PdfPages('figures.pdf')
    for task in tasks:
        f = task()
        pdf.savefig(f)

    pdf.close()