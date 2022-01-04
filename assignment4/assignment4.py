#!/usr/bin/env python

""" Python code submission file.

IMPORTANT:
- Do not include any additional python packages.
- Do not change the existing interface and return values of the task functions.
- Prior to your submission, check that the pdf showing your plots is generated.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from numpy.testing._private.utils import assert_no_gc_cycles

def task1():

    """ Graphical Models

        Requirements for the plots:
            - fig_l The line plots for different parameters showing the noisy input,
                    your solution for the MAP and Expectation method and the ground truth
            - fig_c Belief/Costs plots for MAP and Expectation (invert y-axis if necessary)
    """

    fig_l, ax_l = plt.subplots(3, 1, figsize=(15,10))
    fig_l.suptitle('Task 1 - Input, GT, MAP, Expectation', fontsize=16)

    ax_l[0].set_title(r'$\lambda=0.7$, $T=1$')
    ax_l[1].set_title(r'Your $\lambda$, $T=1$')
    ax_l[2].set_title(r'$\lambda=0.7$, $T=20$')
    fig_l.tight_layout(rect=[0, 0, 1, .95])

    fig_c, ax_c = plt.subplots(3, 3, figsize=(15,10))
    fig_c.suptitle('Task 1 - Costs g, Min-Marginals (MAP), Marginals (Expectation)', fontsize=16)

    ax_c[0,0].set_title(r'$\lambda=0.7$, $T=1$')
    ax_c[0,1].set_title(r'Your $\lambda$, $T=1$')
    ax_c[0,2].set_title(r'$\lambda=0.7$, $T=20$')
    ax_c[0,0].set_ylabel(r'Costs $g$')
    ax_c[1,0].set_ylabel(r'Min-Marginals')
    ax_c[2,0].set_ylabel(r'Marginals')
    fig_c.tight_layout(rect=[0, 0, 1, .95])

    with np.load('./data.npz') as data:
        y = data['data']

    """ Start of your code
    """
    # unary costs
    def f_g(x, s, T):
        xr = x.reshape(1, -1)
        sr = s.reshape(-1, 1)
        return 1/T * (xr * np.log(xr/sr) + sr - xr)

    # pairwise costs
    def f_f(s, t, lam, T):
        sr = s.reshape(-1, 1)
        tr = t.reshape(1, -1)
        return lam/T * np.abs(sr-tr)

    #def f_poisson(x, yi):
    #    return np.exp(yi) * yi**x / [np.math.factorial(x[i]) for i in range(0,len(x)) ] 

    xl = []
    
    for i in range(0, len(y)):
        xi = np.random.poisson(y[i])
        # check for range within [1,255]
        if xi > 255:
            xi = 255
        elif xi < 1:
            xi = 1
        xl.append(xi)

    K = 128 # number of labels
    #print('Original x: %s' % xl)
    x = ((np.array(xl)+1) // 2) # data within [1,128]
    #print('Updated x: %s' % x)
    s = np.arange(1, K+1) # labels
    print('Shape of s: %d ; First element: %d' % (len(s), s[0] ) )

    T, lam = 1, 0.7
    # compute unary and pairwise costs
    g = f_g(x, s, T)
    f = f_f(s, s, lam, T)

    #print('Dimensions of g: %d, %d' % (g.shape[0], g.shape[1]) )
    #print('Dimensions of f %d, %d' % (f.shape[0], f.shape[1]) )
    # min-marginals
    def f_mm(x, g, f):
        K = len(g)
        N = len(x)
        F = 0
        FM = np.zeros((K, N))    
        for i in range(0, N-1):
            # compute F_i+1
            F = np.min(g[:, i] + f + F, axis=1)
            FM[:, i+1] = F
        B = 0
        BM = np.zeros((K, N))
        for i in range(N-1, 0, -1):
            #idx = np.argmin(g[:, i])
            B = np.min(g[:, i] + f + B, axis=1)
            BM[:, i-1] = B
        #print('Shapes of F and B [%d, %d]; [%d, %d]' % (FM.shape[0], FM.shape[1], BM.shape[0], BM.shape[1])  )
        # add up matrices
        return g + BM + FM
    
    # marginal computation
    def f_pm(x, g, f):
        K = len(g)
        N = len(x)
        F = 0
        FM = np.zeros((K, N))    
        for i in range(0, N-1):
            # compute F_i+1
            A = -g[:, i]-f- F
            Z = np.min(A) # axis=1
            F = -Z - np.log(np.sum(np.exp(A-Z), axis=1)) # sum
            FM[:, i+1] = F
        B = 0
        BM = np.zeros((K, N))
        for i in range(N-1, 0, -1):
            A = -g[:, i]-f- B
            Z = np.min(A) # axis=1
            B = -Z - np.log(np.sum(np.exp(A-Z), axis=1)) # sum
            BM[:, i-1] = B
        #print(BM.shape)
        #print(FM.shape)
        Pm = np.zeros((K, N))
        for i in range(0, N):
            A = -g[:,i]-FM[:,i]-BM[:,i]
            Z = np.min(A)
            Pi = np.exp(A - Z) / np.sum(np.exp(A - Z))
            Pm[:, i] = Pi
        # p-marginal
        return Pm

    # MAP Solution
    E = f_mm(x, g, f)
    idxs = np.argmin(E, axis=0) # look for minium along label axis
    # get values at respective indices of label vector
    x_map = s[idxs]

    # Marginal Solution
    Pm = f_pm(x, g, f)
    # compute expectation with marginals
    x_marginal = np.dot(s.transpose(), Pm) #np.sum(Pm*s.reshape(-1,1), axis=0)
    #print('Sum of Pm at i=100 : %3.3f' % np.sum(Pm[:,100]))
    #print('Label vector: %s' % s)

    ax_l[0].plot(x, label='input-data', alpha=0.3)
    ax_l[0].plot(x_map, label='MAP') #np.array(idxs) + 1
    ax_l[0].plot(x_marginal, label='Marginal')
    ax_l[0].plot(y/2, label='GT', alpha=0.5)
    ax_l[0].legend()

    ax_c[0,0].imshow(g) # unary costs
    ax_c[1,0].imshow(E) # min-marginals
    ax_c[2,0].imshow(Pm) # marginals

    T, lam = 1, 0.2
    # compute unary and pairwise costs
    g = f_g(x, s, T)
    f = f_f(s, s, lam, T)
    E = f_mm(x, g, f)
    idxs = np.argmin(E, axis=0) 
    x_map = s[idxs]
    # Marginal Solution
    Pm = f_pm(x, g, f)
    # compute expectation with marginals
    x_marginal = np.dot(s.transpose(), Pm)
    ax_l[1].plot(x, label='input-data', alpha=0.3)
    ax_l[1].plot(x_map, label='MAP')
    ax_l[1].plot(x_marginal, label='Marginal')
    ax_l[1].plot(y/2, label='GT', alpha=0.5) 
    ax_l[1].legend()
    ax_c[0,1].imshow(g) # unary costs
    ax_c[1,1].imshow(E) # min-marginals
    ax_c[2,1].imshow(Pm) # marginals

    T, lam = 20, 0.7
    # compute unary and pairwise costs
    g = f_g(x, s, T)
    f = f_f(s, s, lam, T)
    E = f_mm(x, g, f)
    idxs = np.argmin(E, axis=0) 
    x_map = s[idxs]
    # Marginal Solution
    Pm = f_pm(x, g, f)
    # compute expectation with marginals
    x_marginal = np.dot(s.transpose(), Pm)
    ax_l[2].plot(x, label='input-data', alpha=0.3)
    ax_l[2].plot(x_map, label='MAP')
    ax_l[2].plot(x_marginal, label='Marginal')
    ax_l[2].plot(y/2, label='GT', alpha=0.5) 
    ax_l[2].legend()
    ax_c[0,2].imshow(g) # unary costs
    ax_c[1,2].imshow(E) # min-marginals
    ax_c[2,2].imshow(Pm) # marginals

    """
         End of your code
    """
    return [fig_l, fig_c]

if __name__ == '__main__':
    tasks = [task1]

    pdf = PdfPages('figures.pdf')
    for task in tasks:
        f = task()
        for f_ in f:
            pdf.savefig(f_)

    pdf.close()