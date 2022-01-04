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
from scipy.special import gamma
import gzip

def task2():

    """ Distribution Calculus

        Requirements for the plots:
        - ax[0] Should show the a 2D scatter plot with a 2D contour plot on
          the domain [-1,1]^2
        - ax[1] Should show the marginal distribution p(X_1) on [-1,1]
        - ax[1] Should show the conditionals p(X_1 | X_2=-0.3) and 
          p(X_1 | X_2=0.3) on [-1,1]
    """
    fig, ax = plt.subplots(1, 3, figsize=(15,5))
    fig.suptitle('Task 2 - Distribution Calculus', fontsize=16)

    ax[0].set_title(r'Joint prob. $p(X_1, X_2)$')
    ax[1].set_title(r'Marginal $p(X_1)$')
    ax[2].set_title(r'Conditionals $p(X_1 \mid X_2)$')
    ax[0].set_ylabel(r'$X_2$')
    ax[0].set_xlabel(r'$X_1$')
    ax[1].set_xlabel(r'$X_1$')
    ax[2].set_xlabel(r'$X_1$')

    """ Start of your code
    """
    def f_p_2d(X_1, X_2, sigma, alpha):
        res = 1/(2*np.pi*sigma**2*np.sqrt(1-alpha**2)) * np.exp(-(X_1**2-2*alpha*X_1*X_2+X_2**2)/(2*sigma**2*(1-alpha**2)))
        return res

    def f_p_1d(X_1, sigma, alpha):
        res = 1/(np.sqrt(2*np.pi*sigma**2)) * np.exp(-X_1**2/(2*sigma**2))
        return res

    def f_p1gp2(X_1, X_2, sigma, alpha):
        res = 1 / np.sqrt(2*np.pi*sigma**2*(1-alpha**2)) * np.exp(-(X_1**2-2*alpha*X_1*X_2+X_2**2)/(2*sigma**2*(1-alpha**2)) + X_2**2/(2*sigma**2))
        return res
        
    sigma = np.sqrt(0.1)
    alpha = -0.8
    mu = np.array([0, 0])
    S = np.array([[sigma**2, alpha*sigma**2], [alpha*sigma**2, sigma**2]]) # covariance matrix
    n_draw = 1000
    xy = np.random.multivariate_normal(mu, S, size=n_draw)
    # number of points for linear sampling
    n = 100
    xmin = -1
    xmax = 1
    dx1 = dx2 = (xmax-xmin)/n
    x1 = np.linspace(xmin, xmax, n+1)
    x2 = np.linspace(xmin, xmax, n+1)
    X1, X2 = np.meshgrid(x1, x2)
    p_2d = f_p_2d(X1, X2, sigma, alpha)
    # the total prob should sum up to 1
    #int_p_2d = np.sum(p_2d) * dx1 * dx2
    p_1d = f_p_1d(x1, sigma, alpha)
    #int_p_1d = np.sum(p_1d) * dx1
    # print('int_p_2d : %2.3f; int_p_1d : %2.3f' % (int_p_2d, int_p_1d))

    ax[0].contour(x1, x2, p_2d.transpose()) 
    ax[0].scatter(xy[:, 0], xy[:,1])
    ax[1].plot(x1, p_1d)
    ax[1].hist(xy[:, 0], density=True)
    # integral should also be the same
    ax[1].plot(x1, np.sum(p_2d, axis=1) * dx2)

    ax[2].plot(x1, f_p1gp2(x1, 0.3, sigma, alpha))
    ax[2].plot(x1, f_p1gp2(x1, -0.3, sigma, alpha))
    
    # sum should again be one
    # print(np.sum( f_p1gp2(x1, -0.3, sigma, alpha))*dx1)
    """ End of your code
    """

    return fig


def task3():
    """ Bayesian Denoising

        Requirements for the plots:
        - the first row should show your results for \lambda=1
        - the second row should show your results for \lambda=10^{-2}
        - the third row should show your results for \lambda=10^{-3}
        - arange your K images as a grid
    """
    fig, ax = plt.subplots(3, 4, figsize=(15,10))
    fig.suptitle('Task 3 - Bayesian Denoising', fontsize=16)

    # load the data
    with np.load('./data.npz') as data:
        x = data['x']
        y = data['y']
        y_star = data['y_star']

    for ax_ in ax.reshape(-1): ax_.set_xticks([]), ax_.set_yticks([])
    ax[0,0].title.set_text(r'$\mathbf{y}^*$')
    ax[0,1].title.set_text(r'$\mathbf{x}$')
    ax[0,2].title.set_text(r'$\mathbf{\hat y}_{\operatorname{MMSE}}(\mathbf{x})$')
    ax[0,3].title.set_text(r'$\mathbf{\hat y}_{\operatorname{MAP}}(\mathbf{x})$')
    ax[0,0].set_ylabel(r'$\lambda=1$')
    ax[1,0].set_ylabel(r'$\lambda=10^{-2}$')
    ax[2,0].set_ylabel(r'$\lambda=10^{-3}$')

    """ Start of your code
    """

    # exponential part of p_xoy
    def f_r(k, theta, lam, x, y):
        return (-lam/theta * np.sum(x / y) + lam * (k-1) * np.sum(np.log(x / y)))


    def f_p_xoy(k, theta, d, lam, x, y):
        #factor =  (theta**k*gamma(k))**-d
        #res = factor * np.exp(-theta/lam * np.sum(x / y) + lam * (k-1) * np.sum(np.log(x / y)))
        # not working because of overflow
        # logarithm of p_xoy
        log_res = -d * np.log(theta**k * gamma(k)) + f_r(k, theta, lam, x, y)
        return (log_res) # np.exp()

    k, theta = 4, 0.25
    nx = len(x)
    ny = len(y)
    x_flt = x.reshape(nx, -1)
    y_flt = y.reshape(ny, -1)   
    _, d = y_flt.shape 
    _, dx = x_flt.shape

    assert d == dx
    #sqrt of d
    s_d = int(np.sqrt(d))

    # select k=16 images
    k = 16
    intervall = 2 # increase this value k times
    #sqrt of k
    s_k = int(np.sqrt(k))
    lam_list = [1, 1e-2, 1e-3]
    y_mmse_dict = {}
    y_map_dict = {}

    for lam in lam_list:
        y_mmse_list = []
        y_map_list = []
        for i in range(0, intervall*k, intervall):
            x0 = x_flt[i]
            # evaluate likelihood for all y
            p_xoy_x0 = np.array([f_p_xoy(k, theta, d, lam, x0, y_eval) for y_eval in y_flt])
            # y_mmse implementation
            rn = np.array([f_r(k, theta, lam, x0, y_flt[i]) for i in range (0, ny) ])
            s = np.max(rn)
            y0_mmse_top = np.array([y_flt[i] * np.exp(rn[i] - s) for i in range(0, ny)])
            y0_mmse_bot = np.array([np.exp(rn[i] - s) for i in range(0, ny)])
            y0_mmse = np.sum(y0_mmse_top, axis=0) / np.sum(y0_mmse_bot, axis=0)

            # y_map implementation
            y0_map = y_flt[np.argmax(p_xoy_x0)]

            # reshape
            y_mmse_list.append(y0_mmse.reshape(s_d, s_d))
            y_map_list.append(y0_map.reshape(s_d, s_d))

        y_mmse_dict[lam] = y_mmse_list
        y_map_dict[lam] = y_map_list

    def merge_images(imgs, a, b):
        imgs_grid = imgs.reshape(a, a, b, b).swapaxes(1, 2) # swap dimensions such that x-cord is next to x grid num
        return imgs_grid.reshape(a*b, a*b)

    # select k images with fixed intervall
    x_list = [x[i] for i in range(0, intervall*k, intervall)]
    x_res = merge_images(np.array(x_list), s_k, s_d)

    y_star_list = [y_star[i] for i in range(0, intervall*k, intervall)]
    y_star_res = merge_images(np.array(y_star_list), s_k, s_d)

    y_mmse_res = merge_images(np.array(y_mmse_dict[1]), s_k, s_d)
    y_map_res = merge_images(np.array(y_map_dict[1]), s_k, s_d)

    ax[0, 0].imshow(y_star_res)
    ax[0, 1].imshow(x_res)
    ax[0, 2].imshow(y_mmse_res)
    ax[0, 3].imshow(y_map_res)
    
    ax[1, 2].imshow(merge_images(np.array(y_mmse_dict[1e-2]), s_k, s_d))
    ax[1, 3].imshow(merge_images(np.array(y_map_dict[1e-2]), s_k, s_d))
    
    ax[2, 2].imshow(merge_images(np.array(y_mmse_dict[1e-3]), s_k, s_d))
    ax[2, 3].imshow(merge_images(np.array(y_map_dict[1e-3]), s_k, s_d))

    # question 7 
    # the value lambda defines how confident we are in our training data.
    # the mlp selects the same results irrespective of lambda because even though the peak is less sharp the maxima still remain equal
    # mmse on the other hand starts to interpolate more and more between the likliest results when the distribution becomes wider and less peakier
    # one could also say that mmse with lower lambda generalizes better to samples which are not part of y

    """ End of your code
    """
    return fig



if __name__ == '__main__':
    tasks = [task2, task3]
    #tasks = [task2]
    
    pdf = PdfPages('figures.pdf')
    for task in tasks:
        f = task()
        pdf.savefig(f)
    
    pdf.close()
