"""
This module will contain any statistical models that are used in the backend of the Vio program. 
"""

import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp


def lr_comp_serr(X, Y, params):
    """Compute the standard error of the slope coefficent"""
    
    n = X.shape[0]

    # Begin by computing the fit values
    fit = (X*params['b'])+params['a']

    # Now get the residuals yi-yhati
    resid = Y-fit

    # Now the standard error is given by
    s2 = (1/(n-2))*np.sum(np.multiply(resid, resid))

    sb2 = s2/np.sum(np.multiply(X-X.mean(), X-X.mean()))

    tb = params['b']/np.sqrt(sb2)

    return {'S2': s2, 'Sb2': sb2, 'tb': tb}

def AIC(s2, n, k):
    return np.log(s2)+((2*k)/n)

def BIC(s2, n, k):
    return np.log(s2)+((k*np.log(n))/n)

def RMSE(Y, fit):
    return np.sqrt(np.sum(np.multiply(Y-fit, Y-fit))/Y.shape[0])

def MAE(Y, fit):
    return np.sum(np.abs(Y-fit))/Y.shape[0]

def plot_fit(X, Y, fit):
    """Plot the fit of data superposed on the data scatter"""
    fig, ax = plt.subplots()
    ax.grid()
    ax.scatter(X, Y)
    ax.plot(X, fit)

    plt.show()

def lin_reg(X, Y, plot=True):
    """Compute the regression coefficents for the passed in data and return paramters"""

    # Define usable parameters
    Xbar = X.mean()
    YBar = Y.mean()

    # Compute Beta
    beta = np.sum(np.multiply(X-Xbar, Y-YBar))/(np.sum(np.multiply((X-Xbar), (X-Xbar))))

    # Compute Alpha
    alpha = YBar-(beta*Xbar)

    # Get the fit of the data and the residuals
    fit = (X*beta) + alpha
    resid = Y-fit

    if plot:
        plot_fit(X, Y, fit)

    return {'a': alpha, 'b': beta, 'fit': fit, 'e': resid}

def comp_SE(resids, k):
    """Compute the standard error of a OLS estimate"""
    return np.dot(np.transpose(resids), resids)/(resids.shape[0]-k)

def T_stat_beta(a_jj, s, b):
    """Compute the T test statistic for a single beta paramter in OLS estimation"""
    return b/(s*np.sqrt(a_jj))

def T_test(stat, alpha, dof, lower=False, two_sided=False):
    """Determine the p-value of test statistic assuming mean 0 of t-dist"""

    # Obtain the p-value of the test-statistic
    if lower:
        p = float(sp.stats.t.cdf(stat, dof))
    else:
        if two_sided:
            p = float(2*min(sp.stats.t.cdf(np.abs(stat), dof), 1 - sp.stats.t.cdf(np.abs(stat), dof)))
        else:
            p = float(1 - sp.stats.t.cdf(stat, dof))

    if alpha<p:
        tag = 'Accept'
    else:
        tag = 'Reject'
    
    return {'p': p, 'outcome': tag}

def multi_reg(X, Y, add_intercept=True):
    """Conduct Multiple Linear Regression on Y given X"""

    if add_intercept:
        ones = np.ones((Y.shape[0],1))
        X_temp = np.hstack((ones, X))
    else:
        X_temp = X

    n = Y.shape[0]
    k = X_temp.shape[1]

    # Compute the beta vector according to the traditional equation
    X_p = np.transpose(X_temp)
    prod_1 = np.dot(X_p,X_temp)
    prod_1 = sp.linalg.inv(prod_1)
    prod_2 = np.dot(prod_1, np.transpose(X_temp))

    # Outcomne of computations
    beta = np.dot(prod_2, Y)

    # Error metrics
    fits = np.dot(X_temp, beta)

    resids = Y-fits

    s2 = np.dot(np.transpose(resids), resids) / (n-k)

    R2 = 1 - (s2*(n-k))/np.sum(np.multiply(Y-np.mean(Y), Y-np.mean(Y)))

    p_stats = [T_stat_beta(prod_1[i,i], np.sqrt(s2), float(beta[i])) for i in range(k)]

    p_vals = [T_test(p_stats[i], 0.05, n-k, two_sided=True) for i in range(k)]

    return {'b': beta, 's2': s2, 'R2': R2, 'fit': fits, 'e': resids, 'pvals':p_vals}

def F_test(F_stat, n, g, k, alpha=0.05, lower=False,two_sided=False, RESET=False):
    
    if lower:
        if not RESET:
            p = sp.stats.f.cdf(F_stat, g, n-k)
        else:
            p = sp.stats.f.cdf(F_stat, g, n-k-g)
    else:
        if two_sided:
            if not RESET:
                p = 2*min(sp.stats.f.cdf(F_stat, g, n-k), 1-sp.stats.f.cdf(F_stat, g, n-k))
            else:
                p = 2*min(sp.stats.f.cdf(F_stat, g, n-g-k), 1-sp.stats.f.cdf(F_stat, g, n-g-k))
        else:
            if not RESET:
                p = 1-sp.stats.f.cdf(F_stat, g, n-k)
            else:
                p = 1-sp.stats.f.cdf(F_stat, g, n-g-k)

    if alpha<p:
        tag = 'Accept'
    else:
        tag = 'Reject'
    
    return {'p': p, 'outcome': tag}

def RESET_test(X, Y, fits, p, resids,alpha=0.05):
    # We begin by creating the new set of fitted values
    powers = np.transpose(np.array([np.power(fits, i+1) for i in range(1,p)]))
    X_alt  = np.concat([X]+powers, axis=1)

    # Now, we need to run a regression on this
    regress = multi_reg(X_alt, Y)

    # Now in order to compute the F-test on the null we just need to
    # grab the residuals from the latest regression
    resids_UR = regress['e']

    # Now we can pass all of our residuals into our F-test
    
    F_stat = ((np.dot(np.transpose(resids), resids_UR)-np.dot(np.transpose(resids_UR), resids_UR))/p)/((np.dot(np.transpose(resids_UR),resids_UR))/(resids_UR.shape[0]-X.shape[1]-p))

    
    return F_test(F_stat, X.shape[0],p,X.shape[1],alpha=alpha, RESET=True)

def CHOW_break(X, Y, split_idx):
    
    # Get the split up training sets
    X1 = X.iloc[0:split_idx,:]
    X2 = X.iloc[split_idx:,:]
    Y1 = Y.iloc[0:split_idx]
    Y2 = Y.iloc[split_idx:]

    # Now run the regressions
    regress_1 = multi_reg(X1, Y1)
    regress_2 = multi_reg(X2, Y2)
    regress_UR = multi_reg(X, Y)

    e_1 = np.reshape(regress_1['e'], (regress_1['e'].shape[0],1))
    e_2 = np.reshape(regress_2['e'], (regress_2['e'].shape[0],1))

    # Now get the relevant residuals 
    e_u = np.vstack((e_1,e_2))
    e_r = regress_UR['e']

    F_stat = ((np.dot(np.transpose(e_r), e_r)-np.dot(np.transpose(e_u), e_u))/X.shape[1]) \
    /(np.dot(np.transpose(e_u), e_u)/(X.shape[0]-(2*X.shape[1])))

    F_stat = F_stat[0][0]

    return F_test(F_stat, X.shape[0], X.shape[1], 2*X.shape[1], two_sided=True)

def CHOW_forecast(X, Y, split_idx):
    D = np.zeros((X.shape[0], X.shape[0]-split_idx))
    for i in range(X.shape[0]-split_idx):
        D[split_idx+i,i] = 1
    X_dummy = pd.concat([X,pd.DataFrame(D)], axis=1)

    # Regress with the dummies
    regress_dumm = multi_reg(X_dummy, Y)
    reggress_UR = multi_reg(X, Y)

    # Regress without the dummies
    regress_R = multi_reg(X, Y) 

    # Compute the F-stat
    e_R = regress_R['e']
    e_UR = reggress_UR['e']
    F_stat = ((np.dot(np.transpose(e_R), e_R)-np.dot(np.transpose(e_UR),e_UR))/(X.shape[0]-split_idx))/(np.dot(np.transpose(e_UR),e_UR)/(split_idx-X.shape[1]))

    return F_test(F_stat, X.shape[0],X.shape[0]-split_idx, X.shape[0]-split_idx+X.shape[1],two_sided=True)

def JBera(resids, alpha=0.05):
    """Perform the Jaque-Bera Test on a set of residuals"""

    # Get the skewness
    S = get_skew(resids)
    # get the kurtosis
    K = get_kurt(resids)

    # Now compute the test statistic
    JB = (np.sqrt(resids.shape[0]/6)*S)**2 + (np.sqrt(resids.shape[0]/24)*(K-3))**2

    # And get the p value
    p = 1 - sp.stats.chi2.cdf(JB, df=2)

    if alpha<p:
        tag = 'Accept'
    else:
        tag = 'Reject'
    
    return {'p': p, 'outcome': tag}

def get_kurt(X):
    """Compute the sample kurtosis of a data sample X"""

    # first standardize the data 
    std = np.std(X)
    mu = np.mean(X)
    Z = (X-mu)/std

    # Now get the fourth power of the Zs
    Z4 = np.power(Z, 4)

    # get n
    n = X.shape[0]

    # Now estimate 
    return ((1/n)*np.sum(Z4))-3

def get_skew(X):
    """Compute the sample skewness of a data sample X"""

    # Get the sample mean
    mu = np.mean(X)

    # Create a diff series
    diffX = X-mu

    # get n
    n = X.shape[0]

    # Now estimate 
    return ((1/n)*np.sum(np.power(diffX, 3)))/(((1/n)*np.sum(np.power(diffX, 2)))**(3/2))
