# @TODO: Explanation of JSE Research

import time
from tqdm import tqdm

import numpy as np
from scipy.sparse.linalg import eigsh

def calc_beta(eta):
    '''Compute beta vector with mean 1 and dispersion 0.5 from eta vector.'''
    
    mu_eta = np.mean(eta)
    d_eta = np.std(eta) / mu_eta
    c = 0.5 / d_eta
    
    return (c*eta/mu_eta)+(1-c)

def portvol(weights, cov, alpha):
    '''Compute estimated ANNUALIZED portfolio volatility from weights vector
    and returns covariance matrix.'''

    return np.sqrt(weights.T @ cov @ weights) * np.sqrt(alpha)

def portret(weights, returns, alpha):
    '''Compute estimated ANNUALIZED portfolio return from weights vector
    and returns data matrix.'''

    return np.exp( np.log1p( np.dot(weights, returns) ).mean()*alpha )-1

def minvar(beta, fvar, svar):
    '''Compute minimum variance portfolio for 
    covariance matrix of single factor model.'''
    
    bthr = 1/fvar + sum(beta**2 / svar)
    bthr = bthr / sum(beta / svar)
    w = (1 - beta/bthr) / svar
    
    return w / sum(w)

def calc_eigen(Y):
    '''Compute first eigenvector & eigenvalue of p x n returns 
    data matrix Y. Also get eigenvalue gap and fvar, svar estimates.'''

    p, n = Y.shape
    L = Y.T @ Y / p
    vals, vecs = eigsh(L, k=1, which='LA')
    valh, hvec = vals[-1], vecs[:, -1]

    h = Y @ hvec / np.sqrt(p * valh)
    el_sq = (p/n)*(np.trace(L) - np.sum(vals))/(n-1)
    Y = L = None
    
    seig = valh * p / n

    if np.sum(h) < 0:
        h = h * -1.0
    h = h / np.linalg.norm(h)

    mu_h = np.mean(h)
    fvar_est = seig*mu_h**2
    svar_est = (n/p)*el_sq

    return seig, h, el_sq, fvar_est, svar_est

def calc_jse(seig, h, el_sq, z):
    '''Compute JS-adjusted eigenvector.'''

    psi_sq = (seig - el_sq) / seig
    phz = h @ z
    blind = (phz*(1 - psi_sq))/(psi_sq - phz**2)

    hjse = h + blind*z
    hjse = hjse / np.linalg.norm(hjse)

    return hjse

def calc_portfolios(train, z):
    '''Estimate minimum-variance portfolios for PCA 
    and JSE models from train data having shape p x _.'''

    seig, h, el_sq, fvar_est, svar_est = calc_eigen(Y=train)
    train = None

    h_jse = calc_jse(seig=seig, h=h, el_sq=el_sq, z=z)
    seig = el_sq = z = None

    portfolios = {}
    for name, beta in zip(['pca', 'jse'], [h, h_jse]):
        beta /= np.mean(beta)
        portfolios[name] = minvar(beta=beta, fvar=fvar_est, svar=svar_est)

    return portfolios

def calc_realized_returns(test, portfolios, alpha):
    '''Compute realized return for PCA and JSE models, respectively.
    This is based on test data, which has shape p x _.'''

    return {name: portret(weights=portfolio, returns=test, alpha=alpha) for name, portfolio in portfolios.items()}

def calc_realized_volatilities(test, portfolios, alpha):
    '''Compute realized volatility for PCA and JSE models, respectively.
    This is based on test data, which has shape p x _.'''
    
    cov_test = (test @ test.T) / test.shape[1]
    test = None

    return {name: portvol(weights=portfolio, cov=cov_test, alpha=alpha) for name, portfolio in portfolios.items()}

def simulation(seed, beta, z, p, sigma, delta, n1, n2, f1, f2):
    '''Run one train / test simulation.'''
    
    np.random.seed(seed)
    
    train = np.outer(beta, np.random.normal(loc=0, scale=sigma/np.sqrt(f1), size=n1))
    train += np.random.normal(loc=0, scale=delta/np.sqrt(f1)*np.ones(p), size=(n1, p)).T
    
    portfolios = calc_portfolios(train=train, z=z)
    train = z = None

    test = np.outer(beta, np.random.normal(loc=0, scale=sigma/np.sqrt(f2), size=n2))
    test += np.random.normal(loc=0, scale=delta/np.sqrt(f2)*np.ones(p), size=(n2, p)).T
    
    returns = calc_realized_returns(test=test, portfolios=portfolios, alpha=f2)
    volatilities = calc_realized_volatilities(test=test, portfolios=portfolios, alpha=f2)
    test = portfolios = None

    return returns, volatilities

def simulate(p, nsim, params):
    '''Run many train / test simulations.'''
    
    np.random.seed(0)
    
    z = np.ones(shape=p) / np.sqrt(p)
    beta = calc_beta(eta=np.random.normal(loc=1, scale=1, size=p))

    return zip(*[simulation(seed=seed, beta=beta, z=z, p=p, **params) for seed in tqdm(range(1, nsim+1), leave=False)])

def main():
    '''Perform a sample run of many train / test simulations.'''

    start = time.time()
    p = 3000
    nsim = 500
    
    params = {
        'sigma': .16, 'delta': .50, 
        'n1': 252, 'f1': 252, 'n2': 3, 'f2': 12
    }

    returns, volatilities = simulate(p=p, nsim=nsim, params=params)
    print(f'Took {time.time() - start} seconds.')

    return None

if __name__ == '__main__':    
    main()
