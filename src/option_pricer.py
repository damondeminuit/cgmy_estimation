import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.special import gamma

from mle import CGMY


class MonteCarloOptionPricer:
    def __init__(self, sampler):
        self.sampler = sampler

    def __call__(self, S, K, T, r, q=0, n_samples=10**5, sigma_prop=0.1):
        self.mu_neutral = (
            r
            - q
            - np.real(np.log(self.sampler.chara_jump(-1j)))
            - 0.5 * self.sampler.sigma**2
        )
        self.samples = self.sampler.sample(L=10, n=n_samples, t=T, sigma=sigma_prop)

        payoff = S * np.exp(self.samples) - K
        payoff = np.where(payoff > 0, payoff, 0)
        return np.mean(payoff * np.exp(-r * T))


def pricing_chara(S, K, T, r, params=(0.0244, 0.0765, 7.5515, 1.2945)):
    """
    Pricing function using the characteristic function.

    Parameters:
    S : float
        Current stock price.
    K : float
        Strike price.
    T : float
        Time to maturity in years.
    r : float
        Risk-free interest rate.
    q : float
        Dividend yield.
    sigma : float, optional
        Volatility of the underlying asset (default is 0.5).

    Returns:
    --------
    price : float
        The price of the European call option.
    """
    C, G, M, Y, sigma = params
    if any([param < 0 for param in (C, G, M, sigma)]):
        raise ValueError
    if Y > 2:
        raise ValueError

    def characteristic_function(u, t=1):
        """
        Characteristic function of the risk neutral log price process
        """
        mu = (
            r
            - C * gamma(-Y) * ((M - 1) ** Y + (G + 1) ** Y - G**Y - M**Y)
            - 0.5 * sigma**2
        )
        logits = (
            t * C * gamma(-Y) * ((M - 1j * u) ** Y + (G + 1j * u) ** Y - M**Y - G**Y)
        )
        logits += 1j * u * (mu * t + np.log(S))
        logits -= 0.5 * t * (sigma * u) ** 2

        return np.exp(logits)

    # def characteristic_function(u, t=1):
    #  return np.exp(-0.5 * sigma**2 * u**2 * t) * np.exp(
    #      1j * u * (np.log(S) + (r - q - 0.5 * sigma**2) * T)
    #  )

    # Define the integrand function for the Carr-Madan formula
    def integrand_1(u):
        res = np.exp(-u * 1j * np.log(K)) * characteristic_function(u - 1j, t=T)
        res /= 1j * u * characteristic_function(-1j, t=T)
        return np.real(res)

    def integrand_2(u):
        res = np.exp(-u * 1j * np.log(K)) * characteristic_function(u, t=T)
        res /= 1j * u
        return np.real(res)

    int_1 = quad(integrand_1, 0, 100)[0]
    int_2 = quad(integrand_2, 0, 100)[0]

    return S * (0.5 + int_1 / np.pi) - K * np.exp(-r * T) * (0.5 + int_2 / np.pi)
