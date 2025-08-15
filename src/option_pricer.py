import numpy as np
from scipy.special import gamma
from scipy.integrate import quad
import matplotlib.pyplot as plt
import pandas as pd
from src_clean.mle import CGMY


class OptionPricer:
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
