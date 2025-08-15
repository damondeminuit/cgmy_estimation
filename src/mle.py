import numpy as np
import pandas as pd
from scipy.special import gamma
from scipy.integrate import quad
from scipy.optimize import minimize, LinearConstraint
import matplotlib.pyplot as plt


class CGMY:
    def __init__(self, data, params, adjust_L=False):
        self.data = data
        self.L = 2.5 * np.max(np.abs(self.data))
        self.N = 2**15

        self.c, self.g, self.m, self.y, self.sigma = params
        self.mu = 0
        if adjust_L:
            if np.abs(self.mean) + 3 * np.sqrt(self.var) > np.max(np.abs(self.data)):
                self.L = 2.5 * max(
                    np.max(np.abs(self.data)), np.abs(self.mean) + 3 * np.sqrt(self.var)
                )

        if self.c < 0 or self.g < 0 or self.m < 0:
            raise ValueError("Parameters c, g and m must be positive.")

        if self.y > 2:
            raise ValueError("Parameter y must be less than 2.")

        self.m_power_y = self.m**self.y
        self.g_power_y = self.g**self.y
        self.gamma_y = gamma(-self.y)
        self.mult_cst = self.c * self.gamma_y

        self.density = pd.DataFrame({"rets": self.data})
        self.density["interval"] = ((self.data + self.L / 2) * self.N / self.L).astype(
            "int32"
        )
        self.l = np.inf

    def chara_jump(self, u, t=1):
        # Characteristic function of the jump part
        logits = (self.m - 1j * u) ** self.y + (self.g + 1j * u) ** self.y
        logits -= self.m_power_y + self.g_power_y
        logits *= t * self.mult_cst
        return np.exp(logits)

    def chara(self, u, t=1):
        # Full Characteristic function
        res = self.chara_jump(u, t)
        res *= np.exp(1j * self.mu * u * t)
        res *= np.exp(-0.5 * t * (self.sigma * u) ** 2)
        return res

    # Use of FFT take from chatgpt
    def compute_pdf_from_cf(self, N=None, L=None, t=1):
        """
        Compute PDF from characteristic function using FFT.
        N: Number of grid points (power of 2 for FFT efficiency).
        """
        if N is None:
            N = self.N
        if L is None:
            L = self.L

        ## Frequency domain grid
        f_max = np.pi * N / L  # Maximum frequency
        f = np.linspace(-f_max, f_max, N, endpoint=False)
        # Evaluate characteristic function
        phi = self.chara(f, t=t)
        # Apply FFT
        pdf = np.fft.fft(np.fft.fftshift(phi))
        # Adjust for scaling and shift
        pdf = np.fft.ifftshift(pdf) / L
        # x-axis for PDF
        x = np.linspace(-L / 2, L / 2, N, endpoint=False)
        return x, np.maximum(
            np.real(pdf), 1e-300
        )  # Return real part to avoid numerical noise

    def log_lik(self):
        _, pdf = self.compute_pdf_from_cf()
        self.density["pdf"] = pdf[self.density["interval"]]
        self.l = np.sum(np.log(self.density["pdf"]))

        return self.l

    def compute_density(self, x, t=1):
        # Compute Fourier transform to retrieve the density
        def integrand(u):
            return np.exp(-1j * x * u) * self.chara(u=u, t=t)

        res, _ = quad(integrand, -np.inf, np.inf)
        return res / (2 * np.pi)

    def sample_cgmy_negative(self, n, t=1):
        _lambda = self.c * gamma(-self.y) * (self.g**self.y + self.m**self.y)
        p = self.g**self.y / (self.g**self.y + self.m**self.y)
        rate = _lambda * t
        samples = []
        for i in range(n):
            x = 0
            nb = np.random.poisson(lam=rate)
            for j in range(nb):
                if np.random.uniform() < p:
                    x -= np.random.gamma(shape=-self.y, scale=1 / self.g)
                else:
                    x += np.random.gamma(shape=-self.y, scale=1 / self.m)
            samples.append(x)

        return samples

    def sample_cgmy_positive(self, n, L, N, t=1, sigma_prop=0.1):
        _, pdf = self.compute_pdf_from_cf(chara=self.chara_jump, N=N, L=L, t=t)

        curr = np.random.normal(0, sigma_prop)
        ix_curr = np.int64((curr + L / 2) * N / L)
        log_pdf_curr = np.log(pdf[ix_curr])
        samples = [curr]
        for i in range(n - 1):
            prop = samples[-1] + np.random.normal(0, sigma_prop)
            ix_prop = np.int64((prop + L / 2) * N / L)
            r = np.log(pdf[ix_prop]) - log_pdf_curr
            if np.log(np.random.uniform()) < r:
                curr = prop
                ix_curr = ix_prop
                log_pdf_curr = np.log(pdf[ix_curr])
            samples.append(curr)

        return samples

    def sample(self, n, L=None, N=None, t=1, sigma_prop=0.1):
        if L is None:
            L = self.L
        if N is None:
            N = self.N

        if self.y > 0:
            samples = self.sample_cgmy_positive(n, L, N, t, sigma_prop)
        else:
            samples = self.sample_cgmy_negative(n, t)

        samples = np.array(samples)
        samples = (
            samples
            + self.mu * t
            + np.sqrt(t) * self.sigma * np.random.normal(0, 1, size=n)
        )

        return samples

    @property
    def mean(self):
        return (
            self.c
            * gamma(1 - self.y)
            * (self.m ** (self.y - 1) - self.g ** (self.y - 1))
        )

    @property
    def var(self):
        return (
            self.c
            * (self.m ** (self.y - 2) + self.g ** (self.y - 2))
            * gamma(2 - self.y)
            + self.sigma**2
        )


def negative_log_lik(params, data):
    cgmy = CGMY(data, params, adjust_L=True)
    return -cgmy.log_lik()
