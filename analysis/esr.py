"""Shared parts of the ESR analysis: constants, loading the raw data, field calibration, line models, fit."""
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import fftconvolve

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"

H = 6.62607015e-34
MU_B = 9.2740100783e-24
K_B = 1.380649e-23
G_DPPH, DG_DPPH = 2.0036, 0.0002

# Frequency counter readings (GHz) from the lab notebook
NU_DPPH = np.array([9.359428163, 9.359371608, 9.359354867, 9.359397036, 9.359386064,
                    9.359387208, 9.359376563, 9.359356459, 9.359355432, 9.359374192,
                    9.359358125, 9.359393299]) * 1e9
NU_MNCL2 = np.array([9.352759537, 9.352680672, 9.352682294, 9.352728502]) * 1e9


# Sign convention of the reported signal. The lock-in phase was set with Auto Phase at U = 5.9 V, on the high-field
# flank of the MnCl2 line, where the absorption derivative is negative; Auto Phase made it positive there. The raw X is
# therefore the negative of the usual signal: the derivative of an absorption line comes out negative on its rising
# (low-field) side. The fits run on the raw X; the figures, tables and all signed amplitudes in the report use
# S = SIGN * X, for which the absorption derivative is positive on the rising and negative on the falling side.
# (The line models are linear in the amplitudes a and d, so fitting S instead of X only flips their signs.)
SIGN = -1


def load(name):
    """Lock-in file: columns U (V), X, Y, R (V). The first row is dropped in every file (signal not settled)."""
    return np.loadtxt(DATA / name).T[:, 1:]


def b_res(nu, g=G_DPPH):
    """Resonance field in T."""
    return H * nu / (g * MU_B)


def g_from(nu, B):
    return H * nu / (MU_B * B)


# ---------------------------------------------------------------- field calibration
def hall_probe(file):
    d = np.loadtxt(DATA / file)[1:]        # first row dropped, as in load
    return d[:, 0], d[:, 1] * 1e3          # U (V), B (mT); analog output 1 V = 1 T


def calibration(files=("01_calibration_up1.txt", "03_calibration_up2.txt"), degree=3):
    """Polynomial B_Hall(U) in mT from the up-sweeps. Below 0.3 V the residual field dominates."""
    U, B = map(np.concatenate, zip(*(hall_probe(f) for f in files)))
    sel = U > 0.3
    p = np.polyfit(U[sel], B[sel], degree)
    return p, (B[sel] - np.polyval(p, U[sel])).std(ddof=degree + 1)


# ---------------------------------------------------------------- line models
# Complex Lorentzian susceptibility chi = 1/(x - i) with x = (U - U0)/G:
# absorption chi'' = 1/(1+x^2), dispersion chi' = x/(1+x^2), same scale.
# The lock-in (first harmonic) measures the field derivative; the bridge mixes
# chi'' and chi' with the angle phi = atan2(d, a).
def _abs(x):
    return 1 / (1 + x**2)


def _disp(x):
    return x / (1 + x**2)


def line(U, U0, G, a, d):
    """d/dx [a chi'' + d chi'] of a Lorentzian line (HWHM G). Peak-to-peak of pure absorption = 1.299 a."""
    x = (U - U0) / G
    return a * (-2 * x / (1 + x**2) ** 2) + d * ((1 - x**2) / (1 + x**2) ** 2)


def line_gauss(U, U0, s, a, d):
    from scipy.special import dawsn
    x = (U - U0) / s
    u = x / np.sqrt(2)
    return a * (-x * np.exp(-x**2 / 2)) + d * (1 - 2 * u * dawsn(u))


DU = 4e-5


def line_conv(U, U0, G, a, d, lam):
    """Lorentzian line convolved with a one-sided exponential distribution (length lam, toward higher U):
    part of the sample sees a smaller field and reaches resonance only at a higher coil current."""
    if lam < 5 * DU:
        return line(U, U0, G, a, d)
    g = np.arange(U.min() - 10 * G, U.max() + 2 * G, DU)
    f = line(g, U0, G, a, d)
    k = np.exp(-np.arange(0, 12 * lam, DU) / lam)
    f = fftconvolve(f, k / k.sum())[: len(g)]
    return np.interp(U, g, f)


def line_kernel(U, U0, G, a, d, lam, k=1.0, shape="gamma"):
    """Like line_conv, but with another one-sided field distribution: gamma distribution with shape k and scale lam
    (k = 1 is the exponential distribution) or half-Cauchy distribution with width lam."""
    g = np.arange(U.min() - 10 * G, U.max() + 2 * G, DU)
    f = line(g, U0, G, a, d)
    u = (np.arange(int(12 * lam / DU) + 1) + 0.5) * DU
    w = u ** (k - 1) * np.exp(-u / lam) if shape == "gamma" else 1 / (1 + (u / lam) ** 2)
    f = fftconvolve(f, w / w.sum())[: len(g)]
    return np.interp(U, g, f)


def fit(U, S, model, p0, window=None, bounds=None):
    """Fit model + linear baseline. Returns parameters, errors (scaled with the residual scatter), residuals, function."""
    sel = np.ones_like(U, bool) if window is None else (U > window[0]) & (U < window[1])
    U, S = U[sel], S[sel]
    um = U.mean()
    n = len(p0)

    def f(u, *p):
        return model(u, *p[:n]) + p[n] + p[n + 1] * (u - um)

    lo, hi = (-np.inf, np.inf) if bounds is None else bounds
    if bounds is not None:
        lo, hi = list(lo) + [-np.inf] * 2, list(hi) + [np.inf] * 2
    p, C = curve_fit(f, U, S, p0=list(p0) + [0, 0], bounds=(lo, hi), maxfev=40000)
    r = S - f(U, *p)
    return p, np.sqrt(np.diag(C)), r, f


def initial_guess(U, S):
    i, j = np.argmin(S), np.argmax(S)
    G = abs(U[j] - U[i]) * np.sqrt(3) / 2
    return [0.5 * (U[i] + U[j]), G, (S[j] - S[i]) / 1.299 * np.sign(U[i] - U[j]), 0]
