"""ESR analysis of the measurement on 23 October 2026. Computes every number in the report and writes them to
results/results.json; the fit parameters of all DPPH scans go to results/dpph_fits.csv.

    cd analysis && python3 analysis.py
"""
import json

import numpy as np
from scipy.optimize import brentq, curve_fit, least_squares, minimize_scalar
from scipy.signal import fftconvolve, hilbert
from scipy.special import dawsn

from esr import (DG_DPPH, DU, G_DPPH, H, K_B, MU_B, NU_DPPH, NU_MNCL2, RESULTS, SIGN, _abs, _disp, b_res, calibration,
                 fit, g_from, hall_probe, initial_guess, line, line_conv, line_gauss, line_kernel, load)

RESULTS.mkdir(exist_ok=True)
R = {}            # all numerical results -> results/results.json


# =====================================================================================
# 1  Field: Hall probe, DPPH scale, hysteresis
# =====================================================================================
p_cal, s_cal = calibration()
nu_D, dnu_D = NU_DPPH.mean(), NU_DPPH.std(ddof=1)
B_D = b_res(nu_D) * 1e3                                   # mT
dB_D = B_D * DG_DPPH / G_DPPH
B_cal = lambda U: np.polyval(p_cal, U)                     # noqa: E731
slope = lambda U: np.polyval(np.polyder(p_cal), U)         # noqa: E731

U1, B1 = hall_probe("01_calibration_up1.txt")
U2, B2 = hall_probe("02_calibration_down.txt")
U3, B3 = hall_probe("03_calibration_up2.txt")
o = np.argsort(U2)
hyst = lambda U: np.interp(U, U2[o], B2[o]) - B_cal(U)     # noqa: E731   down-sweep minus up-sweeps

# Measurement 19: gaussmeter display at the sample position
U19, B19 = np.array([4.90, 4.95, 5.00]), np.array([324.0, 326.0, 328.0])
# Analog output and display read at the same time (lab notebook 10:38-10:53), at 0.1, 3.5, 4.0, 5.5 and 8.0 V
AN_M, DI_M = np.array([24.3, 225.3, 252.7, 340.0, 454.5]), np.array([25.0, 242.0, 269.0, 361.0, 483.0])
(disp_fac, disp_off), C_disp = np.polyfit(AN_M, DI_M, 1, cov=True)

# =====================================================================================
# 2  DPPH: line fits of all scans
# =====================================================================================
#           file                         mod (V) X382A X375A direction
SCANS = {"07b": ("07b_DPPH_mod3V.txt",      3.0, 1, 2, +1),
         "08":  ("08_DPPH_mod2V.txt",       2.0, 1, 2, +1),
         "09":  ("09_DPPH_mod1V.txt",       1.0, 1, 2, +1),
         "10":  ("10_DPPH_mod0p5V.txt",     0.5, 1, 2, +1),
         "11":  ("11_DPPH_mod0p25V.txt",    0.25, 1, 2, +1),
         "12":  ("12_DPPH_mod0p1V.txt",     0.1, 1, 2, +1),
         "13":  ("13_DPPH_P0dB.txt",        1.0, 0, 2, +1),
         "14":  ("14_DPPH_P3dB.txt",        1.0, 3, 2, +1),
         "15":  ("15_DPPH_P6dB.txt",        1.0, 6, 2, +1),
         "16":  ("16_DPPH_P10dB.txt",       1.0, 10, 2, +1),
         "16b": ("16b_DPPH_P10dB.txt",      1.0, 10, 2, +1),
         "17b": ("17b_DPPH_P20dB.txt",      1.0, 20, 2, +1),
         "18":  ("18_DPPH_P6dB_down.txt",   1.0, 6, 2, -1),
         "20":  ("20_DPPH_dispersion.txt",  1.0, 0, 2, +1),
         "24":  ("24_DPPH_X375A_6dB.txt",   1.0, 0, 6, +1),
         "25":  ("25_DPPH_X375A_10dB.txt",  1.0, 0, 10, +1),
         "26":  ("26_DPPH_X375A_15dB.txt",  1.0, 0, 15, +1),
         "27":  ("27_DPPH_X375A_20dB.txt",  1.0, 0, 20, +1)}

FITS = {}
for key, (file, u_mod, x382, x375, direction) in SCANS.items():
    U, X, _, _ = load(file)
    X = X * 1e6                                               # µV
    p0 = initial_guess(U, X)
    if key in ("16", "16b", "17b", "20"):
        p0 = [U[np.argmax(X)], 0.006, 0.0, X.max()]           # mainly dispersion
    pL, eL, rL, fL = fit(U, X, line, p0)
    _, _, rG, _ = fit(U, X, line_gauss, [pL[0], pL[1], pL[2], pL[3]])
    # Convolution model: several starting phases, keep the best fit (the cost function has local minima)
    c = np.hypot(pL[2], pL[3])
    best = None
    for dphi in (0, 45, 90, -45):
        ph = np.arctan2(pL[3], pL[2]) + np.radians(-dphi)
        try:
            res = fit(U, X, line_conv, [pL[0] - 0.0015, 0.0036, c * np.cos(ph), c * np.sin(ph), 0.012],
                      bounds=([-np.inf, 5e-4, -np.inf, -np.inf, 0], [np.inf, 0.05, np.inf, np.inf, 0.05]))
        except RuntimeError:
            continue
        if best is None or res[2].std() < best[2].std():
            best = res
    pC, eC, rC, fC = best
    FITS[key] = dict(U=U, X=X, mod=u_mod, x382=x382, x375=x375, dir=direction,
                     pp=X.max() - X.min(), pL=pL, eL=eL, rL=rL, fL=fL, pC=pC, eC=eC, rC=rC, fC=fC,
                     rms_L=rL.std(), rms_G=rG.std(), rms_C=rC.std())

with open(RESULTS / "dpph_fits.csv", "w") as fh:
    # signed amplitudes and phases refer to the reported signal S = SIGN * X (see esr.py)
    fh.write("scan,mod_V,X382A_dB,X375A_dB,direction,pp_uV,"
             "U0_L,G_L_mV,phi_L_deg,rms_L,rms_G,"
             "U0_C,dU0_C,G_C_mV,dG_C_mV,lam_mV,dlam_mV,a_C,d_C,phi_C_deg,magnitude_C,rms_C\n")
    for k, f in FITS.items():
        pL, pC, eC = f["pL"], f["pC"], f["eC"]
        # full precision: rounding happens only once, in tables.py
        fh.write(f"{k},{f['mod']},{f['x382']},{f['x375']},{f['dir']},{f['pp']:.5f},"
                 f"{pL[0]:.7f},{pL[1]*1e3:.5f},{np.degrees(np.arctan2(SIGN*pL[3], SIGN*pL[2])):.4f},{f['rms_L']:.6f},{f['rms_G']:.6f},"
                 f"{pC[0]:.7f},{eC[0]:.7f},{pC[1]*1e3:.5f},{eC[1]*1e3:.5f},{pC[4]*1e3:.5f},{eC[4]*1e3:.5f},"
                 f"{SIGN*pC[2]:.5f},{SIGN*pC[3]:.5f},{np.degrees(np.arctan2(SIGN*pC[3], SIGN*pC[2])):.4f},{np.hypot(pC[2], pC[3]):.5f},"
                 f"{f['rms_C']:.6f}\n")


def gamma(g):
    """Gyromagnetic ratio g muB/hbar of the sample (rad/(s T))."""
    return g * MU_B / (H / (2 * np.pi))


GAMMA_D = gamma(G_DPPH)


def wmean(x, e):
    """Weighted mean; its uncertainty is the larger of the internal and the external one."""
    w = 1 / np.asarray(e) ** 2
    m = np.sum(w * x) / w.sum()
    return m, max(np.sqrt(1 / w.sum()), np.std(x, ddof=1) / np.sqrt(len(x)))


# --- Field scale k from DPPH. Reference scans: modulation <= 1 V, up-sweep, mainly absorption
REF = ["09", "10", "11", "12", "13", "14", "15", "24", "25", "26", "27"]
U_D = np.array([FITS[k]["pC"][0] for k in REF])
U_Dm = U_D.mean()
k_fac = B_D / B_cal(U_Dm)
mT_per_V = k_fac * slope(U_Dm)                               # true field slope at the line
uu_h = np.linspace(1.0, 5.5, 451)
R["field"] = dict(nu_DPPH_GHz=nu_D / 1e9, sigma_nu_kHz=dnu_D / 1e3, B_DPPH_mT=B_D, dB_DPPH_mT=dB_D,
                  calibration_poly_mT=p_cal.tolist(), calibration_scatter_mT=s_cal,
                  hysteresis_1_to_5p5V_mT=[float(hyst(uu_h).min()), float(hyst(uu_h).max())],
                  U_DPPH_V=U_Dm, U_DPPH_scatter_V=U_D.std(ddof=1), B_hall_at_U_DPPH_mT=float(B_cal(U_Dm)),
                  k=k_fac, mT_per_V=mT_per_V, display_over_analog=float(disp_fac),
                  g_apparent_analog=float(g_from(nu_D, B_cal(U_Dm) * 1e-3)),
                  g_apparent_display=float(g_from(nu_D, np.interp(U_Dm, U19, B19) * 1e-3)),
                  U_DPPH_07b_08_V=float(np.mean([FITS[k]["pC"][0] for k in ("07b", "08")])))

# --- Line position and field lag. Time at which each scan passed the line (lab notebook)
TIME = {"07b": 13 + 33 / 60, "08": 13 + 40 / 60, "09": 13 + 56 / 60, "10": 14 + 4 / 60, "11": 14 + 11 / 60,
        "12": 14 + 19 / 60, "13": 14 + 29 / 60, "14": 14 + 34 / 60, "15": 14 + 39 / 60, "16": 14 + 44 / 60,
        "16b": 15 + 2 / 60, "17b": 15 + 9 / 60, "18": 15 + 17 / 60, "24": 15 + 25 / 60, "25": 15 + 33 / 60,
        "26": 15 + 41 / 60, "27": 15 + 48 / 60, "20": 15 + 55 / 60}
START = {k: FITS[k]["U"][0] for k in FITS}
SAME = ["13", "14", "15", "16", "16b", "17b", "24", "25", "26", "27", "20"]   # start 4.910 V, 60 s, 0.48 mV/s, 1 V
t_h = np.array([TIME[k] for k in SAME]) * 60                                    # min
u_h = np.array([FITS[k]["pC"][0] for k in SAME])
p_dr = np.polyfit(t_h, u_h, 1)                                                 # slow drift
rms_dr = np.sqrt(np.sum((u_h - np.polyval(p_dr, t_h)) ** 2) / (len(u_h) - 2))
offset = {k: FITS[k]["pC"][0] - np.polyval(p_dr, TIME[k] * 60) for k in ("10", "11", "12", "07b", "08", "18")}
gain = {k: offset[k] / (4.910 - START[k]) for k in ("10", "11", "12", "07b", "08")}   # -dU_line/dU_start
f_lag = np.array([v / (1 + v) for v in gain.values()])                               # lag per mV swept
# The offsets before 14:29 rest on the drift extrapolated linearly. Limiting cases: no drift before 14:29, or 1.5 times
# the drift.
t0_dr = TIME["13"] * 60


def trend(t, factor=1.0):
    return np.polyval(p_dr, t) if t >= t0_dr else np.polyval(p_dr, t0_dr) + factor * p_dr[0] * (t - t0_dr)


f_drift = {}
for fac_ in (0.0, 1.0, 1.5):
    v_ = {k: FITS[k]["pC"][0] - trend(TIME[k] * 60, fac_) for k in ("10", "11", "12", "07b", "08")}
    g_ = [v_[k] / (4.910 - START[k]) for k in ("10", "11", "12", "07b", "08")]
    f_drift[f"{fac_:.1f}"] = float(np.mean([x / (1 + x) for x in g_]))
df_scatter = float(f_lag.std(ddof=1))
df_drift = float(np.ptp([f_drift[k_] for k_ in ("0.0", "1.0", "1.5")]) / 2)


def zero_crossing(U, X):
    """Model-free line position: zero between the extrema (baseline from the edges)."""
    X = X - np.median(np.r_[X[:8], X[-8:]])
    i, j = sorted((np.argmin(X), np.argmax(X)))
    for n in range(i, j):
        if X[n] * X[n + 1] <= 0:
            return U[n] - X[n] * (U[n + 1] - U[n]) / (X[n + 1] - X[n])
    return np.nan


zc_minus_U0 = np.array([zero_crossing(FITS[k]["U"], FITS[k]["X"]) - FITS[k]["pC"][0] for k in REF])
# Uncertainty of the slope at the line: degree of the calibration polynomial (3-6), local straight lines through
# 01 and 03 (+-0.3/0.5/0.7 V), and an offset error instead of a scale error of the display
s_var = [k_fac * np.polyval(np.polyder(calibration(degree=n)[0]), U_Dm) for n in (3, 4, 5, 6)]
for U_, B_ in ((U1, B1), (U3, B3)):
    for w_ in (0.3, 0.5, 0.7):
        sel_ = np.abs(U_ - U_Dm) <= w_
        s_var.append(k_fac * np.polyfit(U_[sel_], B_[sel_], 1)[0])
s_var = np.array(s_var)
ds_poly = np.ptp(s_var) / 2
ds_offset = abs(slope(U_Dm) * disp_fac - mT_per_V)
ds_rel = np.hypot(ds_poly, ds_offset) / mT_per_V
R["field"]["lag"] = dict(drift_rms_mV=rms_dr * 1e3, offset_mV={k: v * 1e3 for k, v in offset.items()},
                         f_lag=f_lag.tolist(), f_mean=f_lag.mean(), f_drift_variants=f_drift,
                         df_scatter=df_scatter, df_drift=df_drift, df_total=float(np.hypot(df_scatter, df_drift)),
                         zc_minus_U0_mV=zc_minus_U0.mean() * 1e3, slope_variants_mT_per_V=s_var.tolist(),
                         dslope_poly_mT_per_V=ds_poly, dslope_offset_mT_per_V=ds_offset, dslope_rel=ds_rel)

# --- Line width, T2, field distribution
CLEAN = ["09", "10", "11", "12", "15", "24", "25", "26", "27"]
G_C, dG_C = wmean([FITS[k]["pC"][1] for k in CLEAN], [FITS[k]["eC"][1] for k in CLEAN])
G_L = np.mean([FITS[k]["pL"][1] for k in CLEAN])
lam, dlam = wmean([FITS[k]["pC"][4] for k in CLEAN], [FITS[k]["eC"][4] for k in CLEAN])
dBpp_L = 2 * G_L / np.sqrt(3) * mT_per_V
T2_L = 2 / (np.sqrt(3) * GAMMA_D * dBpp_L * 1e-3)


# --- Shape of the field distribution: its shape near the edge also sets G. A gamma distribution (free shape k) and a
# half-Cauchy distribution fit as well as the exponential distribution; their G give the model uncertainty.
def fit_kernel(key, shape):
    f = FITS[key]
    U, X, p = f["U"], f["X"], f["pC"]
    best = None
    if shape == "gamma":
        model = lambda u, U0, G, a, d, lm, k: line_kernel(u, U0, G, a, d, lm, k, "gamma")   # noqa: E731
        starts = [list(p[:5]) + [k0] for k0 in (0.8, 1.2, 2.0)]
        lo, hi = [-np.inf, 5e-4, -np.inf, -np.inf, 2e-4, 0.3], [np.inf, 0.05, np.inf, np.inf, 0.05, 6]
    else:
        model = lambda u, U0, G, a, d, lm: line_kernel(u, U0, G, a, d, lm, 1, "cauchy")     # noqa: E731
        starts = [[p[0], p[1] * gf, p[2], p[3], p[4] * lf] for gf in (0.8, 1.0) for lf in (0.15, 0.4)]
        lo, hi = [-np.inf, 5e-4, -np.inf, -np.inf, 2e-4], [np.inf, 0.05, np.inf, np.inf, 0.05]
    for p0 in starts:
        try:
            q, e, r, _ = fit(U, X, model, p0, bounds=(lo, hi))
        except (RuntimeError, ValueError):
            continue
        if best is None or r.std() < best[2]:
            best = (q, e, r.std())
    return best


KERNEL = {shape: {k: fit_kernel(k, shape) for k in CLEAN} for shape in ("gamma", "cauchy")}
G_shape = {"exp": G_C,
           "gamma": np.mean([KERNEL["gamma"][k][0][1] for k in CLEAN]),
           "cauchy": np.mean([KERNEL["cauchy"][k][0][1] for k in CLEAN])}
rms_shape = {"gamma": np.mean([KERNEL["gamma"][k][2] / FITS[k]["rms_C"] for k in CLEAN]),
             "cauchy": np.mean([KERNEL["cauchy"][k][2] / FITS[k]["rms_C"] for k in CLEAN])}
G_mid = (max(G_shape.values()) + min(G_shape.values())) / 2
dG_model = (max(G_shape.values()) - min(G_shape.values())) / 2
dBpp_mid = 2 * G_mid / np.sqrt(3) * mT_per_V
T2_mid = 2 / (np.sqrt(3) * GAMMA_D * dBpp_mid * 1e-3)
d_rel_dpph = np.sqrt((dG_model / G_mid) ** 2 + (dG_C / G_mid) ** 2 + ds_rel**2)    # model, statistics, slope
R["dpph"] = dict(G_shape_mV={f_: g_ * 1e3 for f_, g_ in G_shape.items()}, rms_shape_over_exp=rms_shape,
                 G_mid_mV=G_mid * 1e3, dG_model_mV=dG_model * 1e3, dG_stat_mV=dG_C * 1e3,
                 dBpp_mT=dBpp_mid, d_dBpp_mT=dBpp_mid * d_rel_dpph, d_dBpp_model_mT=dBpp_mid * dG_model / G_mid,
                 d_dBpp_slope_mT=dBpp_mid * ds_rel, d_dBpp_stat_mT=dBpp_mid * dG_C / G_mid,
                 T2_ns=T2_mid * 1e9, dT2_ns=T2_mid * 1e9 * d_rel_dpph,
                 T2_with_lag_ns=T2_mid * 1e9 / (1 - f_lag.mean()),          # field rising (1 - f) times slower
                 dBpp_lorentz_mT=dBpp_L, T2_lower_bound_ns=T2_L * 1e9,
                 lam_mV=lam * 1e3, dlam_mV=dlam * 1e3, lam_mT=lam * mT_per_V, lam_over_B0=lam * mT_per_V / B_D,
                 lam_15_mV=FITS["15"]["pC"][4] * 1e3, lam_18_mV=FITS["18"]["pC"][4] * 1e3,
                 rms_lorentz_over_conv=np.mean([FITS[k]["rms_L"] / FITS[k]["rms_C"] for k in CLEAN]),
                 rms_gauss_over_lorentz=np.mean([FITS[k]["rms_G"] / FITS[k]["rms_L"] for k in CLEAN]))

# =====================================================================================
# 3  Modulation: how large is B_m?
# =====================================================================================
MOD = ["12", "11", "10", "09", "08", "07b"]
um = np.array([FITS[k]["mod"] for k in MOD])
pp = np.array([FITS[k]["pp"] for k in MOD])
Gm = np.array([FITS[k]["pC"][1] for k in MOD])
dGm = np.array([FITS[k]["eC"][1] for k in MOD])
pp_per_V = np.sum(um[:4] * pp[:4]) / np.sum(um[:4] ** 2)   # line through the origin, 0.1 ... 1 V
U_eq_lim = pp[4:].mean() / pp_per_V                         # 2 and 3 V act like ... V


def first_harmonic(x, m):
    """First harmonic of the Lorentzian absorption 1/(1+x^2) (x in units of the HWHM) at modulation amplitude m,
    in closed form: (1/pi) int_0^2pi cos(t) dt/(w + m cos t) = (2/m)(1 - 1/sqrt(1 - m^2/w^2)), w = x - i; imaginary
    part. For m -> 0 it equals m d/dx [1/(1+x^2)]."""
    w = np.asarray(x) - 1j
    return np.imag(2 / m * (1 - 1 / np.sqrt(1 - m**2 / w**2)))


def width_and_pp(m):
    """Apparent width (peak-to-peak distance / small-signal value 2/sqrt3) and pp amplitude / (m 3 sqrt3/4)."""
    if m < 1e-6:
        return 1.0, 1.0
    r = minimize_scalar(lambda t: first_harmonic(t, m), bounds=(1e-6, 10 + 2 * m), method="bounded",
                        options=dict(xatol=1e-12))
    return r.x * np.sqrt(3), 2 * abs(first_harmonic(r.x, m)) / (m * 3 * np.sqrt(3) / 4)


mm = np.linspace(0.02, 3.6, 90)
bp = np.array([width_and_pp(m) for m in mm])                # G_app/G and pp/(m * small signal)
G0, dG0 = Gm[:4].mean(), Gm[:4].std(ddof=1) / 2
Gs, dGs = Gm[4:].mean(), np.hypot(Gm[4:].std(ddof=1) / np.sqrt(2), dGm[4:].mean())
ratio = Gs / G0
d_ratio = ratio * np.hypot(dG0 / G0, dGs / Gs)
ratio_limit = ratio + 2 * d_ratio                           # 2 sigma


def m_limited(m1):
    """Limited amplitude (units of G) for the model B_m = m1 G (Sine Out/V) up to 1 V: it has to produce the measured
    pp ratio U_eq_lim; pp grows like m r(m), r = pp/linear, and pp_per_V is the line through the origin."""
    r = np.array([width_and_pp(u * m1)[1] for u in um[:4]])
    target = U_eq_lim * m1 * np.sum(um[:4] ** 2 * r) / np.sum(um[:4] ** 2)
    m_top = mm[np.argmax(mm * bp[:, 1])]                     # pp of the first harmonic is largest there
    return brentq(lambda ms: ms * width_and_pp(ms)[1] - target, 1e-4, m_top)


def width_ratio(m1):
    """Width in the limited state relative to the mean of the scans with 0.1 ... 1 V, if B_m(1 V) = m1 G."""
    ref = np.mean([width_and_pp(u * m1)[0] for u in um[:4]])
    return width_and_pp(m_limited(m1))[0] / ref


m1_max = brentq(lambda m: width_ratio(m) - ratio_limit, 0.01, 0.4)   # B_m(1 V)/G, upper 2-sigma limit (Lorentzian)


# The same limit for the measured line profile (Lorentzian convolved with the field distribution) instead of a pure
# Lorentzian line: first harmonic of the profile averaged over the modulation period, fitted again with the
# small-signal model -> apparent G(m) and pp(m); then inverted as above.
def antiderivative_conv(U, U0, G, a, d, lam):
    """G [a chi'' + d chi'] convolved as in line_conv: antiderivative of the model with respect to U."""
    g = np.arange(U.min() - 10 * G, U.max() + 2 * G, DU)
    x = (g - U0) / G
    f = G * (a * _abs(x) + d * _disp(x))
    k = np.exp(-np.arange(0, 12 * lam, DU) / lam)
    f = fftconvolve(f, k / k.sum())[: len(g)]
    return np.interp(U, g, f)


def harmonic_conv(U, p, m, n=64):
    """First harmonic / m of the convolved profile (m in V of the control voltage); for m -> 0 equal to dF/dU."""
    th = 2 * np.pi * np.arange(n) / n
    g = np.arange(U.min() - m - 0.004, U.max() + m + 0.004, 2e-5)
    F = antiderivative_conv(g, *p)
    uu_ = U[:, None] + m * np.cos(th)[None, :]
    return 2 * np.mean(np.interp(uu_, g, F) * np.cos(th)[None, :], axis=1) / m


c_ref = np.mean([complex(*FITS[k]["pC"][2:4]) / abs(complex(*FITS[k]["pC"][2:4])) for k in CLEAN])
p_mod = [4.95, G_C, c_ref.real, c_ref.imag, lam]
U_mod = np.arange(4.91, 4.99 + 5e-4, 1e-3)
m_grid = np.array([1e-5, 2e-4, 4e-4, 6e-4, 8e-4, 1.0e-3, 1.2e-3, 1.5e-3, 1.8e-3, 2.2e-3, 2.6e-3, 3.0e-3])
G_app_full, pp_full = [], []
for m_ in m_grid:
    S_ = harmonic_conv(U_mod, p_mod, m_)
    q_, _, _, _ = fit(U_mod, S_, line_conv, p_mod,
                      bounds=([-np.inf, 5e-4, -np.inf, -np.inf, 0], [np.inf, 0.05, np.inf, np.inf, 0.05]))
    G_app_full.append(q_[1])
    pp_full.append(np.ptp(S_))
G_app_full = np.array(G_app_full) / G_app_full[0]
pp_full = np.array(pp_full) / pp_full[0]
G_rel_f = lambda m: np.interp(m, m_grid, G_app_full)          # noqa: E731
pp_rel_f = lambda m: np.interp(m, m_grid, pp_full)            # noqa: E731


def m_limited_full(m1):
    target = U_eq_lim * m1 * np.sum(um[:4] ** 2 * pp_rel_f(um[:4] * m1)) / np.sum(um[:4] ** 2)
    return brentq(lambda ms: ms * pp_rel_f(ms) - target, 1e-6, m_grid[-1])


def width_ratio_full(m1):
    return G_rel_f(m_limited_full(m1)) / np.mean(G_rel_f(um[:4] * m1))


m1_full = brentq(lambda m: width_ratio_full(m) - ratio_limit, 1e-5, m_grid[-1] / 2.2)   # V control voltage
# 2017 calibration from a previous report: B_m = 5.06 G/V * U_m + 1.8 G
Bm_2017 = (5.06 * 1 + 1.8) * 0.1                            # mT at 1 V
m_2017 = Bm_2017 / mT_per_V / G0
m_2017_01 = (5.06 * 0.1 + 1.8) * 0.1 / mT_per_V / G0        # at 0.1 V
pp_abs = lambda m: m * width_and_pp(m)[1]                     # noqa: E731   pp amplitude up to a factor
R["modulation"] = dict(pp_per_V=pp_per_V, U_equiv_limited_V=U_eq_lim, G_limited_over_G_linear=ratio,
                       d_ratio=d_ratio, ratio_limit_2sigma=ratio_limit,
                       Bm_1V_max_mT=m1_full * mT_per_V, Bm_limited_max_mT=m_limited_full(m1_full) * mT_per_V,
                       G_app_1V_at_limit=float(G_rel_f(m1_full)), Bm_1V_2017_mT=Bm_2017,
                       width_2017_1V_over_0p1V=width_and_pp(m_2017)[0] / width_and_pp(m_2017_01)[0],
                       pp_2017_1V_over_0p1V=pp_abs(m_2017) / pp_abs(m_2017_01),
                       pp_measured_1V_over_0p1V=float(pp[3] / pp[0]), G_measured_1V_over_0p1V=float(Gm[3] / Gm[0]),
                       dG_measured_1V_over_0p1V=float(Gm[3] / Gm[0] * np.hypot(dGm[3] / Gm[3], dGm[0] / Gm[0])))

# =====================================================================================
# 4  Power: X375A (cavity and reference) and X382A (reference arm only)
# =====================================================================================
PW = ["13", "24", "25", "26", "27"]
att = np.array([FITS[k]["x375"] for k in PW], float)
P = 10 ** (-(att - 2) / 10)                                  # relative to 2 dB
S = np.array([FITS[k]["pp"] for k in PW])                    # model-free
dS = np.array([3 * FITS[k]["rms_C"] for k in PW])
(n_exp, lnA), C_n = np.polyfit(np.log(P[1:]), np.log(S[1:]), 1, cov=True)
deficit = S[0] / np.exp(lnA)
# Absorption and dispersion part (convolution model) per power: which part is missing in scan 13?
aP = np.array([FITS[k]["pC"][2] for k in PW]) / P
dP = np.array([FITS[k]["pC"][3] for k in PW]) / P
# Saturation: expected size from B1 (stored energy in the TE102 cavity, critical coupling)
MU0 = 4e-7 * np.pi
Q_L, a_wg, b_wg = 930.0, 22.86e-3, 10.16e-3
lam0_ = 299792458.0 / nu_D
lamg_ = lam0_ / np.sqrt(1 - (299792458.0 / (2 * a_wg) / nu_D) ** 2)
V_res = a_wg * b_wg * lamg_                                   # TE102: length = lambda_g
P_res = np.array([2e-3, 5e-3, 10e-3])                         # W in the cavity, generous (source >= 13 dBm)
W_res = 2 * Q_L * P_res / (2 * np.pi * nu_D)                  # Q_U = 2 Q_L
B1_lin = np.sqrt(8 * MU0 * W_res * (lam0_ / lamg_) ** 2 / V_res)   # amplitude of the linear field at the sample
B1_rot = B1_lin / 2                                           # co-rotating part
s_exp = (GAMMA_D * B1_rot) ** 2 * T2_mid**2                   # T1 = T2 (exchange-narrowed DPPH)
R["power"] = dict(exponent_24_27=n_exp, d_exponent_24_27=float(np.sqrt(C_n[0, 0])), deficit_13=deficit,
                  absorption_13_rel=float(aP[0] / aP[1:].mean()), dispersion_13_rel=float(dP[0] / dP[1:].mean()),
                  scatter_aP_24_27=float(aP[1:].std(ddof=1) / abs(aP[1:].mean())),
                  scatter_dP_24_27=float(dP[1:].std(ddof=1) / abs(dP[1:].mean())),
                  P_cavity_mW=(P_res * 1e3).tolist(), B1_uT=(B1_rot * 1e6).tolist(), s_expected=s_exp.tolist(),
                  P_for_s_0p1_W=float(P_res[1] * 0.1 / s_exp[1]))

BR = ["13", "14", "15", "16", "16b", "17b"]
c = np.array([complex(*FITS[k]["pC"][2:4]) for k in BR])
phi_br = np.degrees(np.unwrap(np.angle(SIGN * c)))          # phase of the reported signal S = SIGN X
# DC voltage of the diode at zero field, read on the oscilloscope (lab notebook)
D2_dB = np.array([0, 2, 4, 6, 10, 15, 20.0])
D2_mV = np.array([-1.50, -0.228, 0.247, 0.351, 0.318, 0.254, 0.210])


def d2_model(A, z, V0, rho, cpsi):
    """Diode DC voltage at zero field: reference (twice through X382A, amplitude r = 10^(-A/10)) plus the leak
    rho e^(i psi) from the cavity, square-law detected (negative polarity), z = zero of the oscilloscope.
    V = z - V0 |r + rho e^(i psi)|^2 is a quadratic polynomial in r: only V0, rho cos(psi) and z - V0 rho^2 are fixed."""
    r = 10 ** (-A / 10)
    return z - V0 * (r**2 + rho**2 + 2 * r * rho * cpsi)


r_D2 = 10 ** (-D2_dB / 10)
(k2, k1, k0), C_d2 = np.polyfit(r_D2, D2_mV, 2, cov=True)
V0_d2 = -k2                                                   # reference alone at 0 dB (mV)
rho_cos = k1 / (2 * k2)                                       # in-phase leak relative to the reference
z_min = D2_mV.max()                                           # V <= z: the zero is at least the largest reading
z_D2 = 0.42                                                   # fits the line amplitudes of 13-17b (0.41-0.44 mV)
rho_d2 = np.sqrt((z_D2 - k0) / V0_d2)
cpsi_d2 = rho_cos / rho_d2
p_d2 = np.array([z_D2, V0_d2, rho_d2, cpsi_d2])
Vb = (z_D2 - d2_model(D2_dB, *p_d2))                          # bias |V_D - z| (mV)
r_D2_1 = 10 ** (-D2_dB / 20)                                  # single pass, for comparison
# Absorption part of the series 14-18 linear in the reference amplitude? a = alpha 10^(-A/10) + beta
BRa = ["14", "15", "16", "16b", "17b", "18"]
xa = np.array([10 ** (-FITS[k]["x382"] / 10) for k in BRa])
ya = np.array([FITS[k]["pC"][2] for k in BRa])
alpha_a, beta_a = np.polyfit(xa, ya, 1)
# Step between the blocks: 13 and 24-27 (scaled to 2 dB) against the line from 14-18 at r = 1
a_2dB = np.array([FITS["13"]["pC"][2]] + [FITS[k]["pC"][2] / 10 ** (-(FITS[k]["x375"] - 2) / 10) for k in ("24", "25", "26", "27")])
block_ratio = a_2dB / (alpha_a + beta_a)
R["bridge"] = dict(scan=BR, X382A_dB=[FITS[k]["x382"] for k in BR], magnitude_uV=abs(c).tolist(),
                   phase_deg=phi_br.tolist(), magnitude_0_over_20dB=float(abs(c[0]) / abs(c[-1])),
                   phase_0_to_20dB=float(phi_br[-1] - phi_br[0]),
                   absorption_alpha_uV=float(SIGN * alpha_a), absorption_beta_uV=float(SIGN * beta_a),
                   absorption_rms_uV=float((ya - np.polyval([alpha_a, beta_a], xa)).std()),
                   A_zero_absorption_dB=float(-10 * np.log10(-beta_a / alpha_a)),
                   a_16_uV=float(SIGN * FITS["16"]["pC"][2]), a_16b_uV=float(SIGN * FITS["16b"]["pC"][2]),
                   d_16_uV=float(SIGN * FITS["16"]["pC"][3]), d_16b_uV=float(SIGN * FITS["16b"]["pC"][3]),
                   D2_mV=D2_mV.tolist(), D2_V0_mV=float(V0_d2), D2_dV0_mV=float(np.sqrt(C_d2[0, 0])),
                   D2_rho_cos_psi=float(rho_cos), D2_rms_mV=float((D2_mV - np.polyval([k2, k1, k0], r_D2)).std()),
                   D2_rms_single_pass_mV=float((D2_mV - np.polyval(np.polyfit(r_D2_1, D2_mV, 2), r_D2_1)).std()),
                   D2_zero_min_mV=float(z_min), leak_lower_bound=float(-rho_cos), D2_zero_mV=float(z_D2),
                   leak_rho=float(rho_d2), leak_cos_psi=float(cpsi_d2), bias_mV=Vb.tolist(),
                   A_min_bias_dB=float(-10 * np.log10(-rho_cos)), bias_min_mV=float(V0_d2 * rho_d2**2 * (1 - cpsi_d2**2)),
                   a_line_r1_uV=float(SIGN * (alpha_a + beta_a)), a_13_24_27_at_2dB_uV=(SIGN * a_2dB).tolist(),
                   block_ratio=block_ratio.tolist(), block_ratio_mean=float(block_ratio.mean()),
                   block_step_dB=float(-10 * np.log10(block_ratio.mean())),
                   block_step_scatter_dB=float(np.std(-10 * np.log10(block_ratio), ddof=1)))

# =====================================================================================
# 5  Dispersion: plunger moved by 5.60 mm -> phase rotation, lambda_g, Hilbert transform
# =====================================================================================
c0, cd = complex(*FITS["13"]["pC"][2:4]), complex(*FITS["20"]["pC"][2:4])
dphi = np.degrees(np.angle(cd / c0))
dphi_L = (np.degrees(np.arctan2(FITS["20"]["pL"][3], FITS["20"]["pL"][2])
                     - np.arctan2(FITS["13"]["pL"][3], FITS["13"]["pL"][2])) + 180) % 360 - 180
phi_k = lambda k: np.degrees(np.arctan2(FITS[k]["pC"][3], FITS[k]["pC"][2]))   # noqa: E731
# Phase stability at a fixed bridge: scatter within 09-12 and within 24-27
s_phase = np.sqrt((np.var([phi_k(k) for k in ("09", "10", "11", "12")], ddof=1)
                   + np.var([phi_k(k) for k in ("24", "25", "26", "27")], ddof=1)) / 2)
# Reference phase: at the same plunger position (20.00 mm), 24-27 lie about 12 degrees away from 13. 13 has the same
# settings as 20, 24-27 were measured closer in time. Both are reasonable references; the difference is the systematic
# uncertainty of the rotation.
dphi_alt = (phi_k("20") - np.mean([phi_k(k) for k in ("24", "25", "26", "27")]) + 180) % 360 - 180
ref_sys = abs(dphi - dphi_alt)


# Check without a line model (Kramers-Kronig): is scan 20 in the space spanned by scan 13 and its Hilbert transform?
# For a causal response H[absorption] = dispersion, so S and H[S] give the line at any detection phase. The tails
# outside the scan window come from the convolution fit of the reference scan.
def analytic(key, edge=0.6, du=1e-3):
    f = FITS[key]
    U, X, p = f["U"], f["X"], f["pC"]
    o = np.argsort(U)
    Us, Ss = U[o], (X - (p[5] + p[6] * (U - U.mean())))[o]
    g = np.arange(Us[0] - edge, Us[-1] + edge + du / 2, du)
    full = line_conv(g, *p[:5])
    i0 = int(round((Us[0] - g[0]) / du))
    full[i0:i0 + len(Us)] = Ss
    return g, hilbert(full)


def transfer(ref, target):
    """target = k [cos D S_ref + sin D H[S_ref]](U - dU) + baseline. D is the rotation of c = a + i d."""
    g, an = analytic(ref)
    U, X = FITS[target]["U"], FITS[target]["X"]

    def f(u, D, k, dU, b0, b1):
        re, im = np.interp(u - dU, g, an.real), np.interp(u - dU, g, an.imag)
        return k * (np.cos(D) * re + np.sin(D) * im) + b0 + b1 * (u - U.mean())
    best = None
    for D0 in np.radians(np.arange(-180, 180, 30)):
        try:
            q, C = curve_fit(f, U, X, p0=[D0, np.ptp(X) / np.ptp(an.real), 0, 0, 0], maxfev=20000)
        except RuntimeError:
            continue
        r = (X - f(U, *q)).std()
        if best is None or r < best[2]:
            best = (q, C, r)
    q, C, r = best
    D, k = q[0], q[1]
    if k < 0:
        D, k = D + np.pi, -k
    return dict(D=float((np.degrees(D) + 180) % 360 - 180), dD=float(np.degrees(np.sqrt(C[0, 0]))), k=float(k), rms=float(r))


# Sign check: the model of 24 with c rotated by +30 degrees must give D = +30 degrees
_p = FITS["24"]["pC"]
_c = complex(_p[2], _p[3]) * np.exp(1j * np.radians(30))
FITS["_check"] = dict(U=FITS["24"]["U"], X=line_conv(FITS["24"]["U"], _p[0], _p[1], _c.real, _c.imag, _p[4]))
_D = transfer("24", "_check")["D"]
del FITS["_check"]
assert 20 < _D < 40, f"sign of the Hilbert transfer: {_D}"
H1320 = transfer("13", "20")
H2420 = [transfer(k, "20") for k in ("24", "25", "26")]
model_scatter = max(abs(dphi - dphi_L), abs(dphi - H1320["D"]))     # convolution, Lorentzian, Hilbert
dphi_err = np.sqrt(2 * s_phase**2 + model_scatter**2 + ref_sys**2)
c_light = 299792458.0
a_wr90 = 22.86e-3
dz = 5.60e-3
lam0 = c_light / nu_D
f_c_th = c_light / (2 * a_wr90)
lam_g_th = lam0 / np.sqrt(1 - (f_c_th / nu_D) ** 2)
dphi_th = -720 * dz / lam_g_th
lam_g = 720 * dz / abs(dphi)
lam_g_lo, lam_g_hi = 720 * dz / (abs(dphi) + dphi_err), 720 * dz / (abs(dphi) - dphi_err)
f_c_from = lambda lg: nu_D * np.sqrt(1 - (lam0 / lg) ** 2)    # noqa: E731
R["dispersion"] = dict(dphi_deg=dphi, phase_scatter_deg=s_phase, dphi_lorentz_deg=float(dphi_L),
                       dphi_ref_24_27_deg=dphi_alt, ref_systematic_deg=ref_sys, model_scatter_deg=float(model_scatter),
                       dphi_err_deg=dphi_err, magnitude_20_over_13=abs(cd) / abs(c0),
                       hilbert_13_20=H1320, hilbert_24_26_20_D=[h["D"] for h in H2420],
                       hilbert_24_26_20_rms=[h["rms"] for h in H2420], rms_free_fit_20=FITS["20"]["rms_C"],
                       lam_g_mm=lam_g * 1e3, lam_g_range_mm=[lam_g_lo * 1e3, lam_g_hi * 1e3],
                       f_c_GHz=f_c_from(lam_g) / 1e9, f_c_range_GHz=[f_c_from(lam_g_lo) / 1e9, f_c_from(lam_g_hi) / 1e9],
                       dphi_theory_deg=dphi_th, lam_g_theory_mm=lam_g_th * 1e3, f_c_theory_GHz=f_c_th / 1e9,
                       lam0_mm=lam0 * 1e3, dphi_free_space_deg=-720 * dz / lam0)

# =====================================================================================
# 6  MnCl2: an exchange-narrowed line
# =====================================================================================
U_M, X_M, _, _ = load("22_MnCl2_wide.txt")
X_M = X_M * 1e6
nu_M = NU_MNCL2.mean()
# Field reference: DPPH position for the history of scan 22 (from 0 V, run-up 2.9 V at 2.86 mV/s). Bracketed by
# 07b/08 (run-up 128 mV, 0.95 mV/s) and 06 (run-up 383 mV, 5 mV/s; zero crossing 4.985(5) V from the lab notebook,
# converted to the convention U0 of the convolution model). Uncertainty: bracket (uniform), reading, drift until 11:58.
U_0708 = R["field"]["U_DPPH_07b_08_V"]
U_06 = 4.985 - zc_minus_U0.mean()
U_D_M = (U_0708 + U_06) / 2
dU_D_M = np.sqrt(((U_06 - U_0708) / 2 / np.sqrt(3)) ** 2 + 0.004**2 + 0.010**2)


def field_M(U, UD=None, model="scale", poly=None):
    UD = U_D_M if UD is None else UD
    Bc = B_cal if poly is None else (lambda u: np.polyval(poly, u))
    if model == "scale":
        return B_D / Bc(UD) * Bc(U)
    return disp_fac * Bc(U) + (B_D - disp_fac * Bc(UD))       # display with an offset error instead of a scale error


def mn_shape(B, B0, G, shape):
    """Field derivatives of chi'' and chi' (common prefactor); x = (B-B0)/G, xm = (B+B0)/G (counter-rotating line)."""
    x, xm = (B - B0) / G, (B + B0) / G

    def dA(u):
        return -2 * u / (1 + u**2) ** 2 / G

    def dD(u):
        return (1 - u**2) / (1 + u**2) ** 2 / G
    if shape == "vvw":                # Eq. (vvw); by Kramers-Kronig chi' ∝ D(x) - D(xm)
        return dA(x) + dA(xm), dD(x) - dD(xm)
    if shape == "gauss":              # Gaussian with HWHM G and its Hilbert transform (Dawson), counter-rotating as VVW
        c = np.sqrt(np.log(2))

        def dAg(u):
            return -2 * c**2 * u * np.exp(-(c * u) ** 2) / G

        def dDg(u):
            return 2 / np.sqrt(np.pi) * c * (1 - 2 * c * u * dawsn(c * u)) / G
        return dAg(x) + dAg(xm), dDg(x) - dDg(xm)
    raise ValueError(shape)


def mn_fit(shape="vvw", nb=2, phase=None, UD=None, model="scale", window=(0.0, 0.0), dB=0.0, poly=None):
    """Signal = a dchi''/dB + d dchi'/dB + polynomial (order nb-1) in (B-330 mT)/100 mT; fixed phase: (a, d) = c(cos, sin)."""
    B = field_M(U_M, UD, model, poly) + dB
    n = len(B)
    s = slice(int(round(window[0] * n)), n - int(round(window[1] * n)))

    def m(B, *q):
        sA, sD = mn_shape(B, q[0], q[1], shape)
        if phase is None:
            a, d, k = q[2], q[3], 4
        else:
            a, d, k = q[2] * np.cos(phase), q[2] * np.sin(phase), 3
        return a * sA + d * sD + sum(q[k + j] * ((B - 330) / 100) ** j for j in range(nb))
    p0 = [330, 89 if shape != "gauss" else 65, -30 if phase is None else 30] + ([-2] if phase is None else []) + [0] * nb
    q, C = curve_fit(m, B[s], X_M[s], p0=p0, maxfev=200000)
    r = X_M[s] - m(B[s], *q)
    e = np.sqrt(np.diag(C))
    g = g_from(nu_M, q[0] * 1e-3)
    phi = np.degrees(np.arctan2(q[3], q[2])) if phase is None else np.degrees(phase)
    return dict(B=B, q=q, e=e, f=m, g=g, dg=g * e[0] / q[0], G=q[1], dG=e[1], rms=r.std(),
                aic=r.size * np.log(r @ r / r.size) + 2 * q.size, phi=phi)


main = mn_fit()                                     # Lorentzian (VVW), free dispersion admixture
gauss_M = mn_fit("gauss")
offset_M = mn_fit(model="offset")

# Robustness: fit window (10 % of the points removed at one or both ends) x baseline order 0-2 x phase free/fixed.
# Weights exp(-dAIC/2) within a window, windows weighted equally.
WINDOWS = [(0.0, 0.0), (0.1, 0.0), (0.0, 0.1), (0.1, 0.1)]
GRID = []
for wi in WINDOWS:
    row = []
    for nb in (1, 2, 3):
        for ph in (None, np.pi):
            f_ = mn_fit(nb=nb, phase=ph, window=wi)
            row.append(dict(window=wi, nb=nb, free=ph is None, g=f_["g"], dg=f_["dg"], G=f_["G"], phi=f_["phi"],
                            aic=f_["aic"], rms=f_["rms"]))
    amin = min(z["aic"] for z in row)
    wsum = sum(np.exp(-(z["aic"] - amin) / 2) for z in row)
    for z in row:
        z["daic"] = z["aic"] - amin
        z["w"] = np.exp(-z["daic"] / 2) / wsum / len(WINDOWS)
    GRID += row
g_w = sum(z["w"] * z["g"] for z in GRID)
G_w = sum(z["w"] * z["G"] for z in GRID)
g_free = np.array([z["g"] for z in GRID if z["free"]])
g_abs = np.array([z["g"] for z in GRID if not z["free"]])
G_free = np.array([z["G"] for z in GRID if z["free"]])
daic_abs = np.array([z["daic"] for z in GRID if not z["free"]])
dg_dB0 = -main["g"] / main["q"][0]                                         # per mT of a uniform field shift
dg_dUD = (mn_fit(UD=U_D_M + 1e-3)["g"] - mn_fit(UD=U_D_M - 1e-3)["g"]) / 2e-3   # per V of the reference position

# Width and T2 (gamma of the sample). Uncertainty of the width: fit window and baseline, offset instead of scale model,
# degree of the calibration polynomial (3-6).
G_M = G_w                                                                 # HWHM (mT), grid mean
FIELD_POLY = {n: mn_fit(poly=calibration(degree=n)[0]) for n in (3, 4, 5, 6)}
dG_fit = max(np.ptp(G_free) / 2, main["dG"])
dG_off_M = abs(offset_M["G"] - main["G"])
dG_poly_M = np.ptp([v["G"] for v in FIELD_POLY.values()]) / 2
dg_poly_M = np.ptp([v["g"] for v in FIELD_POLY.values()]) / 2
dG_M = np.sqrt(dG_fit**2 + dG_off_M**2 + dG_poly_M**2)

# Static susceptibility (Curie-Weiss) and radiation damping (Bloembergen-Pound: extra HWHM eta p Q_L chi0 B0/2).
# Filling factor of a thin cylinder (radius r, full height b) at the B1 maximum of the TE102 mode:
# eta = 4 (lambda0/lambda_g)^2 pi r^2/(a d).
N_MN, S_MN, T_ROOM, THETA_CW = 1.42e28, 2.5, 296.0, -3.3
CHI0_MN = MU0 * N_MN * (2.0 * MU_B) ** 2 * S_MN * (S_MN + 1) / (3 * K_B * (T_ROOM - THETA_CW))
CHI0_DPPH, PACKING = 5.7e-5, 0.5
a_wg_, d_res_ = 22.86e-3, 299792458.0 / NU_DPPH.mean() / np.sqrt(1 - (299792458.0 / (2 * 22.86e-3) / NU_DPPH.mean()) ** 2)
lam0_r = 299792458.0 / NU_DPPH.mean()
eta_r = {r_: 4 * (lam0_r / d_res_) ** 2 * np.pi * r_**2 / (a_wg_ * d_res_) for r_ in (1.0e-3, 1.25e-3, 1.5e-3)}
B_rd_M = {k_: eta_ * PACKING * 930.0 * CHI0_MN * main["q"][0] / 2 for k_, eta_ in eta_r.items()}   # mT
dg_pull = {k_: abs(dg_dB0) * v_ * 0.2 / (1 + 0.2**2) for k_, v_ in B_rd_M.items()}                 # |xi| <= 0.2

# Exchange narrowing with the non-secular terms (Kubo-Tomita), powder weights c_m/c_0 = 2/3, 1, 1, 2/3, 0:
#   Gamma = c0 [j(0) + 5/3 j(w0) + 2/3 j(2 w0)],  delta_w = c0 [k(w0)/3 + 2/3 k(2 w0)]  (> 0: line toward lower field),
# c0 = gamma^2 M2, j/k cosine/sine transform of the correlation function (Gaussian or exponential).
# Lattice: CdCl2 type, a = 3.71 Å, layer spacing 5.86 Å, ABC stacking.
a_l, c_l = 3.71, 5.86
v1, v2 = np.array([a_l, 0, 0]), np.array([-a_l / 2, a_l * np.sqrt(3) / 2, 0])
v3 = 2 / 3 * v1 + 1 / 3 * v2 + np.array([0, 0, c_l])
ii = np.arange(-14, 15)
n1_, n2_, n3_ = np.meshgrid(ii, ii, ii, indexing="ij")
r_l = np.linalg.norm(n1_[..., None] * v1 + n2_[..., None] * v2 + n3_[..., None] * v3, axis=-1).ravel()
r_l = r_l[(r_l > 1e-6) & (r_l < 40)]
S6 = {"6 neighbors": 6 / a_l**6,
      "lattice": np.sum(r_l**-6.0) + 4 * np.pi / (3 * 40.0**3) / (np.sqrt(3) / 2 * a_l**2 * c_l)}     # Å^-6
M2_M = {k_: 0.6 * (1e-7 * 2.0 * MU_B) ** 2 * S_MN * (S_MN + 1) * v_ * 1e60 for k_, v_ in S6.items()}   # T^2
GAM2 = gamma(2.0)
w0_M = 2 * np.pi * nu_M


def jk(w, we, kind):
    if kind == "gauss":
        return np.sqrt(np.pi / 2) / we * np.exp(-w**2 / (2 * we**2)), np.sqrt(2) / we * dawsn(w / (np.sqrt(2) * we))
    return 1 / we / (1 + (w / we) ** 2), (w / we) / we / (1 + (w / we) ** 2)


def width_shift(we, M2, kind, nonsecular=True):
    c0 = GAM2**2 * M2
    j0, _ = jk(0.0, we, kind)
    if not nonsecular:
        return c0 * j0, 0.0
    j1, k1 = jk(w0_M, we, kind)
    j2, k2 = jk(2 * w0_M, we, kind)
    return c0 * (j0 + 5 / 3 * j1 + 2 / 3 * j2), c0 * (k1 / 3 + 2 / 3 * k2)


B_ex, dg_dyn = [], []
for G_ in (G_M, G_M - max(B_rd_M.values())):                             # measured / without radiation damping
    for M2v in M2_M.values():
        for kind in ("gauss", "exp"):
            for ns in (False, True):
                we = brentq(lambda W: width_shift(W, M2v, kind, ns)[0] - GAM2 * G_ * 1e-3, 1e8, 1e15)
                if ns:
                    dg_dyn.append(2.0 * width_shift(we, M2v, kind, ns)[1] / w0_M)
                else:
                    B_ex.append(we / GAM2)

# Error budget of g. Reference: U0 of the convolution model is the high-field edge of the field distribution in the
# DPPH sample; if MnCl2 sees the same distribution, its mean field is lower by lambda -> g up to g lambda/B0 higher
# (half of it applied).
dg_model = np.ptp(g_free) / 2
dg_ref = abs(dg_dUD) * dU_D_M
dg_dist = g_w * lam * mT_per_V / main["q"][0] / 2
dg_dpph = g_w * DG_DPPH / G_DPPH
dg_offset = abs(offset_M["g"] - main["g"])
dg_stat = main["dg"]
dg_pull_M = dg_pull[1.25e-3]
# Demagnetization: the static field in the sample shifts by p chi0 (N - 1/3) B0, the resonance of the precessing
# magnetization by 3/2 of that (Kittel: transverse demagnetizing fields). |N - 1/3| <= 2/3, so at most p chi0 B0;
# half of it as 1 sigma.
dg_demag = g_w * CHI0_MN * PACKING / 2
g_M = g_w + dg_dist
dg_M = np.sqrt(dg_model**2 + dg_ref**2 + dg_dist**2 + dg_offset**2 + dg_pull_M**2 + dg_demag**2 + dg_stat**2
               + dg_dpph**2 + dg_poly_M**2)
T2_M = 1 / (gamma(g_M) * G_M * 1e-3)
# Other branch: if scan 22 ran from 2.00 V after the lock-in phase was set at 5.9 V, without returning to 0 V, the field
# is higher by E(U). Preisach model with broad hysterons: E(U) = O(U) - O(5.9 V), O = opening of the main loop.
E_branch = np.where(U_M < 5.9, np.clip(hyst(U_M) - hyst(5.9), 0, None), 0.0)
branch = mn_fit(nb=2, dB=k_fac * E_branch)
dg_branch = branch["g"] - main["g"]

# Search scan 22 (50 mV/s; X spoiled by a GPIB error, signal in Y because the lock-in phase was set only afterwards)
# against scan 22 (2.86 mV/s): common width and admixture, separate positions and baselines.
U_s, X_s8, Y_s, R_s8 = load("22_MnCl2_search.txt")
# The GPIB error returned X and R as 8 and Y correctly, except at 4.80 and 4.85 V, where the readout went differently.
# At 4.85 V, Y fits the curve. At 4.80 V, the X column fits the curve of Y and the Y column fits X of 4.85 V: X and Y
# are swapped there. This point is dropped.
gpib = ~((X_s8 == 8) & (R_s8 == 8))
swapped = gpib & np.isclose(U_s, 4.80)
U_s, Y_s = U_s[~swapped], Y_s[~swapped] * 1e6
B_s, B_b = field_M(U_s), main["B"]


def joint_res(p, w_s, G_b=None):
    B0b, B0s, G, phi, Ab, As, b0, b1, s0, s1 = p
    Ab_, Db_ = mn_shape(B_b, B0b, G if G_b is None else G_b, "vvw")
    As_, Ds_ = mn_shape(B_s, B0s, G, "vvw")
    mb = Ab * (np.cos(phi) * Ab_ + np.sin(phi) * Db_) + b0 + b1 * (B_b - 330) / 100
    ms = As * (np.cos(phi) * As_ + np.sin(phi) * Ds_) + s0 + s1 * (B_s - 330) / 100
    return np.r_[(mb - X_M) / main["rms"], (ms - Y_s) * w_s]


def mn_lin(B, B0, G, a, d, b0, b1):
    sA, sD = mn_shape(B, B0, G, "vvw")
    return a * sA + d * sD + b0 + b1 * (B - 330) / 100


q_s0, _ = curve_fit(mn_lin, B_s, Y_s, p0=[330, 89, -30, -2, 0, 0], maxfev=200000)
w_s = 1 / (Y_s - mn_lin(B_s, *q_s0)).std()
# The first ten points of the search scan (2.00-2.45 V) scatter about four times as much as the rest. The joint fit
# therefore weights each part with its own scatter (w_seg).
N_EDGE = 10
r_search = Y_s - mn_lin(B_s, *q_s0)
rms_edge, rms_rest = np.sqrt(np.mean(r_search[:N_EDGE]**2)), np.sqrt(np.mean(r_search[N_EDGE:]**2))
w_seg = np.where(np.arange(U_s.size) < N_EDGE, 1 / rms_edge, 1 / rms_rest)
r_s = least_squares(joint_res, [main["q"][0], q_s0[0], main["G"], np.radians(main["phi"]), 30, 30, 0, 0, 0, 0],
                    args=(w_seg,), x_scale="jac")


# Width test: as above, but separate widths, G_22 = rho G_search. The search scan started from 0 V (normal branch). On
# the other branch E(U) would fall across the line, the true field slope would be smaller, and scan 22 would appear
# wider on the normal-branch scale. Prediction: the same data and model, fitted once on each scale.
def stretch_res(p, w_s):
    B0b, B0s, G, rho, phi, Ab, As, b0, b1, s0, s1 = p
    return joint_res([B0b, B0s, G, phi, Ab, As, b0, b1, s0, s1], w_s, G_b=G * rho)


r_st = least_squares(stretch_res, np.r_[r_s.x[:3], 1.0, r_s.x[3:]], args=(w_s,), x_scale="jac")
rho_branch = float(main["G"] / branch["G"])
r_stw = least_squares(stretch_res, r_st.x, args=(w_seg,), x_scale="jac")
C_stw = np.linalg.pinv(r_stw.jac.T @ r_stw.jac) * np.sum(r_stw.fun**2) / (r_stw.fun.size - r_stw.x.size)
rho_w, drho_w = float(r_stw.x[3]), float(np.sqrt(C_stw[3, 3]))
pp_M = float(np.ptp(main["f"](main["B"], *main["q"]) - main["f"](main["B"], *np.r_[main["q"][:4] * [1, 1, 0, 0],
                                                                                   main["q"][4:]])))
R["mncl2"] = dict(
    B_range_mT=[float(main["B"].min()), float(main["B"].max())], pp_uV=pp_M, rms_over_pp=main["rms"] / pp_M,
    rms_gauss_over_lorentz=gauss_M["rms"] / main["rms"], chi0=CHI0_MN, chi0_over_dpph=CHI0_MN / CHI0_DPPH,
    G_mT=G_M, dG_mT=dG_M, dBpp_mT=2 * G_M / np.sqrt(3), T2_ps=T2_M * 1e12, dT2_ps=T2_M * 1e12 * dG_M / G_M,
    G_free_range_mT=[G_free.min(), G_free.max()], dG_fit_mT=dG_fit, dG_offset_mT=dG_off_M, dG_poly_mT=dG_poly_M,
    hwhm_dipolar_mT={k_: np.sqrt(2 * np.log(2) * v_) * 1e3 for k_, v_ in M2_M.items()},
    B_ex_secular_T=[min(B_ex), max(B_ex)], dg_dynamic=[min(dg_dyn), max(dg_dyn)],
    U_line_V=float(np.interp(main["q"][0], main["B"], U_M)),
    U_DPPH_07b_08_V=U_0708, U_DPPH_06_V=U_06, U_DPPH_ref_V=U_D_M, dU_DPPH_ref_V=dU_D_M,
    g=g_M, dg=dg_M, g_free_range=[g_free.min(), g_free.max()], g_abs_range=[g_abs.min(), g_abs.max()],
    daic_abs_min=daic_abs.min(), g_grid_mean=g_w,
    dg_model=dg_model, dg_ref=dg_ref, dg_stat=dg_stat, dg_distribution=dg_dist, dg_pulling=dg_pull_M,
    dg_demagnetization=dg_demag, dg_offset_scale=dg_offset, dg_poly=dg_poly_M, dg_dpph=dg_dpph,
    dg_sample=float(np.sqrt(dg_dist**2 + dg_pull_M**2 + dg_demag**2)),
    dg_calibration_and_dpph=float(np.sqrt(dg_offset**2 + dg_poly_M**2 + dg_dpph**2)),
    dg_stat_scale_dpph=float(np.sqrt(dg_stat**2 + dg_offset**2 + dg_poly_M**2 + dg_dpph**2)),
    g_0_mn=2.001, dg_branch=dg_branch, g_branch=g_M + dg_branch,
    gap_4p2_5p6V_mT=[float(k_fac * np.interp(u, U_M, E_branch)) for u in (4.2, 5.6)],
    width_ratio_branch_predicted=rho_branch, width_ratio=[rho_w, drho_w],
    sigma_other_branch=(rho_branch - rho_w) / drho_w, sigma_normal_branch=(1 - rho_w) / drho_w,
    search_rms_edge_over_rest=float(rms_edge / rms_rest), search_dropped_V=[4.80])

# =====================================================================================
# 7  Cu2+ ((enH2)CuCl4): no signal
# =====================================================================================
U, X, Y, _ = load("29_CuCl4.txt")
sig_X = X.std() * 1e6
nu_Cu, Q = 9.199218937e9, 930
f_cav = 9.36e9                                              # cavity with sample, about as for DPPH
factor = 1 / (1 + (2 * Q * (f_cav - nu_Cu) / f_cav) ** 2)
pp_dpph_4V = np.mean([FITS[k]["pp"] for k in ("07b", "08")])
dB_dpph_app = 2 * G_L / np.sqrt(3) * mT_per_V
R["cu"] = dict(sigma_X_uV=sig_X, nu_GHz=nu_Cu / 1e9, cavity_factor=factor,
               N_Cu_over_N_DPPH_limit_on_resonance={str(dB): 3 * sig_X / pp_dpph_4V * (dB / dB_dpph_app) ** 2
                                                    for dB in (5.0, 10.0, 30.0)})

with open(RESULTS / "results.json", "w") as fh:
    json.dump(R, fh, indent=1, ensure_ascii=False, default=float)
