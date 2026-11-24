"""Figures of the report (vector PDF, LaTeX font) -> report/figures/.

    cd analysis && python3 figures.py      # runs analysis.py first (about 30 s)

All numbers come from analysis.py; this script only draws.
"""
from pathlib import Path

import numpy as np
from matplotlib.gridspec import GridSpec

import analysis as A                       # the analysis runs on import
from esr import SIGN                        # reported signal S = SIGN * X (lock-in phase 180 degrees off, see esr.py)
from style import BLUE, GRAY, ORANGE, RED, WIDTH, math_ticks, panel, plt

OUT = Path(__file__).resolve().parents[1] / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
mTV = A.mT_per_V


def save(fig, name):
    fig.savefig(OUT / name)
    plt.close(fig)
    print(name)


# =====================================================================================
# Theory: chi', chi''; phasor of the bridge; modulation (preparatory exercises)
# =====================================================================================
def fig_theory():
    with plt.rc_context({"axes.spines.top": True, "axes.spines.right": True}):     # full frame for the sketches
        _fig_theory()


def _fig_theory():
    fig = plt.figure(figsize=(WIDTH, 2.15))
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1, 0.9, 1.25], height_ratios=[0.8, 1.6], wspace=0.42, hspace=0.12)

    # (a) chi', chi''  ->  dR, dX
    ax = fig.add_subplot(gs[:, 0])
    x = np.linspace(-6, 6, 600)
    ax.plot(x, 1 / (1 + x**2), color=BLUE, label=r"$\chi'' \propto \delta R$")
    ax.plot(x, x / (1 + x**2), color=RED, label=r"$\chi' \propto \delta X$")
    ax.axhline(0, color=GRAY, lw=0.5)
    ax.set(xlabel=r"$(B-B_\mathrm{r})/\Delta B_{1/2}$", ylabel=r"$\chi/\chi''_\mathrm{max}$", xlim=(-6, 6), ylim=(-0.62, 1.12))
    ax.legend(loc="lower right", fontsize=6.6, handlelength=0.9, handletextpad=0.4, borderaxespad=0.2)
    math_ticks(ax)
    panel(ax, "a")

    # (b) phasor diagram: delta Gamma ~ chi'' + i chi' runs on a circle; the diode projects it onto the reference
    ax = fig.add_subplot(gs[:, 1])
    xx = np.linspace(-60, 60, 6000)
    z = 1 / (1 - 1j * xx)                      # chi'' + i chi'
    ax.plot(z.real, z.imag, color=BLUE)
    for xv in (-1, 0, 1):
        zv = 1 / (1 - 1j * xv)
        ax.plot(zv.real, zv.imag, "o", ms=2.6, color=BLUE)
        pos = {-1: (0.5, -0.38), 0: (0.76, 0.1), 1: (0.5, 0.38)}[xv]
        ax.text(*pos, rf"$x={xv}$", fontsize=6.3, color=BLUE, ha="center", va="center")
    ax.annotate("", xy=(1.0, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="k", lw=0.9, shrinkA=0, shrinkB=0))
    ax.annotate("", xy=(0, 0.62), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=RED, lw=0.9, shrinkA=0, shrinkB=0))
    th = np.linspace(0, np.pi / 2, 40)
    ax.plot(0.32 * np.cos(th), 0.32 * np.sin(th), color=RED, lw=0.6, ls=":")
    ax.text(0.05, 0.05, r"$\tfrac{\lambda_\mathrm{g}}{8}$", fontsize=7, color=RED, ha="left", va="bottom")
    ax.text(0.55, -0.07, r"$\varphi=0$: $\chi''$", ha="center", va="top", fontsize=7)
    ax.text(-0.05, 0.66, r"$\varphi=90^\circ$: $\chi'$", ha="left", va="bottom", fontsize=7, color=RED)
    ax.set(xlabel=r"$\mathrm{Re}\,\delta\Gamma$", ylabel=r"$\mathrm{Im}\,\delta\Gamma$", xlim=(-0.22, 1.18), ylim=(-0.72, 0.86), aspect="equal")
    ax.set_xticks([0, 0.5, 1]); ax.set_yticks([-0.5, 0, 0.5])
    math_ticks(ax)
    panel(ax, "b")

    # (c) time trace: B(t) and diode signal; the lock-in gives the derivative
    t = np.linspace(0, 1, 6000)
    Bm, fm = 0.25, 34                          # modulation (units of Delta B_1/2), cycles per sweep
    B0 = -4 + 8 * t
    Bt = B0 + Bm * np.sin(2 * np.pi * fm * t)
    a1 = fig.add_subplot(gs[0, 2])
    a1.plot(t, Bt, color="k", lw=0.6)
    a1.set(ylabel=r"$B(t)$", xlim=(0, 1), yticks=[])
    panel(a1, "c")
    a2 = fig.add_subplot(gs[1, 2], sharex=a1)
    V = 1 / (1 + Bt**2)
    a2.plot(t, V, color=GRAY, lw=0.5, label=r"Diode $\propto\chi''(B(t))$")
    lock = -2 * B0 / (1 + B0**2) ** 2 * 0.45    # first harmonic ~ B_m dchi''/dB (arbitrary scale)
    a2.plot(t, lock, color=RED, lw=1.1, label=r"Lock-in: $B_\mathrm{m}\,\mathrm{d}\chi''/\mathrm{d}B$")
    a2.axhline(0, color=GRAY, lw=0.4)
    a2.set(xlabel=r"Time $t/t_\mathrm{sweep}$", yticks=[], xlim=(0, 1), ylim=(-0.42, 1.68))
    a2.legend(loc="upper right", handlelength=1.2, fontsize=6.6, borderaxespad=0.25, labelspacing=0.25)
    math_ticks(a2)
    plt.setp(a1.get_xticklabels(), visible=False)
    save(fig, "theory.pdf")


# =====================================================================================
# Field: Hall probe, DPPH scale, line position of all scans
# =====================================================================================
def fig_field():
    fig = plt.figure(figsize=(WIDTH, 2.35))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.0, 1.0, 1.15], wspace=0.42)

    ax = fig.add_subplot(gs[0])
    ax.plot(A.U1, A.B1, ".", ms=1.4, color=BLUE, label="up")
    ax.plot(A.U3, A.B3, ".", ms=1.4, color=BLUE)
    ax.plot(A.U2, A.B2, ".", ms=1.4, color=RED, label="down")
    uu = np.linspace(0.3, 8, 200)
    ax.plot(uu, A.k_fac * A.B_cal(uu), "-", color=GRAY, lw=0.8, label="DPPH scale")
    ax.plot(A.U_Dm, A.B_D, "*", ms=7, color=ORANGE, mec="k", mew=0.3, label="DPPH", zorder=5)
    ax.set(xlabel=r"Control voltage $U$ (V)", ylabel=r"$B$ (mT)", xlim=(0, 8.3), ylim=(0, 520))
    ax.legend(loc="upper left", handlelength=1.0, fontsize=6.6)
    math_ticks(ax)
    panel(ax, "a")

    ax = fig.add_subplot(gs[1])
    win = (4.8, 5.1)
    for U, B, c in [(A.U1, A.B1, BLUE), (A.U3, A.B3, BLUE), (A.U2, A.B2, RED)]:
        s = (U > win[0]) & (U < win[1])
        ax.plot(U[s], B[s], "o", ms=2.2, color=c, mfc="none", mew=0.6)
    uu = np.linspace(*win, 50)
    ax.plot(uu, A.B_cal(uu), color=BLUE, lw=0.7)
    ax.plot(A.U19, A.B19, "s", ms=3.2, color="k", mfc="none", mew=0.7, label="Display at sample")
    ax.plot(uu, A.k_fac * A.B_cal(uu), color=GRAY, lw=0.8, ls="--", label="DPPH scale")
    ax.plot(A.U_Dm, A.B_D, "*", ms=7, color=ORANGE, mec="k", mew=0.3, zorder=5)
    ax.annotate(r"$-\qty{8.5}{\percent}$", xy=(A.U_Dm, A.B_cal(A.U_Dm)), xytext=(A.U_Dm + 0.02, A.B_cal(A.U_Dm) - 9),
                fontsize=6.8, color=BLUE, arrowprops=dict(arrowstyle="-", lw=0.5, color=BLUE))
    ax.set(xlabel=r"$U$ (V)", ylabel=r"$B$ (mT)", xlim=win, ylim=(290, 350))
    ax.legend(loc="upper left", handlelength=1.0, fontsize=6.6)
    math_ticks(ax)
    panel(ax, "b")

    # (c) line position of all DPPH scans versus time of day
    ax = fig.add_subplot(gs[2])
    time = {"07b": 13.55, "08": 13.67, "09": 13.93, "10": 14.03, "11": 14.17, "12": 14.30, "13": 14.45,
            "14": 14.55, "15": 14.65, "16": 14.73, "16b": 14.85, "17b": 15.13, "18": 15.23, "24": 15.40,
            "25": 15.52, "26": 15.68, "27": 15.80, "20": 15.92}
    for k, t in time.items():
        f = A.FITS[k]
        v = f["dir"] * (0.952 if k in ("07b", "08") else 0.476)
        c = RED if v < 0 else (ORANGE if v > 0.7 else BLUE)
        m = "v" if v < 0 else ("^" if v > 0.7 else "o")
        ax.errorbar(t, (f["pC"][0] - A.U_Dm) * 1e3, f["eC"][0] * 1e3, fmt=m, ms=3.0, color=c, mfc="none", mew=0.7)
    ax.axhline(0, color=GRAY, lw=0.5)
    ax.text(13.5, 3, r"$v=+\qty{0.95}{mV/s}$", color=ORANGE, fontsize=6.6)
    ax.text(14.25, -6.5, r"$+\qty{0.48}{mV/s}$", color=BLUE, fontsize=6.6)
    ax.text(15.3, -25.5, r"$-\qty{0.48}{mV/s}$", color=RED, fontsize=6.6, ha="right")
    ax.set(xlabel="Time of day (h)", ylabel=r"$U_0-U_\mathrm{D}$ (mV)", ylim=(-33, 22), xlim=(13.3, 16.1))
    ax2 = ax.secondary_yaxis("right", functions=(lambda u: u * mTV / 1e3, lambda b: b * 1e3 / mTV))
    ax2.set_ylabel(r"$\Delta B$ (mT)")
    ax.tick_params(axis="y", right=False)
    math_ticks(ax, ax2)
    panel(ax, "c")
    save(fig, "field.pdf")


# =====================================================================================
# DPPH: line shape in up- and down-sweep
# =====================================================================================
def fig_lineshape():
    fig = plt.figure(figsize=(WIDTH, 2.9))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[2.6, 1], hspace=0.08, wspace=0.28)
    for j, (key, title) in enumerate([("15", "up (15)"), ("18", "down (18)")]):
        f = A.FITS[key]
        U, X = f["U"], f["X"]
        x = (U - f["pC"][0]) * mTV                       # mT relative to B(U0) of the convolution model
        a0 = fig.add_subplot(gs[0, j])
        a1 = fig.add_subplot(gs[1, j], sharex=a0)
        a0.plot(x, SIGN * X, "o", ms=2.0, color="k", mfc="none", mew=0.5, label="Data")
        a0.plot(x, SIGN * f["fL"](U, *f["pL"]), color=GRAY, lw=0.9, label="Lorentz")
        a0.plot(x, SIGN * f["fC"](U, *f["pC"]), color=BLUE, lw=1.0, label="Convolution model")
        a1.plot(x, SIGN * f["rL"], color=GRAY, lw=0.8)
        a1.plot(x, SIGN * f["rC"], color=BLUE, lw=0.9)
        a1.axhline(0, color=GRAY, lw=0.4)
        lam = f["pC"][4] * mTV
        a0.text(0.97, 0.06, title + "\n" + rf"$\lambda=\qty{{{lam:.2f}}}{{mT}}$", transform=a0.transAxes,
                ha="right", va="bottom", fontsize=7)
        a0.set(xlim=(-1.5, 2.2))
        plt.setp(a0.get_xticklabels(), visible=False)
        a1.set(xlabel=r"$B-B(U_0)$ (mT)")
        if j == 0:
            a0.set_ylabel(r"Signal $S=-X$ (\si{\micro V})")
            a1.set_ylabel("Residual")
            a0.legend(loc="upper right", fontsize=6.6, handlelength=0.9, handletextpad=0.4, borderaxespad=0.2)
        math_ticks(a0, a1)
        panel(a0, "ab"[j])
    save(fig, "lineshape.pdf")


# =====================================================================================
# Modulation
# =====================================================================================
def fig_modulation():
    fig, ax = plt.subplots(1, 3, figsize=(WIDTH, 2.15), gridspec_kw=dict(width_ratios=[1, 0.9, 1.15], wspace=0.45))
    um, pp, Gm, dGm = A.um, A.pp, A.Gm, A.dGm
    ax[0].plot(um, pp, "o", color=BLUE, ms=3.4)
    uu = np.linspace(0, 3.2, 10)
    ax[0].plot(uu, A.pp_per_V * uu, "--", color=GRAY, lw=0.8)
    ax[0].axhline(pp[4:].mean(), color=RED, lw=0.7, ls=":")
    ax[0].text(1.15, 30, rf"\qty{{{A.pp_per_V:.0f}}}{{\micro V/V}}" + "\n" + rf"from \qty{{2}}{{V}}: as \qty{{{A.U_eq_lim:.2f}}}{{V}}",
               fontsize=6.8)
    ax[0].set(xlabel=r"Sine Out (V)", ylabel=r"$S_\mathrm{pp}$ (\si{\micro V})", xlim=(0, 3.2), ylim=(0, 230))
    math_ticks(ax[0])
    panel(ax[0], "a")
    ax[1].errorbar(um, Gm * mTV * 2 / np.sqrt(3), dGm * mTV * 2 / np.sqrt(3), fmt="s", ms=3, color=BLUE, mfc="none", mew=0.7)
    ax[1].set(xlabel=r"Sine Out (V)", ylabel=r"$\Delta B_\mathrm{pp,intr}$ (mT)", xlim=(0, 3.2), ylim=(0.18, 0.32))
    math_ticks(ax[1])
    panel(ax[1], "b")
    mm, bp = A.mm, A.bp
    ax[2].plot(mm, bp[:, 0], color=BLUE, label=r"Width$/\Delta B_\mathrm{pp,intr}$")
    ax[2].plot(mm, bp[:, 1], color=RED, label=r"$S_\mathrm{pp}/$linear")
    ax[2].axvspan(A.m1_max, 4, color=GRAY, alpha=0.18, lw=0)
    ax[2].axvline(A.m_2017, color=ORANGE, lw=0.9)
    ax[2].text(1.25, 0.8, "excluded\nat \\qty{1}{V}", fontsize=6.8, va="bottom")
    ax[2].text(A.m_2017 + 0.06, 2.6, "2017 calibration", fontsize=6.8, color=ORANGE, rotation=90, va="center")
    ax[2].set(xlabel=r"$B_\mathrm{m}/\Delta B_{1/2}$", xlim=(0, 3.6), ylim=(0, 5.2))
    ax[2].legend(loc="upper left", fontsize=6.6, handlelength=1.2)
    math_ticks(ax[2])
    panel(ax[2], "c")
    save(fig, "modulation.pdf")


# =====================================================================================
# Power and reference arm
# =====================================================================================
def fig_power():
    fig, ax = plt.subplots(1, 2, figsize=(WIDTH, 2.2), gridspec_kw=dict(wspace=0.38))
    # (a) power series with X375A
    P, S, dS = A.P, A.S, A.dS
    ax[0].errorbar(P[1:], S[1:], dS[1:], fmt="o", ms=3.4, color=BLUE)
    ax[0].errorbar(P[:1], S[:1], dS[:1], fmt="o", ms=3.4, color=RED, mfc="none", mew=0.8)
    pp_ = np.logspace(-2.2, 0.1, 30)
    ax[0].plot(pp_, np.exp(A.lnA) * pp_ ** A.n_exp, "--", color=GRAY, lw=0.8)
    ax[0].set(xscale="log", yscale="log", xlabel=r"Power $P/P(\qty{2}{dB})$, main attenuator X375A", ylabel=r"$S_\mathrm{pp}$ (\si{\micro V})")
    ax[0].text(0.012, 60, rf"$\propto P^{{\num{{{A.n_exp:.2f}}}}}$", fontsize=7)
    ax[0].text(0.75, 160, "13", fontsize=6.8, color=RED, ha="right")
    panel(ax[0], "a")
    # (b) series with X382A: magnitude of the line versus attenuation, compared with a power series
    att = np.array([A.FITS[k]["x382"] for k in A.BR], float)
    mag = np.array([abs(complex(*A.FITS[k]["pC"][2:4])) for k in A.BR])
    phase = np.array([np.degrees(np.angle(SIGN * complex(*A.FITS[k]["pC"][2:4]))) for k in A.BR])
    ax[1].semilogy(att, mag, "o", ms=3.4, color=BLUE, label="measured")
    aa = np.linspace(0, 20.5, 50)
    ax[1].semilogy(aa, mag[0] * 10 ** (-aa / 10), "--", color=GRAY, lw=0.8, label=r"expected without leak ($\propto E_b$)")
    for a_, b_, p_, k in zip(att, mag, phase, A.BR):
        dy = 1.18 if k != "16b" else 0.80
        ax[1].text(a_ + 0.4, b_ * dy, rf"$\ang{{{p_:.0f}}}$", fontsize=6.0, color=BLUE, va="center")
    ax[1].set(xlabel=r"Reference-arm attenuator X382A (dB)", ylabel=r"Line amplitude (\si{\micro V})", xlim=(-1, 23.5), ylim=(2, 800))
    ax[1].legend(loc="lower left", fontsize=6.4, handlelength=1.4)
    math_ticks(ax[1])
    panel(ax[1], "b")
    save(fig, "power.pdf")


# =====================================================================================
# Dispersion
# =====================================================================================
def fig_dispersion():
    fig, ax = plt.subplots(1, 2, figsize=(WIDTH, 2.45), gridspec_kw=dict(width_ratios=[1.55, 1], wspace=0.35))
    f13, f20 = A.FITS["13"], A.FITS["20"]
    x13 = (f13["U"] - f13["pC"][0]) * mTV
    x20 = (f20["U"] - f13["pC"][0]) * mTV
    ax[0].plot(x13, SIGN * f13["X"], "o", ms=1.8, color=GRAY, mfc="none", mew=0.5, label=r"13: plunger \qty{20.00}{mm}")
    ax[0].plot(x20, SIGN * f20["X"], "o", ms=1.8, color="k", mfc="none", mew=0.5, label=r"20: plunger \qty{25.60}{mm}")
    ax[0].plot(x13, SIGN * f13["fC"](f13["U"], *f13["pC"]), color=BLUE, lw=0.9, label="Fit")
    ax[0].plot(x20, SIGN * f20["fC"](f20["U"], *f20["pC"]), color=BLUE, lw=0.9)
    ax[0].set(xlabel=r"$B-B(U_0^{(13)})$ (mT)", ylabel=r"Signal $S=-X$ (\si{\micro V})", xlim=(-1.6, 2.4))
    ax[0].legend(loc="lower right", fontsize=6.4, handlelength=1.2)
    math_ticks(ax[0])
    panel(ax[0], "a")
    c0, cd = SIGN * A.c0, SIGN * A.cd
    # 24-27 scaled to the power of 13 (X375A scales signal and reference, signal ∝ P)
    c_ref = SIGN * np.array([complex(*A.FITS[k]["pC"][2:4]) / 10 ** (-(A.FITS[k]["x375"] - 2) / 10) for k in ("24", "25", "26", "27")])
    ax[1].plot(c_ref.real, c_ref.imag, "o", ms=2.6, color=BLUE, mfc="none", mew=0.6)
    ax[1].plot([0, c0.real], [0, c0.imag], "-o", color=GRAY, ms=3, lw=1)
    ax[1].plot([0, cd.real], [0, cd.imag], "-o", color="k", ms=3, lw=1)
    th = np.linspace(np.angle(c0), np.angle(c0) + np.radians(A.dphi_th), 60)
    ax[1].plot(abs(c0) * np.cos(th), abs(c0) * np.sin(th), ":", color=GRAY, lw=0.8)
    ax[1].text(c0.real + 10, c0.imag + 40, "13", ha="center", va="bottom", fontsize=7, color=GRAY)
    ax[1].text(c_ref.real.mean() + 20, c_ref.imag.mean() - 45, "24--27", ha="center", va="top", fontsize=7, color=BLUE)
    ax[1].text(cd.real + 15, cd.imag, "20", ha="left", va="center", fontsize=7)
    ax[0].text(0.97, 0.96, rf"13$\to$20: $\ang{{{A.dphi:.0f}}}$" + "\n" + rf"24--27$\to$20: $\ang{{{A.dphi_alt:.0f}}}$" + "\n"
               + rf"WR-90: $\ang{{{A.dphi_th:.1f}}}$", transform=ax[0].transAxes, va="top", ha="right", fontsize=6.8)
    ax[1].axhline(0, color=GRAY, lw=0.4)
    ax[1].axvline(0, color=GRAY, lw=0.4)
    ax[1].set(xlabel=r"Absorption component $a$ (\si{\micro V})", ylabel=r"Dispersion component $d$ (\si{\micro V})",
              aspect="equal", xlim=(-300, 530), ylim=(-470, 90))
    math_ticks(ax[1])
    panel(ax[1], "b")
    save(fig, "dispersion.pdf")


# =====================================================================================
# MnCl2
# =====================================================================================
def fig_mncl2():
    fig = plt.figure(figsize=(WIDTH, 2.9))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[2.6, 1], hspace=0.08)
    a0 = fig.add_subplot(gs[0])
    a1 = fig.add_subplot(gs[1], sharex=a0)
    h = A.main
    B = h["B"]
    X = SIGN * A.X_M
    gs_ = A.gauss_M
    a0.plot(B, X, "o", ms=1.6, color="k", mfc="none", mew=0.4, label="Data (22)")
    a0.plot(B, SIGN * h["f"](B, *h["q"]), color=BLUE, lw=1.1, label="Lorentz (VVW), free admixture")
    a0.plot(B, SIGN * gs_["f"](gs_["B"], *gs_["q"]), color=RED, lw=0.9, ls="--", label="Gauss")
    BD = A.B_D * A.nu_M / A.nu_D
    a0.axvline(BD, color=ORANGE, lw=0.8)
    a0.axvline(h["q"][0], color=BLUE, lw=0.8, ls=":")
    a0.text(BD + 3, 0.3, r"DPPH: $g=\num{2.0036}$", color=ORANGE, fontsize=7)
    a0.text(h["q"][0] - 3, -0.3, "Line center", color=BLUE, fontsize=7, ha="right")
    a0.set(ylabel=r"Signal $S=-X$ (\si{\micro V})")
    a0.legend(loc="lower left", fontsize=6.8, handlelength=1.4)
    plt.setp(a0.get_xticklabels(), visible=False)
    a1.plot(B, X - SIGN * h["f"](B, *h["q"]), color=BLUE, lw=0.7)
    a1.plot(B, X - SIGN * gs_["f"](gs_["B"], *gs_["q"]), color=RED, lw=0.7)
    a1.axhline(0, color=GRAY, lw=0.4)
    a1.set(xlabel=r"$B$ (mT, DPPH scale)", ylabel="Residual")
    math_ticks(a0, a1)
    save(fig, "mncl2.pdf")


if __name__ == "__main__":
    fig_theory()
    fig_field()
    fig_lineshape()
    fig_modulation()
    fig_power()
    fig_dispersion()
    fig_mncl2()
