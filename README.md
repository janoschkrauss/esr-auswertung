# ESR of DPPH and MnCl₂

Raw data, analysis code and lab report for the electron spin resonance experiment of the Physics Laboratory III at
ETH Zürich, measured on 23 October 2026.

The report is [`report/ESR_Lab_Report.pdf`](report/ESR_Lab_Report.pdf) (revised version of 24 November 2026, after the
review by the course supervisor Toni Shiroka). The first version of 31 October 2026, as it was submitted, is kept as
[`report/ESR_Lab_Report_v1_original.pdf`](report/ESR_Lab_Report_v1_original.pdf).

## Revision after review (24 November 2026)

Corrections following the comments of Toni Shiroka:

- **Wording:** "the energy of a photon (typically in the microwave region of the EM spectrum)"; "from the resonance
  field one can determine the g-factor of the electron"; "the spin S together with its charge −e implies a magnetic
  moment"; "five $d$ electrons"; "the resonance condition of the electron spins changes the diode voltage only
  slightly"; "read its display directly"; "a thousand times smaller than the line, because the resonance is very broad".
- **Theory (Sec. 2):** "the solution" is now the steady-state solution of the Bloch equations. References were added
  for Sec. 2.2 (Bloch 1946, Abragam 1961, Slichter 1990) and the other introductory paragraphs (Weil & Bolton, Poole,
  Pozar, SR830 manual).
- **λg/4 vs. λg/8:** the two plunger settings are λg/4 apart in the path of the reference wave; because of the
  reflection geometry this is a plunger travel of λg/8 (Fig. 1 caption, exercise 2, Sec. 4.6).
- **Waveguide:** half a wavelength fits across the larger dimension a, which sets the lowest frequency the waveguide can
  propagate (cutoff frequency).
- **Fig. 2 (setup):** Aux In (ADC) and Aux Out (DAC); the Aux Out label sits on the right of its line; the plunger
  branch is on top (opposite the cavity) and the diode branch on the right, as on the real magic T; both attenuators
  are labeled with their role.
- **Lock-in time constant τ:** defined in Sec. 2.5 where it is used.
- **Sweep rate:** 0.48 mV/s of control voltage now also as field and current rate (29 µT/s, about 42 µA/s); Table 1
  has a new column with the field rate of every scan.
- **Sign of the derivative:** the lock-in phase had been set with Auto Phase on the high-field flank of the MnCl₂ line
  and was therefore 180° off. All measured lines are now shown as S = −X (Figs. 4, 7a, 8, signed amplitudes in the
  text, phases in Fig. 6b and Table 4). The fits run on the raw X; positions, widths and all results are unchanged.
  Fig. 1a (χ″ and χ′ themselves, no derivative) is kept: in a field sweep χ′ is negative below B_r.
- **Fig. 6:** caption and axis labels give the role of the attenuators: X375A main attenuator, X382A reference-arm
  attenuator.
- **Gaussmeter in AC mode:** the text now states the range and frequency used and cites the data sheet of the
  F. W. Bell 5080 (lowest range 300 G, bandwidth dc to 20 kHz); the limit was the sensitivity.
- **Pick-up coil:** according to the course supervisor, a pick-up coil is available in the lab.

Smaller changes: "effective bridge phase" (the leak from the cavity contributes to the reference phase), Table 1
and Table 4 layout, revision date on the title page.

## Contents

| Folder | Content |
|---|---|
| `data/` | raw data as recorded by the LabVIEW program |
| `analysis/` | analysis code |
| `results/` | every number in the report (`results.json`) and the fit results of all DPPH scans (`dpph_fits.csv`) |
| `report/` | LaTeX source, figures and PDF of the report |

## Data

Tab-separated text files without a header. The number at the start of each file name is the scan number from the
report and the lab notebook.

- Field calibration without a sample (`01`–`03`): column 0 = control voltage U (V), column 1 = analog output of the
  gaussmeter (V, 1 V = 1 T).
- All other scans: column 0 = U (V), columns 1–3 = lock-in X, Y, R (V). The raw X and Y have the opposite sign of
  the usual convention (lock-in phase 180° off); the report shows S = −X (`SIGN` in `analysis/esr.py`).

The rest of the file name gives the sample and the setting that was changed: `mod0p5V` = modulation of 0.5 V
(Sine Out), `P6dB` = 6 dB on the attenuator X382A in the reference arm, `X375A_10dB` = 10 dB on the main attenuator
X375A, `down` = down-sweep, `dispersion` = plunger moved by 5.60 mm. Scans 06 (search scan, lock-in overloaded) and
07 and 17 (cut off, repeated as 07b and 17b) are not used in the analysis.

## Running the analysis

```bash
pip install -r requirements.txt
cd analysis
python3 analysis.py     # all numbers -> ../results/ (about 10 s)
```

To rebuild the report:

```bash
cd report
make figures            # runs the analysis again, draws the figures and writes the appendix table
make                    # ESR_Lab_Report.pdf with latexmk
```

The figure labels are typeset with LaTeX (`text.usetex` in Matplotlib), so `make figures` needs a LaTeX installation.
Tested with Python 3.13, NumPy 2.5, SciPy 1.18 and Matplotlib 3.11.

| File in `analysis/` | Content |
|---|---|
| `esr.py` | constants, loading the data, field calibration, line models, fit routine |
| `analysis.py` | the complete analysis, in the order of the report |
| `figures.py`, `style.py` | figures of the report |
| `tables.py` | table of the DPPH fit results in the appendix |
