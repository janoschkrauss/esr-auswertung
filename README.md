# ESR of DPPH and MnCl₂

Raw data, analysis code and lab report for the electron spin resonance experiment of the Physics Laboratory III at
ETH Zürich, measured on 23 October 2026.

The report is [`report/ESR_Lab_Report.pdf`](report/ESR_Lab_Report.pdf) (revised version of 24 November 2026, after the
review by the course supervisor Toni Shiroka). The first version of 31 October 2026, as it was submitted, is kept as
[`report/ESR_Lab_Report_v1_original.pdf`](report/ESR_Lab_Report_v1_original.pdf).

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
