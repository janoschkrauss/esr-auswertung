# make          -> ESR_Lab_Report.pdf
# make figures  -> rerun the analysis, draw all figures and write the appendix table

ESR_Lab_Report.pdf: ESR_Lab_Report.tex parts/*.tex figures/*.pdf setup.tikz
	latexmk -pdf -interaction=nonstopmode -halt-on-error ESR_Lab_Report.tex

figures:
	cd ../analysis && python3 figures.py && python3 tables.py

clean:
	latexmk -c ESR_Lab_Report.tex

.PHONY: figures clean
