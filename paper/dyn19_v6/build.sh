#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export TEXINPUTS=".:../:${TEXINPUTS:-}"
export BIBINPUTS=".:../:${BIBINPUTS:-}"
export BSTINPUTS=".:../:${BSTINPUTS:-}"

pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex

cp main.pdf GDOR-SLAM_TMM_v6_dyn19_preprint.pdf
cp main.pdf ../GDOR-SLAM_TMM_v6_dyn19_preprint.pdf
