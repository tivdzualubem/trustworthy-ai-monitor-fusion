#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

latexmk -xelatex -interaction=nonstopmode -halt-on-error \
  Risk_Controlled_Decision_Value_Acquisition_Report.tex

cp -f Risk_Controlled_Decision_Value_Acquisition_Report.pdf \
  Budget_Aware_Runtime_Safety_Monitor_Fusion_Report.pdf

latexmk -c Risk_Controlled_Decision_Value_Acquisition_Report.tex >/dev/null
rm -f \
  Risk_Controlled_Decision_Value_Acquisition_Report.bbl \
  Risk_Controlled_Decision_Value_Acquisition_Report.bcf \
  Risk_Controlled_Decision_Value_Acquisition_Report.blg \
  Risk_Controlled_Decision_Value_Acquisition_Report.run.xml \
  Risk_Controlled_Decision_Value_Acquisition_Report.xdv
