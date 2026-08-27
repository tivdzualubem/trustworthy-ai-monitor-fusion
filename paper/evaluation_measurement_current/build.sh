#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
OUT="$ROOT/paper/Evaluation_Measurement_Runtime_Safety_Monitor_Cascades.pdf"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

ENGINE=""

if command -v latexmk >/dev/null 2>&1; then
  ENGINE="latexmk"
  latexmk \
    -pdf \
    -interaction=nonstopmode \
    -halt-on-error \
    -outdir="$TMP" \
    "$HERE/main.tex" >/dev/null
elif command -v pdflatex >/dev/null 2>&1; then
  ENGINE="pdflatex"
  pdflatex \
    -interaction=nonstopmode \
    -halt-on-error \
    -output-directory="$TMP" \
    "$HERE/main.tex" >/dev/null
  pdflatex \
    -interaction=nonstopmode \
    -halt-on-error \
    -output-directory="$TMP" \
    "$HERE/main.tex" >/dev/null
elif command -v lualatex >/dev/null 2>&1; then
  ENGINE="lualatex"
  lualatex \
    -interaction=nonstopmode \
    -halt-on-error \
    -output-directory="$TMP" \
    "$HERE/main.tex" >/dev/null
  lualatex \
    -interaction=nonstopmode \
    -halt-on-error \
    -output-directory="$TMP" \
    "$HERE/main.tex" >/dev/null
elif command -v xelatex >/dev/null 2>&1; then
  ENGINE="xelatex"
  xelatex \
    -interaction=nonstopmode \
    -halt-on-error \
    -output-directory="$TMP" \
    "$HERE/main.tex" >/dev/null
  xelatex \
    -interaction=nonstopmode \
    -halt-on-error \
    -output-directory="$TMP" \
    "$HERE/main.tex" >/dev/null
else
  echo "No LaTeX engine is available. Install either latexmk or a TeX engine such as pdflatex." >&2
  exit 2
fi

if [[ ! -s "$TMP/main.pdf" ]]; then
  echo "LaTeX engine completed without producing main.pdf." >&2
  exit 3
fi

cp "$TMP/main.pdf" "$OUT"
echo "CURRENT_REPORT_BUILD=PASS"
echo "build_engine=$ENGINE"
echo "output=$OUT"
