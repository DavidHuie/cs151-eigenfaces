#!/bin/bash

rm *aux *bbl *blg *log
./clean.sh
pdflatex *tex
bibtex HRR_facialrec
pdflatex *tex
pdflatex *tex
