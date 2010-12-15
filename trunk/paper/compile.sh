#!/bin/bash

if [ -f HRR_facialrec.aux ]
then
rm *aux *bbl *blg *log
fi

pdflatex *tex
bibtex HRR_facialrec
pdflatex *tex
pdflatex *tex
