# Project & Thesis Guidelines

## 1. Code Audit & Bug Fixing Rules
- Before writing or updating any experimental results in the paper, audit `src/` for bugs, data leakage, hardcoded paths, unhandled exceptions, or metric calculation mismatches.
- Never silently alter core model hyperparameters without documenting the change.
- Run existing unit tests or validation scripts after applying any bug fix.

## 2. LaTeX Thesis Report (`paper/`) Rules
- Main document: `paper/main.tex`; Bibliography: `paper/references.bib`.
- Do not fabricate or estimate any metrics. Extract exact numbers only from verified logs in `results/`.
- After editing `.tex` or `.bib` files, compile using:
  `cd paper && latexmk -pdf -interaction=nonstopmode main.tex`
- Inspect `paper/main.log` for any undefined citations, overfull boxes, or missing packages and fix them immediately.
