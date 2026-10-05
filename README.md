GSA2 is two-sided associative memory correction in linear attention (arXiv 2610.02816).

pip install -e .

pytest -q

The recurrences match Eqs. 3 and 9-13 and Table 1. The block uses a short causal conv of width 4 and a rank-16 slot projection; the paper does not state those widths. The output gate is one linear map plus sigmoid, not the figure's low-rank two-layer gate. Chunkwise parallel training is not included. The toy train.py is not the 1.3B language-model experiment.
