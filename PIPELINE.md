# PET-KLS interactive 3D network

Standalone HTML connectome generated from Schaefer-100/Yeo-17 edge-wise group comparisons.

## Input

Place the edge-wise Mann–Whitney U result table at `data/edges_mwu_bh_all.csv`, or set `EDGES_CSV` in the notebook. Required columns:

```text
label_i,label_j,rank_biserial_r,p,q
```

Each row represents one undirected ROI pair; the sign convention is positive rank-biserial r for Healthy controls > EarlyAD. ROI names must match the Schaefer-100 17-network atlas after removal of the optional `17Networks_` prefix.

## Generate

```bash
pip install -r requirements.txt
jupyter notebook PETKLS_3D_Network_Pipeline.ipynb
```

Run the three code cells in order. The default selection retains up to 30 FDR-significant positive and 30 FDR-significant negative edges, sorted by q then p. Set `VIEW_MODE = "all"` to render all qualifying edges, or adjust `TOP_POSITIVE`, `TOP_NEGATIVE`, `Q_THRESHOLD`, and `MIN_ABS_EFFECT`.

Outputs:

- `docs/index.html`: interactive 3D connectome for GitHub Pages.
- `outputs/selected_edges.csv`: exact edge list used in the visualization.

The existing 3D-network repository publishes GitHub Pages from `docs/`; commit the generated `docs/index.html` there. The notebook automatically downloads the Schaefer-100 atlas on first execution.

## Interpretation

Red edges: Healthy controls > EarlyAD. Blue edges: EarlyAD > Healthy controls. Edge magnitude corresponds to the rank-biserial effect size. Node colors identify the 17 Schaefer-Yeo networks. `EDGE_VALUE_GAIN` changes visualization contrast only; its default value of 1.0 preserves the original numerical scale.

Input data are not included in this repository.
