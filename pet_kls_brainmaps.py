#!/usr/bin/env python3
"""PET-KLS Schaefer-100 biological-map analysis pipeline.

Stages: regional graph metrics -> ROI-wise GAM -> group/dynamic spatial nulls.
All input/output paths are provided at runtime; no notebook state is required.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

LOG = logging.getLogger("pet_kls_brainmaps")
N_ROI = 100
TIME_COL = "estdtt0"

BIO_NAMES = {
    "celltype": "celltype_density_Yeo17_aggregated.csv",
    "gene": "gene-expression_Yeo17_aggregated.csv",
    "receptor": "receptor_density_Yeo17_aggregated.csv",
}
ROI_COLUMNS = ("ROI.Name_17_mod", "ROI.Name_17", "ROI", "roi", "roi_name", "ROI.Name")
METRICS = {
    "Strength": "strength_",
    "Betweenness": "betweenness_",
    "Clustering coefficient": "clustering_coeff_",
    "Participation coefficient": "participation_coeff_",
    "Degree": "degree_",
}
DYNAMIC_METRICS = ("Betweenness", "Participation coefficient")
GROUP_METRICS = ("Strength", "Betweenness", "Clustering coefficient", "Participation coefficient")
COLORS = {"celltype": "#0072B2", "gene": "#009E73", "receptor": "#D55E00"}


def roi_name(value: object) -> str:
    name = str(value).strip()
    return name.removeprefix("17Networks_")


def roi_column(frame: pd.DataFrame) -> str:
    for candidate in ROI_COLUMNS:
        if candidate in frame:
            return candidate
    raise ValueError(f"No explicit ROI column found; expected one of {ROI_COLUMNS}")


def roi_reference(path: Path) -> list[str]:
    frame = pd.read_csv(path)
    rois = frame[roi_column(frame)].map(roi_name).tolist()
    if len(rois) != N_ROI or len(set(rois)) != N_ROI:
        raise ValueError(f"ROI reference must contain {N_ROI} unique Schaefer-100 parcels")
    if not (all(r.startswith("LH_") for r in rois[:50]) and
            all(r.startswith("RH_") for r in rois[50:])):
        raise ValueError("ROI ordering must be 50 left-hemisphere then 50 right-hemisphere parcels")
    return rois


def modules_for(rois: list[str]) -> np.ndarray:
    return np.array([r.split("_")[1] if "_" in r else "Unknown" for r in rois])


def parse_edges(frame: pd.DataFrame, rois: list[str]) -> list[tuple[str, int, int]]:
    lookup = {roi: index for index, roi in enumerate(rois)}
    unique: dict[tuple[int, int], str] = {}
    for column in frame.columns:
        if "-" not in str(column):
            continue
        first, second = (roi_name(piece) for piece in str(column).split("-", 1))
        if first not in lookup or second not in lookup or first == second:
            continue
        i, j = sorted((lookup[first], lookup[second]))
        unique.setdefault((i, j), column)
    expected = N_ROI * (N_ROI - 1) // 2
    if len(unique) != expected:
        raise ValueError(f"Expected {expected} unique edges, found {len(unique)}. "
                         "Check atlas, ROI labels, and missing edge columns.")
    return [(column, i, j) for (i, j), column in unique.items()]


def metric_arrays(adjacency: np.ndarray, modules: np.ndarray) -> dict[str, np.ndarray]:
    """Match the notebook's weighted metric definitions.

    The supplied edge value is used as the NetworkX betweenness *distance*,
    while strength/participation/clustering treat it as a weight. These are
    descriptive properties of PET distributional-dissimilarity graphs.
    """
    import networkx as nx

    n = adjacency.shape[0]
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    rows, cols = np.triu_indices(n, 1)
    graph.add_weighted_edges_from(
        (int(i), int(j), float(adjacency[i, j]))
        for i, j in zip(rows, cols)
        if adjacency[i, j] != 0
    )
    strength = adjacency.sum(axis=1)
    participation = np.full(n, np.nan)
    for i in range(n):
        if strength[i] > 0:
            participation[i] = 1.0 - sum(
                (adjacency[i, modules == m].sum() / strength[i]) ** 2
                for m in np.unique(modules)
            )
    betweenness = nx.betweenness_centrality(graph, weight="weight", normalized=True)
    clustering = nx.clustering(graph, weight="weight")
    return {
        "Strength": strength,
        "Degree": np.count_nonzero(adjacency, axis=1),
        "Clustering coefficient": np.array([clustering[i] for i in range(n)]),
        "Betweenness": np.array([betweenness[i] for i in range(n)]),
        "Participation coefficient": participation,
    }


def prepare_metrics(edges_csv: Path, roi_csv: Path, output_csv: Path) -> pd.DataFrame:
    """Rebuild ROI-level metrics from the staging edge table.

    Invalid, incomplete, or out-of-range KLS rows are excluded and counted;
    no private row-level inputs are distributed with this project.
    """
    rois = roi_reference(roi_csv)
    modules = modules_for(rois)
    frame = pd.read_csv(edges_csv, low_memory=False)
    for column in ("estdtt0", "age", "sex"):
        if column not in frame:
            raise ValueError(f"Missing metadata column: {column}")
    edges = parse_edges(frame, rois)
    edge_cols = [col for col, _, _ in edges]
    values = frame[edge_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    valid_edges = np.isfinite(values).all(axis=1) & (values >= 0).all(axis=1) & (values <= 1).all(axis=1)
    valid_meta = frame[["estdtt0", "age", "sex"]].notna().all(axis=1).to_numpy()
    keep = valid_edges & valid_meta
    LOG.info("Valid edge/metadata rows: %d of %d (excluded %d)", keep.sum(), len(frame), (~keep).sum())
    metadata = [c for c in ("subject_id", "session_id", "amypad_id", "age", "sex",
                           "estdtt0", "pet_cl_refwc_gca_gaain", "cdr_glb_score") if c in frame]
    records = []
    for row_index in np.flatnonzero(keep):
        adjacency = np.zeros((N_ROI, N_ROI), dtype=float)
        for edge_index, (_, i, j) in enumerate(edges):
            adjacency[i, j] = adjacency[j, i] = values[row_index, edge_index]
        node_metrics = metric_arrays(adjacency, modules)
        record = {key: frame.iloc[row_index][key] for key in metadata}
        for metric, vector in node_metrics.items():
            record.update({f"{METRICS[metric]}17Networks_{roi}": float(value)
                           for roi, value in zip(rois, vector)})
        records.append(record)
        if len(records) % 50 == 0:
            LOG.info("Computed %d/%d valid rows", len(records), keep.sum())
    if not records:
        raise ValueError("No valid rows remained after edge and metadata quality checks")
    result = pd.DataFrame.from_records(records)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)
    return result


def fit_gams(metrics: pd.DataFrame, rois: list[str], output_dir: Path,
             spline_df: int = 10, spline_degree: int = 3) -> dict[str, Path]:
    """Age/sex-adjusted ROI-wise GAM with the notebook's spline settings."""
    from statsmodels.gam.api import BSplines, GLMGam

    frame = metrics.copy()
    frame["age"] = pd.to_numeric(frame["age"], errors="coerce")
    frame[TIME_COL] = pd.to_numeric(frame[TIME_COL], errors="coerce")
    frame = frame.dropna(subset=["age", "sex", TIME_COL])
    frame["sex"] = frame["sex"].astype("category")
    mode_sex = frame["sex"].mode().iat[0]
    median_age = frame["age"].median()
    outputs: dict[str, Path] = {}
    for metric, prefix in METRICS.items():
        columns = [f"{prefix}17Networks_{roi}" for roi in rois]
        missing = [col for col in columns if col not in frame]
        if missing:
            raise ValueError(f"{metric} is missing {len(missing)} ROI columns")
        predictions = None
        fitted = 0
        for col in columns:
            subset = frame[[col, "sex", "age", TIME_COL]].dropna().copy()
            if subset[TIME_COL].nunique() < max(5, spline_degree + 2):
                LOG.warning("Skipping %s: insufficient distinct SILA values", col)
                continue
            x_min, x_max = subset[TIME_COL].min(), subset[TIME_COL].max()
            grid = np.arange(np.ceil(x_min + 1e-6), np.floor(x_max - 1e-6) + 1)
            grid = grid[(grid >= -15) & (grid <= 15)].astype(float)
            if len(grid) == 0:
                LOG.warning("Skipping %s: no within-range integer SILA times", col)
                continue
            try:
                smoother = BSplines(subset[[TIME_COL]], df=[spline_df], degree=[spline_degree])
                model = GLMGam.from_formula(f'Q("{col}") ~ C(sex) + age', data=subset, smoother=smoother)
                fitted_model = model.fit()
                new_data = pd.DataFrame({"sex": [mode_sex] * len(grid),
                                         "age": [median_age] * len(grid), TIME_COL: grid})
                predicted = fitted_model.get_prediction(
                    new_data, exog_smooth=new_data[[TIME_COL]]).predicted_mean
            except Exception as exc:
                LOG.warning("Skipping GAM for %s: %s", col, exc)
                continue
            entry = pd.DataFrame({TIME_COL: grid, f"17Networks_{roi_name(col[len(prefix):])}": predicted})
            predictions = entry if predictions is None else predictions.merge(entry, on=TIME_COL, how="outer")
            fitted += 1
        if predictions is None:
            raise RuntimeError(f"No successful GAM fits for {metric}")
        if fitted != N_ROI:
            LOG.warning("%s: %d/%d GAMs fitted; spatial analysis requires complete ROI maps", metric, fitted, N_ROI)
        output_dir.mkdir(parents=True, exist_ok=True)
        target = output_dir / f"{metric.lower().replace(' ', '_')}.csv"
        predictions.sort_values(TIME_COL).to_csv(target, index=False)
        outputs[metric] = target
        LOG.info("%s GAM: %d/%d ROI models -> %s", metric, fitted, N_ROI, target)
    return outputs


def is_biological_feature(name: str) -> bool:
    bad = ("roi.label", "roi_label", "roi label", "label_17", "label_7", "label17",
           "label7", "network_17", "network_7", "yeo7", "yeo17", "parcel",
           "parcellation", "index", "unnamed")
    return not any(token in str(name).lower() for token in bad)


def unit_vector(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.shape != (N_ROI,) or not np.isfinite(x).all():
        raise ValueError("A complete finite 100-parcel vector is required")
    centered = x - x.mean()
    length = np.linalg.norm(centered)
    if length == 0:
        raise ValueError("A nonconstant 100-parcel vector is required")
    return centered / length


def unit_columns(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(matrix, dtype=float)
    valid = np.isfinite(matrix).all(axis=0)
    centered = matrix - np.nanmean(matrix, axis=0)
    length = np.linalg.norm(centered, axis=0)
    valid &= np.isfinite(length) & (length > 0)
    result = np.zeros_like(matrix)
    result[:, valid] = centered[:, valid] / length[valid]
    return result, valid


def load_biomaps(directory: Path, rois: list[str], output: Path) -> dict:
    biomaps = {}
    excluded = []
    for modality, filename in BIO_NAMES.items():
        frame = pd.read_csv(directory / filename)
        key = roi_column(frame)
        frame[key] = frame[key].map(roi_name)
        if frame[key].duplicated().any() or not set(rois).issubset(set(frame[key])):
            raise ValueError(f"Invalid or incomplete ROI labels in {filename}")
        frame = frame.set_index(key).reindex(rois)
        numeric = frame.apply(pd.to_numeric, errors="coerce")
        numeric = numeric.loc[:, numeric.notna().any(axis=0)]
        non_artifact = [c for c in numeric if is_biological_feature(c)]
        excluded.extend({"modality": modality, "feature": c, "reason": "atlas/index"}
                        for c in numeric if c not in non_artifact)
        numeric = numeric[non_artifact]
        centered, valid = unit_columns(numeric.to_numpy(dtype=float))
        excluded.extend({"modality": modality, "feature": c, "reason": "nonfinite/constant"}
                        for c, flag in zip(numeric.columns, valid) if not flag)
        biomaps[modality] = {"features": list(numeric.columns[valid]), "unit": centered[:, valid]}
        LOG.info("%s: retained %d valid biological features", modality, sum(valid))
    pd.DataFrame(excluded, columns=["modality", "feature", "reason"]).to_csv(
        output / "excluded_biological_features.csv", index=False)
    return biomaps


def generate_spins(output: Path, n_perm: int, seed: int) -> np.ndarray:
    """Vasa fsLR-32k parcellated spins; reuse cached permutations if present."""
    cache = output / "spin_cache"
    cache.mkdir(parents=True, exist_ok=True)
    cached = cache / f"vasa_schaefer100_{n_perm}_seed{seed}.npy"
    if cached.exists():
        spins = np.load(cached, allow_pickle=False)
    else:
        import nibabel as nib
        from brainspace.datasets import load_parcellation
        from neuromaps import nulls

        left, right = load_parcellation("schaefer", scale=N_ROI, join=False)
        paths = []
        for hemisphere, labels in zip(("L", "R"), (left, right)):
            dest = cache / f"schaefer100_{hemisphere}.label.gii"
            gii = nib.gifti.GiftiImage(darrays=[nib.gifti.GiftiDataArray(
                data=np.asarray(labels, dtype=np.int32), intent="NIFTI_INTENT_LABEL",
                datatype="NIFTI_TYPE_INT32")])
            nib.save(gii, str(dest))
            paths.append(str(dest))
        spins = nulls.vasa(data=None, atlas="fsLR", density="32k",
                           parcellation=tuple(paths), n_perm=n_perm, seed=seed)
    spins = np.asarray(spins)
    if spins.shape == (n_perm, N_ROI):
        spins = spins.T
    if spins.shape != (N_ROI, n_perm) or not np.issubdtype(spins.dtype, np.integer):
        raise ValueError("Spatial permutations have the wrong shape or index dtype")
    if spins.min() != 0 or spins.max() != N_ROI - 1:
        raise ValueError("Spatial permutations contain out-of-range indices")
    for j in range(n_perm):
        if not np.array_equal(np.sort(spins[:, j]), np.arange(N_ROI)):
            raise ValueError(f"Spin {j} is not a permutation of the 100 parcels")
    if not cached.exists():
        np.save(cached, spins)
    LOG.info("Loaded %d cached/generated Vasa spins", n_perm)
    return spins


def cdr_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype("string").str.strip().replace({
        "0 - Normal": "0", "0.5 - Very mild": "0.5", "0,5 - Very mild": "0.5",
        "0,5": "0.5"}), errors="coerce")


def ordered_metric_columns(frame: pd.DataFrame, metric: str, rois: list[str]) -> list[str]:
    prefix = METRICS[metric]
    lookup = {roi_name(col[len(prefix):]): col for col in frame if col.startswith(prefix)}
    missing = [roi for roi in rois if roi not in lookup]
    if missing:
        raise ValueError(f"{metric}: missing {len(missing)} regional columns, e.g. {missing[:3]}")
    return [lookup[roi] for roi in rois]


def group_vectors(metrics: pd.DataFrame, rois: list[str], threshold: float) -> tuple[dict, dict]:
    cdr = cdr_numeric(metrics["cdr_glb_score"])
    cl = pd.to_numeric(metrics["pet_cl_refwc_gca_gaain"], errors="coerce")
    masks = {"Healthy": cdr.eq(0) & cl.lt(threshold),
             "EarlyAD": cdr.eq(0.5) & cl.ge(threshold)}
    counts = {name: int(mask.sum()) for name, mask in masks.items()}
    if not all(counts.values()):
        raise ValueError(f"An analysis group is empty: {counts}")
    vectors = {}
    for metric in GROUP_METRICS:
        subset = metrics[ordered_metric_columns(metrics, metric, rois)].apply(pd.to_numeric, errors="coerce")
        vectors[metric] = {group: subset.loc[mask].mean(axis=0).to_numpy(dtype=float)
                           for group, mask in masks.items()}
    LOG.info("Group observations (not necessarily unique subjects): %s", counts)
    return vectors, counts


def bh_fdr(table: pd.DataFrame, p_col: str, q_col: str) -> None:
    from statsmodels.stats.multitest import multipletests

    table[q_col] = np.nan
    valid = pd.to_numeric(table[p_col], errors="coerce").notna()
    if valid.any():
        table.loc[valid, q_col] = multipletests(table.loc[valid, p_col], method="fdr_bh")[1]


def empirical_p(observed: np.ndarray, null: np.ndarray) -> np.ndarray:
    return (1.0 + (np.abs(null) >= np.abs(observed)[None, :]).sum(axis=0)) / (len(null) + 1)


def group_spatial_test(vectors: dict, biomaps: dict, spins: np.ndarray,
                       out: Path, alpha: float, chunk_size: int) -> pd.DataFrame:
    records = []
    for metric, groups in vectors.items():
        healthy, early = (unit_vector(groups[k]) for k in ("Healthy", "EarlyAD"))
        spun_h, spun_e = healthy[spins], early[spins]
        for modality, bio in biomaps.items():
            values = bio["unit"]
            observed_h, observed_e = healthy @ values, early @ values
            observed_delta = observed_e - observed_h
            p_h = np.empty(values.shape[1]); p_e = np.empty_like(p_h); p_delta = np.empty_like(p_h)
            for start in range(0, values.shape[1], chunk_size):
                end = min(start + chunk_size, values.shape[1]); block = values[:, start:end]
                nh, ne = spun_h.T @ block, spun_e.T @ block
                p_h[start:end] = empirical_p(observed_h[start:end], nh)
                p_e[start:end] = empirical_p(observed_e[start:end], ne)
                p_delta[start:end] = empirical_p(observed_delta[start:end], ne - nh)
            for j, feature in enumerate(bio["features"]):
                records.append({"graph_property": metric, "modality": modality, "feature": feature,
                                "r_Healthy": observed_h[j], "r_EarlyAD": observed_e[j],
                                "delta_r_EarlyAD_minus_Healthy": observed_delta[j],
                                "p_spin_Healthy": p_h[j], "p_spin_EarlyAD": p_e[j],
                                "p_spin_delta": p_delta[j], "n_permutations": spins.shape[1],
                                "n_parcels": N_ROI})
    table = pd.DataFrame(records)
    if table.empty:
        raise ValueError("No group biological features could be tested")
    for kind in ("Healthy", "EarlyAD", "delta"):
        bh_fdr(table, f"p_spin_{kind}", f"q_spin_{kind}")
        table[f"robust_{kind}"] = table[f"q_spin_{kind}"] < alpha
    table["robust_any_group_test"] = table[[f"robust_{k}" for k in ("Healthy", "EarlyAD", "delta")]].any(axis=1)
    table["mean_abs_group_r"] = table[["r_Healthy", "r_EarlyAD"]].abs().mean(axis=1)
    table.to_csv(out / "group_spatial_null_all.csv", index=False)
    table.loc[table["robust_any_group_test"]].to_csv(out / "group_spatial_null_robust.csv", index=False)
    return table


def read_prediction(path: Path, rois: list[str]) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(path)
    if TIME_COL not in frame:
        raise ValueError(f"Missing {TIME_COL} in {path.name}")
    frame[TIME_COL] = pd.to_numeric(frame[TIME_COL], errors="coerce")
    frame = frame.loc[frame[TIME_COL].between(-15, 15)].sort_values(TIME_COL)
    lookup = {roi_name(c): c for c in frame if c != TIME_COL}
    if any(r not in lookup for r in rois):
        raise ValueError(f"Prediction file {path.name} does not contain every Schaefer-100 ROI")
    matrix = frame[[lookup[r] for r in rois]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    centered = matrix - matrix.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1)
    valid = np.isfinite(centered).all(axis=1) & np.isfinite(norms) & (norms > 0)
    if not valid.any():
        raise ValueError(f"No valid predicted timepoints in {path.name}")
    return frame[TIME_COL].to_numpy()[valid], centered[valid] / norms[valid, None]


def dynamic_spatial_test(prediction_paths: dict[str, Path], rois: list[str],
                         biomaps: dict, spins: np.ndarray, out: Path,
                         alpha: float, chunk_size: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = []; time_records = []
    for metric in DYNAMIC_METRICS:
        times, unit = read_prediction(prediction_paths[metric], rois)
        if len(times) < 2:
            raise ValueError(f"{metric}: fewer than two valid SILA timepoints")
        spun_by_time = [row[spins] for row in unit]
        for modality, bio in biomaps.items():
            observed = unit @ bio["unit"]
            for start in range(0, len(bio["features"]), chunk_size):
                end = min(start + chunk_size, len(bio["features"]))
                block = bio["unit"][:, start:end]
                obs = observed[:, start:end]
                max_null = np.zeros((spins.shape[1], block.shape[1]), dtype=np.float32)
                for spun in spun_by_time:
                    np.maximum(max_null, np.abs(spun.T @ block), out=max_null)
                peak = np.argmax(np.abs(obs), axis=0)
                max_observed = np.max(np.abs(obs), axis=0)
                feature_p = (1 + (max_null >= max_observed[None, :]).sum(axis=0)) / (len(max_null) + 1)
                for j, feature in enumerate(bio["features"][start:end]):
                    features.append({"graph_property": metric, "modality": modality, "feature": feature,
                                     "max_abs_r": max_observed[j], "peak_r": obs[peak[j], j],
                                     "peak_estdtt0": times[peak[j]], "mean_abs_r": np.abs(obs[:, j]).mean(),
                                     "p_spin_max_time": feature_p[j], "n_timepoints": len(times),
                                     "n_permutations": spins.shape[1], "n_parcels": N_ROI})
                    for t, tval in enumerate(times):
                        p = (1 + (max_null[:, j] >= abs(obs[t, j])).sum()) / (len(max_null) + 1)
                        time_records.append({"graph_property": metric, "modality": modality,
                                             "feature": feature, TIME_COL: tval,
                                             "pearson_r": obs[t, j], "p_spin_max_time": p})
    feature_df, time_df = pd.DataFrame(features), pd.DataFrame(time_records)
    if feature_df.empty:
        raise ValueError("No dynamic biological features could be tested")
    bh_fdr(feature_df, "p_spin_max_time", "q_spin_feature")
    feature_df["robust_dynamic_feature"] = feature_df["q_spin_feature"] < alpha
    lookup = feature_df.set_index(["graph_property", "modality", "feature"])
    idx = pd.MultiIndex.from_frame(time_df[["graph_property", "modality", "feature"]])
    time_df["q_spin_feature"] = lookup["q_spin_feature"].reindex(idx).to_numpy()
    time_df["robust_dynamic_feature"] = lookup["robust_dynamic_feature"].reindex(idx).to_numpy()
    time_df["robust_timepoint"] = time_df["robust_dynamic_feature"] & (time_df["p_spin_max_time"] < alpha)
    feature_df.to_csv(out / "dynamic_spatial_null_all.csv", index=False)
    feature_df.loc[feature_df["robust_dynamic_feature"]].to_csv(out / "dynamic_spatial_null_robust.csv", index=False)
    time_df.to_csv(out / "dynamic_timepoint_null_all.csv", index=False)
    return feature_df, time_df


def select_display(table: pd.DataFrame, robustness_col: str, rank_col: str,
                   top_genes: int, top_receptors: int) -> pd.DataFrame:
    selection = []
    for (_, modality), subset in table.groupby(["graph_property", "modality"]):
        robust = subset.loc[subset[robustness_col]].copy()
        if modality == "gene":
            robust = robust.nlargest(top_genes, rank_col)
        elif modality == "receptor":
            robust = robust.nlargest(top_receptors, rank_col)
        selection.append(robust)
    return pd.concat(selection, ignore_index=True) if selection else table.iloc[:0].copy()


def plot_group(table: pd.DataFrame, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for metric, data in table.groupby("graph_property"):
        data = data.sort_values(["modality", "mean_abs_group_r"], ascending=[True, False])
        x = np.arange(len(data)); width = 0.36
        colors = [COLORS[m] for m in data["modality"]]
        fig, axes = plt.subplots(2, 1, figsize=(max(10, len(x) * 0.42), 7), sharex=True)
        axes[0].bar(x - width / 2, data["r_Healthy"].abs(), width, alpha=.45,
                    hatch="//", edgecolor="black", color=colors, label="Healthy")
        axes[0].bar(x + width / 2, data["r_EarlyAD"].abs(), width, alpha=.9,
                    edgecolor="black", color=colors, label="EarlyAD")
        axes[0].set_ylabel("Absolute spatial Pearson |r|")
        axes[0].legend()
        axes[1].bar(x, data["delta_r_EarlyAD_minus_Healthy"], color=colors, edgecolor="black")
        axes[1].axhline(0, color="black", linewidth=.8)
        axes[1].set_ylabel("EarlyAD − Healthy (Δr)")
        axes[1].set_xticks(x, labels=data["feature"], rotation=90, fontsize=8)
        fig.suptitle(f"Spatial-null-corrected group correspondence: {metric}")
        fig.tight_layout()
        fig.savefig(output / f"group_{metric.lower().replace(' ', '_')}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_dynamic(display: pd.DataFrame, time_df: pd.DataFrame, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for metric, data in display.groupby("graph_property"):
        modalities = [m for m in BIO_NAMES if m in set(data["modality"])]
        fig, axes = plt.subplots(1, len(modalities), figsize=(6 * len(modalities), 5),
                                 sharex=True, sharey=True, squeeze=False)
        for ax, modality in zip(axes[0], modalities):
            for i, feature in enumerate(data.loc[data["modality"] == modality, "feature"]):
                curve = time_df.loc[(time_df["graph_property"] == metric) &
                                    (time_df["modality"] == modality) &
                                    (time_df["feature"] == feature)].sort_values(TIME_COL)
                style = ("-", "--", "-.", ":")[i % 4]
                ax.plot(curve[TIME_COL], curve["pearson_r"], linestyle=style, label=feature)
                significant = curve.loc[curve["robust_timepoint"]]
                ax.scatter(significant[TIME_COL], significant["pearson_r"], s=18,
                           facecolors="none", edgecolors="black")
            ax.axhline(0, color="black", linestyle="--", linewidth=.7)
            ax.axvline(0, color="black", linestyle=":", linewidth=.7)
            ax.set_xlim(-15, 15); ax.set_title(modality); ax.set_xlabel("SILA years")
            ax.legend(fontsize=7, frameon=False)
        axes[0, 0].set_ylabel("Spatial Pearson r")
        fig.suptitle(f"Max-time spatial-null biological correspondence: {metric}")
        fig.tight_layout()
        fig.savefig(output / f"dynamic_{metric.lower().replace(' ', '_')}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def analyze(metrics_csv: Path, roi_csv: Path, brainmap_dir: Path,
            prediction_paths: dict[str, Path], output: Path, n_perm: int, seed: int,
            alpha: float, threshold: float, chunk_size: int,
            top_genes: int, top_receptors: int, figures: bool) -> None:
    output.mkdir(parents=True, exist_ok=True)
    rois = roi_reference(roi_csv)
    biological = load_biomaps(brainmap_dir, rois, output)
    spins = generate_spins(output, n_perm, seed)
    vectors, counts = group_vectors(pd.read_csv(metrics_csv, low_memory=False), rois, threshold)
    pd.DataFrame([{"group": key, "n_observations": value} for key, value in counts.items()]).to_csv(
        output / "group_sample_counts.csv", index=False)
    group = group_spatial_test(vectors, biological, spins, output, alpha, chunk_size)
    dynamic, times = dynamic_spatial_test(prediction_paths, rois, biological, spins,
                                          output, alpha, chunk_size)
    group_display = select_display(group, "robust_any_group_test", "mean_abs_group_r",
                                   top_genes, top_receptors)
    dynamic_display = select_display(dynamic, "robust_dynamic_feature", "mean_abs_r",
                                     top_genes, top_receptors)
    group_display.to_csv(output / "group_display_post_inference.csv", index=False)
    dynamic_display.to_csv(output / "dynamic_display_post_inference.csv", index=False)
    group.groupby(["graph_property", "modality"])[["robust_Healthy", "robust_EarlyAD", "robust_delta"]].sum().to_csv(
        output / "group_summary.csv")
    dynamic.groupby(["graph_property", "modality"])["robust_dynamic_feature"].agg(["size", "sum"]).to_csv(
        output / "dynamic_summary.csv")
    if figures:
        plot_group(group_display, output)
        plot_dynamic(dynamic_display, times, output)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("stage", choices=("metrics", "gam", "analyze", "all"))
    parser.add_argument("--edges", type=Path, help="Staging table with Schaefer-100 edge columns")
    parser.add_argument("--metrics-csv", type=Path, help="Existing node-metrics CSV (if not recomputing)")
    parser.add_argument("--brainmap-dir", type=Path, required=True,
                        help="Directory holding three Schaefer-100 biological-map CSV files")
    parser.add_argument("--roi-reference", type=Path,
                        help="Optional canonical ROI CSV; defaults to receptor map in brainmap directory")
    parser.add_argument("--out", type=Path, required=True, help="Output directory")
    parser.add_argument("--betweenness-prediction", type=Path, help="Precomputed GAM prediction CSV")
    parser.add_argument("--participation-prediction", type=Path, help="Precomputed GAM prediction CSV")
    parser.add_argument("--n-perm", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--alpha", type=float, default=.05)
    parser.add_argument("--centiloid-threshold", type=float, default=24.0)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--spline-df", type=int, default=10)
    parser.add_argument("--top-genes", type=int, default=10)
    parser.add_argument("--top-receptors", type=int, default=10)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.n_perm < 1 or args.chunk_size < 1 or not (0 < args.alpha < 1):
        parser.error("Require --n-perm >= 1, --chunk-size >= 1, and 0 < --alpha < 1")
    roi_file = args.roi_reference or args.brainmap_dir / BIO_NAMES["receptor"]
    metrics_path = args.metrics_csv or args.out / "node_metrics.csv"
    if args.stage in ("metrics", "all"):
        if args.edges:
            prepare_metrics(args.edges, roi_file, metrics_path)
        elif args.metrics_csv is None:
            parser.error("--edges is required unless --metrics-csv supplies precomputed metrics")
    if args.stage in ("gam", "all"):
        if not metrics_path.exists():
            parser.error(f"Missing node metrics file: {metrics_path}")
        fit_gams(pd.read_csv(metrics_path, low_memory=False), roi_reference(roi_file),
                 args.out / "predictions", spline_df=args.spline_df)
    if args.stage in ("analyze", "all"):
        if not metrics_path.exists():
            parser.error(f"Missing node metrics file: {metrics_path}")
        prediction_paths = {
            "Betweenness": args.betweenness_prediction or args.out / "predictions" / "betweenness.csv",
            "Participation coefficient": args.participation_prediction or
                                         args.out / "predictions" / "participation_coefficient.csv",
        }
        for metric, path in prediction_paths.items():
            if not path.exists():
                parser.error(f"Missing {metric} prediction CSV: {path}")
        analyze(metrics_path, roi_file, args.brainmap_dir, prediction_paths,
                args.out / "spatial_nulls", args.n_perm, args.seed, args.alpha,
                args.centiloid_threshold, args.chunk_size, args.top_genes,
                args.top_receptors, not args.no_figures)


if __name__ == "__main__":
    main()
