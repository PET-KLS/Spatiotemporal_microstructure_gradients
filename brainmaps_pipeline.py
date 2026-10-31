#!/usr/bin/env python3
"""PET-KLS regional topology associations with normative biological brain maps."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import pearsonr


LOG = logging.getLogger("petkls.brainmaps")
METRICS = {
    "Participation coefficient": "participation_coeff_17Networks_",
    "Betweenness": "betweenness_17Networks_",
    "Clustering coefficient": "clustering_coeff_17Networks_",
    "Strength": "strength_17Networks_",
}
ROI_COLUMNS = ("ROI.Name_17_mod", "ROI.Name_17", "roi", "ROI", "roi_name", "ROI.Name")


def normalize_roi(name: str) -> str:
    roi = str(name).strip()
    return roi.replace("17Networks_", "").replace("7Networks_", "")


def resolve_path(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else base / path


def read_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    if not isinstance(cfg, dict):
        raise ValueError("Config must contain a YAML mapping.")
    for key in ("staging_metrics", "group_metrics", "brainmaps", "output_dir"):
        if key not in cfg:
            raise KeyError(f"Missing config entry: {key}")
    base = path.resolve().parent
    cfg["staging_metrics"] = resolve_path(cfg["staging_metrics"], base)
    cfg["group_metrics"] = resolve_path(cfg["group_metrics"], base)
    cfg["output_dir"] = resolve_path(cfg["output_dir"], base)
    cfg["brainmaps"] = {
        modality: resolve_path(value, base)
        for modality, value in cfg["brainmaps"].items()
        if value
    }
    if not {"celltype", "gene", "receptor"} <= set(cfg["brainmaps"]):
        raise ValueError("Cell-type, gene, and receptor brain maps are required.")
    cfg.setdefault("time_min", -15)
    cfg.setdefault("time_max", 15)
    cfg.setdefault("spline_df", 10)
    cfg.setdefault("spline_degree", 3)
    cfg.setdefault("centiloid_threshold", 24)
    if cfg["time_max"] <= cfg["time_min"]:
        raise ValueError("time_max must be greater than time_min.")
    return cfg


def load_metrics(path: Path, *, require_time: bool) -> pd.DataFrame:
    data = pd.read_csv(path, low_memory=False)
    data.columns = data.columns.str.strip()
    required = {"age", "sex"} if require_time else set()
    if require_time:
        required.add("estdtt0")
    else:
        required.update(("cdr_glb_score", "pet_cl_refwc_gca_gaain"))
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"{path}: missing metadata {sorted(missing)}")
    for metric, prefix in METRICS.items():
        roi_cols = [c for c in data if c.startswith(prefix)]
        rois = [normalize_roi(c[len(prefix):]) for c in roi_cols]
        if len(roi_cols) != 100 or len(set(rois)) != 100:
            raise ValueError(f"{path}: {metric} requires 100 unique Schaefer-100 parcels; found {len(roi_cols)}")
    if "subject_id" in data:
        ids = data["subject_id"].dropna()
        if ids.duplicated().any():
            raise ValueError(f"{path}: multiple rows per participant; use one session per subject.")
    return data


def read_brainmap(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    roi_col = next((c for c in ROI_COLUMNS if c in raw), None)
    if roi_col is None:
        raise ValueError(f"{path}: missing ROI column (expected one of {ROI_COLUMNS})")
    raw["__roi"] = raw[roi_col].map(normalize_roi)
    if raw["__roi"].duplicated().any():
        raise ValueError(f"{path}: duplicate ROI names after normalization.")
    drop = {roi_col, "__roi"}
    features = {}
    for col in raw.columns:
        if col in drop:
            continue
        lower = col.lower()
        if any(marker in lower for marker in ("roi.label", "roi_label", "label_7", "label_17", "unnamed")):
            continue
        numeric = pd.to_numeric(raw[col], errors="coerce")
        if numeric.notna().sum() >= 3:
            features[col] = numeric.to_numpy(dtype=float)
    if not features:
        raise ValueError(f"{path}: no numeric biological features.")
    result = pd.DataFrame(features, index=raw["__roi"].tolist())
    if len(result) != 100:
        raise ValueError(f"{path}: expected 100 Schaefer parcels; found {len(result)}")
    return result


def load_maps(config: dict) -> dict[str, pd.DataFrame]:
    return {mod: read_brainmap(path) for mod, path in config["brainmaps"].items()}


def metric_matrix(data: pd.DataFrame, prefix: str) -> pd.DataFrame:
    cols = [col for col in data if col.startswith(prefix)]
    matrix = data[cols].apply(pd.to_numeric, errors="coerce")
    matrix.columns = [normalize_roi(col[len(prefix):]) for col in cols]
    return matrix


def pearson_safe(x: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    n = int(keep.sum())
    if n < 3 or np.ptp(x[keep]) < 1e-12 or np.ptp(y[keep]) < 1e-12:
        return float("nan"), n
    return float(pearsonr(x[keep], y[keep]).statistic), n


def fit_roi_gams(data: pd.DataFrame, config: dict) -> pd.DataFrame:
    # Keep the GAM dependency isolated from input validation and correlations.
    import statsmodels.api as sm
    from statsmodels.gam.api import BSplines, GLMGam

    base = data.copy()
    base["estdtt0"] = pd.to_numeric(base["estdtt0"], errors="coerce")
    base["age"] = pd.to_numeric(base["age"], errors="coerce")
    base = base.dropna(subset=["estdtt0", "age", "sex"])
    if len(base) < 40:
        raise ValueError("At least 40 participants with time, age and sex are required for GAM fitting.")
    sex = base["sex"].astype(str)
    dummies = pd.get_dummies(sex, prefix="sex", drop_first=True, dtype=float)
    reference_age = float(base["age"].median())
    reference_sex = sex.mode().iat[0]
    grid = np.arange(float(config["time_min"]), float(config["time_max"]) + 1e-9, 1.0)
    results = []

    for metric, prefix in METRICS.items():
        roi_matrix = metric_matrix(base, prefix)
        for roi in roi_matrix:
            y = roi_matrix[roi]
            valid = y.notna()
            sub = base.loc[valid]
            if len(sub) < 40 or sub["estdtt0"].nunique() < config["spline_df"] + 1:
                raise ValueError(f"Insufficient usable observations for {metric} / {roi}.")
            time = sub[["estdtt0"]].reset_index(drop=True)
            smooth = BSplines(time, df=[config["spline_df"]], degree=[config["spline_degree"]])
            design = pd.DataFrame({"const": np.ones(len(sub)), "age": sub["age"].to_numpy()})
            sex_part = dummies.loc[valid].reset_index(drop=True)
            design = pd.concat([design, sex_part], axis=1)
            model = GLMGam(
                endog=y.loc[valid].to_numpy(dtype=float),
                exog=design,
                smoother=smooth,
                family=sm.families.Gaussian(),
            )
            fit = model.fit()
            within = grid[(grid >= time["estdtt0"].min()) & (grid <= time["estdtt0"].max())]
            if len(within) < 2:
                raise ValueError(f"No prediction support in requested window for {metric} / {roi}.")
            new_design = pd.DataFrame({"const": np.ones(len(within)), "age": reference_age})
            for dummy in dummies:
                category = dummy.removeprefix("sex_")
                new_design[dummy] = float(category == reference_sex)
            pred = fit.predict(
                exog=new_design,
                exog_smooth=pd.DataFrame({"estdtt0": within}),
            )
            results.extend(
                (metric, roi, float(t), float(value))
                for t, value in zip(within, pred, strict=True)
            )
        LOG.info("Fitted %s: 100 parcels", metric)
    return pd.DataFrame(results, columns=["graph_property", "ROI", "estdtt0", "predicted_value"])


def correlate_trajectories(predictions: pd.DataFrame, brainmaps: dict[str, pd.DataFrame]) -> pd.DataFrame:
    records = []
    for (metric, time), group in predictions.groupby(["graph_property", "estdtt0"], sort=True):
        topo = group.set_index("ROI")["predicted_value"]
        for modality, maps in brainmaps.items():
            common = topo.index.intersection(maps.index)
            if len(common) != 100:
                raise ValueError(f"{metric}/{modality}: {len(common)} parcels matched; expected 100.")
            x = topo.loc[common].to_numpy(dtype=float)
            for feature in maps:
                r, n = pearson_safe(x, maps.loc[common, feature].to_numpy())
                records.append((metric, modality, feature, time, r, n))
    return pd.DataFrame(records, columns=["graph_property", "modality", "feature", "estdtt0", "pearson_r", "n_parcels"])


def parse_cdr(values: pd.Series) -> pd.Series:
    cleaned = values.astype(str).str.extract(r"^\s*(\d+(?:\.\d+)?)", expand=False)
    return pd.to_numeric(cleaned, errors="coerce")


def group_masks(data: pd.DataFrame, threshold: float) -> dict[str, pd.Series]:
    cdr = parse_cdr(data["cdr_glb_score"])
    centiloid = pd.to_numeric(data["pet_cl_refwc_gca_gaain"], errors="coerce")
    return {
        "Healthy": (cdr == 0) & (centiloid < threshold),
        "EarlyAD": (cdr == 0.5) & (centiloid >= threshold),
    }


def correlate_groups(data: pd.DataFrame, brainmaps: dict[str, pd.DataFrame], cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    masks = group_masks(data, cfg["centiloid_threshold"])
    counts = {
        label: int(data.loc[mask, "subject_id"].nunique() if "subject_id" in data else mask.sum())
        for label, mask in masks.items()
    }
    if any(n < 3 for n in counts.values()):
        raise ValueError(f"Groups are empty or too small: {counts}")
    expected = cfg.get("expected_group_counts")
    if expected and any(counts.get(name) != target for name, target in expected.items()):
        raise ValueError(f"Actual group counts {counts} do not match expected {expected}.")
    records = []
    for metric, prefix in METRICS.items():
        matrix = metric_matrix(data, prefix)
        vectors = {name: matrix.loc[mask].mean(axis=0) for name, mask in masks.items()}
        for modality, maps in brainmaps.items():
            common = vectors["Healthy"].index.intersection(maps.index)
            if len(common) != 100:
                raise ValueError(f"{metric}/{modality}: {len(common)} parcels matched; expected 100.")
            for feature in maps:
                r_h, n_h = pearson_safe(vectors["Healthy"].loc[common].to_numpy(), maps.loc[common, feature].to_numpy())
                r_e, n_e = pearson_safe(vectors["EarlyAD"].loc[common].to_numpy(), maps.loc[common, feature].to_numpy())
                records.append((metric, modality, feature, r_h, r_e, r_e - r_h, n_h, n_e))
    result = pd.DataFrame(records, columns=[
        "graph_property", "modality", "feature", "r_Healthy", "r_EarlyAD",
        "delta_r_EarlyAD_minus_Healthy", "n_parcels_Healthy", "n_parcels_EarlyAD",
    ])
    count_df = pd.DataFrame([{"group": name, "n_participants": count} for name, count in counts.items()])
    return result, count_df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="YAML file with input and output locations")
    parser.add_argument("--step", choices=("all", "temporal", "group"), default="all")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    cfg = read_config(args.config)
    outdir = cfg["output_dir"]
    outdir.mkdir(parents=True, exist_ok=True)
    maps = load_maps(cfg)

    if args.step in ("all", "temporal"):
        staging = load_metrics(cfg["staging_metrics"], require_time=True)
        pred = fit_roi_gams(staging, cfg)
        pred.to_csv(outdir / "regional_gam_predictions.csv", index=False)
        temporal = correlate_trajectories(pred, maps)
        temporal.to_csv(outdir / "temporal_spatial_correlations.csv", index=False)
        summary = temporal.assign(abs_r=temporal["pearson_r"].abs()).groupby(
            ["graph_property", "modality", "feature"], as_index=False
        ).agg(mean_abs_r=("abs_r", "mean"), max_abs_r=("abs_r", "max"), n_timepoints=("estdtt0", "nunique"))
        summary.to_csv(outdir / "temporal_feature_summary.csv", index=False)
        LOG.info("Temporal associations written to %s", outdir)

    if args.step in ("all", "group"):
        groups = load_metrics(cfg["group_metrics"], require_time=False)
        corr, counts = correlate_groups(groups, maps, cfg)
        corr.to_csv(outdir / "group_spatial_correlations.csv", index=False)
        counts.to_csv(outdir / "group_sample_sizes.csv", index=False)
        LOG.info("Group associations written to %s", outdir)


if __name__ == "__main__":
    main()
