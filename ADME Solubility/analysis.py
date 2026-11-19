from __future__ import annotations
import json
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split, KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "delaney-processed.csv"
OUT = ROOT / "results"
TARGET = "measured log solubility in mols per litre"
FEATURES = ["Minimum Degree", "Molecular Weight", "Number of H-Bond Donors", "Number of Rings", "Number of Rotatable Bonds", "Polar Surface Area"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(DATA)
    # Keep measured response only; exclude the dataset's existing ESOL prediction to prevent target leakage.
    df = raw[["Compound ID", "smiles", TARGET] + FEATURES].copy()
    rows_loaded = len(df)
    for col in [TARGET] + FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.drop_duplicates(subset=["Compound ID", "smiles", TARGET])
    rows_after_dedup = len(df)
    df = df.dropna(subset=[TARGET] + FEATURES)
    df = df[(df["Molecular Weight"] > 0) & (df["Polar Surface Area"] >= 0)]
    rows_used = len(df)
    df.to_csv(OUT / "cleaned_dataset.csv", index=False)

    # EDA summaries and plots.
    summary = {
        "dataset": "DeepChem MoleculeNet Delaney (ESOL)",
        "rows_downloaded": rows_loaded,
        "rows_after_exact_deduplication": rows_after_dedup,
        "rows_used": rows_used,
        "rows_removed": rows_loaded - rows_used,
        "features": FEATURES,
        "target": TARGET,
        "target_mean": float(df[TARGET].mean()),
        "target_median": float(df[TARGET].median()),
        "target_std": float(df[TARGET].std()),
        "target_min": float(df[TARGET].min()),
        "target_max": float(df[TARGET].max()),
        "feature_correlations_with_target": df[FEATURES + [TARGET]].corr(numeric_only=True)[TARGET].drop(TARGET).sort_values().to_dict(),
    }
    (OUT / "eda_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    axes[0].hist(df[TARGET], bins=30, color="#2a788e", edgecolor="white")
    axes[0].set(title="Measured aqueous solubility", xlabel="log solubility (mol/L)", ylabel="Compounds")
    corr = df[FEATURES + [TARGET]].corr(numeric_only=True)[TARGET].drop(TARGET).sort_values()
    axes[1].barh(corr.index, corr.values, color="#5ec962")
    axes[1].set(title="Descriptor correlation with target", xlabel="Pearson correlation")
    fig.tight_layout(); fig.savefig(OUT / "eda.png", dpi=170); plt.close(fig)

    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index, test_size=0.20, random_state=42
    )
    candidates = {
        "Mean baseline": DummyRegressor(strategy="mean"),
        "Ridge regression": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    }
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    search = RandomizedSearchCV(
        rf,
        param_distributions={
            "n_estimators": [200, 400, 600],
            "max_depth": [None, 8, 12, 18],
            "min_samples_leaf": [1, 2, 4],
            "max_features": [0.6, 0.8, 1.0],
        },
        n_iter=20, cv=KFold(n_splits=5, shuffle=True, random_state=42),
        scoring="neg_mean_absolute_error", random_state=42, n_jobs=-1,
    )
    search.fit(X_train, y_train)
    candidates["Tuned random forest"] = search.best_estimator_
    metrics = []
    fitted = {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        fitted[name] = (model, pred)
        metrics.append({
            "model": name,
            "test_MAE": float(mean_absolute_error(y_test, pred)),
            "test_RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
            "test_R2": float(r2_score(y_test, pred)),
        })
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(OUT / "model_metrics.csv", index=False)
    best = candidates["Tuned random forest"]
    (OUT / "best_parameters.json").write_text(json.dumps(search.best_params_, indent=2), encoding="utf-8")
    joblib.dump(best, OUT / "solubility_random_forest.joblib")
    best_pred = best.predict(X_test)
    pred_df = pd.DataFrame({
        "compound": df.loc[idx_test, "Compound ID"].to_numpy(),
        "measured_logS": y_test.to_numpy(),
        "predicted_logS": best_pred,
        "residual": y_test.to_numpy() - best_pred,
    })
    pred_df.to_csv(OUT / "test_predictions.csv", index=False)
    importances = pd.DataFrame({"feature": FEATURES, "importance": best.feature_importances_}).sort_values("importance", ascending=False)
    importances.to_csv(OUT / "feature_importance.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].scatter(y_test, best_pred, alpha=.65, s=22, color="#31688e")
    lo, hi = float(min(y_test.min(), best_pred.min())), float(max(y_test.max(), best_pred.max()))
    axes[0].plot([lo, hi], [lo, hi], "--", color="#d1495b")
    axes[0].set(title="Holdout: measured vs predicted", xlabel="Measured logS", ylabel="Predicted logS")
    axes[1].barh(importances.feature[::-1], importances.importance[::-1], color="#35b779")
    axes[1].set(title="Random forest feature importance", xlabel="Importance")
    fig.tight_layout(); fig.savefig(OUT / "model_diagnostics.png", dpi=170); plt.close(fig)

    report = {
        "data_cleaning": {"rows_loaded": rows_loaded, "rows_after_exact_deduplication": rows_after_dedup, "rows_used": rows_used, "rows_removed": rows_loaded-rows_used},
        "holdout_fraction": 0.20,
        "random_state": 42,
        "split_note": "Random row split, not a chemical scaffold split; performance may be optimistic for novel chemical scaffolds.",
        "best_random_forest_parameters": search.best_params_,
        "random_forest_tuning_cv_MAE": float(-search.best_score_),
        "test_metrics": metrics,
        "target": "Measured log10 aqueous solubility in mol/L; lower values indicate lower aqueous solubility.",
    }
    (OUT / "case_study_results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
