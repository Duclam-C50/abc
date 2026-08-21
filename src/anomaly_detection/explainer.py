"""
explainer.py
============
Integrated Gradients (IG) explanation pipeline and dashboard data generator.

Public API
----------
- :func:`explain`            - compute and save IG attributions for top-N anomalies.
- :func:`generate_dashboard` - produce a ``dashboard_data.js`` for the web UI.
"""

import json
import os
import shutil

import joblib
import keras
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import HyperParams, Paths, FEATURE_COLUMNS, get_device
from .model import LSTMAutoencoder
from .preprocess import Preprocessor


# ===========================================================================
# Loss wrapper for gradient computation
# ===========================================================================
class _ReconstructionLossWrapper(keras.Model):
    """Wraps the autoencoder so that its output is the per-sample MSE scalar."""

    def __init__(self, model: keras.Model, **kwargs):
        super().__init__(**kwargs)
        self._ae = model

    def call(self, x, training=False):
        reconstructed = self._ae(x, training=training)
        return keras.ops.mean(keras.ops.square(reconstructed - x), axis=(1, 2))


# ===========================================================================
# Integrated Gradients (backend-agnostic)
# ===========================================================================
def _compute_ig(
    wrapper: keras.Model,
    input_seq: np.ndarray,
    baseline_seq: np.ndarray,
    steps: int = 50,
) -> np.ndarray:
    """
    Compute Integrated Gradients for a single sequence.

    Args:
        wrapper      : :class:`_ReconstructionLossWrapper` around the AE.
        input_seq    : Input array of shape (1, seq_len, n_features).
        baseline_seq : Baseline (usually zeros), same shape as *input_seq*.
        steps        : Number of interpolation steps.

    Returns:
        Attribution array of shape (seq_len, n_features).
    """
    alphas = np.linspace(0.0, 1.0, steps)
    path = np.concatenate(
        [baseline_seq + a * (input_seq - baseline_seq) for a in alphas], axis=0
    )  # (steps, seq_len, n_features)

    backend = keras.backend.backend()

    if backend == "torch":
        import torch
        path_t = torch.tensor(path, dtype=torch.float32, requires_grad=True)
        out = wrapper(path_t)
        out.sum().backward()
        grads_np = path_t.grad.cpu().detach().numpy()

    elif backend == "tensorflow":
        import tensorflow as tf
        path_t = tf.Variable(tf.cast(path, tf.float32))
        with tf.GradientTape() as tape:
            out = wrapper(path_t)
        grads_np = tape.gradient(out, path_t).numpy()

    elif backend == "jax":
        import jax, jax.numpy as jnp
        grad_fn = jax.vmap(jax.grad(lambda x: wrapper(jnp.expand_dims(x, 0))[0]))
        grads_np = np.array(grad_fn(path))

    else:
        raise NotImplementedError(f"Backend '{backend}' is not supported for IG.")

    avg_grads = np.mean(grads_np, axis=0, keepdims=True)  # (1, seq_len, n_features)
    return np.squeeze((input_seq - baseline_seq) * avg_grads, axis=0)  # (seq_len, n_features)


# ===========================================================================
# Public: explain
# ===========================================================================
def explain(
    limit: int = -1,
    steps: int = 50,
    hp: HyperParams = HyperParams(),
    paths: Paths = Paths(),
    verbose: bool = True,
) -> None:
    """
    Compute Integrated Gradients for the top-N most anomalous sequences and
    save per-sequence text + PNG reports.

    Args:
        limit   : Number of top anomalies to explain (-1 = all).
        steps   : IG integration steps.
        hp      : Hyperparameter configuration (seq_len).
        paths   : File-path configuration.
        verbose : Print progress when True.
    """
    if verbose:
        print("=" * 60)
        print("  INTEGRATED GRADIENTS -- EXPLANATION PIPELINE (Keras)")
        print("=" * 60)

    if not os.path.exists(paths.detected_anomalies):
        raise FileNotFoundError(
            f"Anomaly list not found at {paths.detected_anomalies}. Run detect_anomalies.py first."
        )

    # ------------------------------------------------------------------
    # Load anomalies list
    # ------------------------------------------------------------------
    df_anom = pd.read_csv(paths.detected_anomalies, low_memory=False)
    df_anom = df_anom.sort_values("Reconstruction_Error", ascending=False)
    if limit > 0:
        df_anom = df_anom.head(limit)

    if verbose:
        print(f"[1/3] Explaining {len(df_anom)} sequence(s) ...")

    # ------------------------------------------------------------------
    # Load model + feature names
    # ------------------------------------------------------------------
    model = keras.models.load_model(paths.model, custom_objects={"LSTMAutoencoder": LSTMAutoencoder})
    wrapper = _ReconstructionLossWrapper(model)

    scaler = joblib.load(paths.scaler)
    feature_names = list(scaler.feature_names_in_) if hasattr(scaler, "feature_names_in_") else FEATURE_COLUMNS

    pp = Preprocessor.load(FEATURE_COLUMNS, paths.encoders, paths.scaler)

    data_path = paths.eval_data
    col_names = list(pd.read_csv(data_path, nrows=0).columns)

    explanations_dir = os.path.join(paths.ig_dir, "explanations")
    os.makedirs(explanations_dir, exist_ok=True)

    if verbose:
        print("[2/3] Loading model ... done")

    # ------------------------------------------------------------------
    # Per-sequence explanation loop
    # ------------------------------------------------------------------
    if verbose:
        print("[3/3] Computing attributions ...")

    summary_records = []

    for count, (_, row) in enumerate(df_anom.iterrows()):
        orig_idx = int(row["Original_Row_Index"])
        seq_start = orig_idx - hp.seq_len + 1
        seq_err = float(row["Reconstruction_Error"])
        label = int(row.get("Label", -1))

        if verbose:
            print(f"  [{count+1}/{len(df_anom)}] seq_start={seq_start}  (trigger row={orig_idx+2})")

        # Read raw sequence rows
        try:
            df_seq = pd.read_csv(
                data_path, skiprows=range(1, seq_start + 1), nrows=hp.seq_len,
                names=col_names, low_memory=False,
            )
        except Exception as e:
            print(f"    [WARN] Could not read rows: {e}")
            continue

        if len(df_seq) < hp.seq_len:
            continue

        # Preprocess
        X = pp.transform(df_seq)
        input_seq = np.expand_dims(X, axis=0)   # (1, seq_len, n_features)
        baseline = np.zeros_like(input_seq)

        # Integrated Gradients
        ig = _compute_ig(wrapper, input_seq, baseline, steps=steps)  # (seq_len, n_features)

        # Per-log attribution scores
        log_scores = np.sum(ig, axis=1)  # (seq_len,)
        worst_t = int(np.argmax(log_scores))
        worst_score = float(log_scores[worst_t])
        worst_row = df_seq.iloc[worst_t]

        # Top features at the worst log
        feat_scores_at_worst = ig[worst_t]
        top5_idx = np.argsort(feat_scores_at_worst)[::-1][:5]
        top5 = [(feature_names[i], float(feat_scores_at_worst[i])) for i in top5_idx]

        # --- Save text report ---
        txt_path = os.path.join(explanations_dir, f"anomaly_row_{orig_idx+2}.txt")
        _write_text_report(txt_path, seq_start, orig_idx, seq_err, label, hp.seq_len,
                           worst_t, worst_score, worst_row, df_seq.iloc[-1],
                           feat_scores_at_worst, feature_names)

        # --- Save bar chart (top 5 sequences only) ---
        if count < 5:
            png_path = os.path.join(explanations_dir, f"anomaly_row_{orig_idx+2}.png")
            _plot_attribution_bar(png_path, top5, worst_score, orig_idx + 2)

        summary_records.append({
            "Original_Row_Index": orig_idx,
            "Excel_Row": orig_idx + 2,
            "Reconstruction_Error": seq_err,
            "Label": label,
            "Worst_Log_Step": worst_t + 1,
            "Worst_Log_Excel_Row": seq_start + worst_t + 2,
            "Worst_Log_Score": worst_score,
            "Worst_Log_SystemTime": worst_row.get("SystemTime", "N/A"),
            "Worst_Log_Image": worst_row.get("Image", "N/A"),
            "Worst_Log_CommandLine": worst_row.get("CommandLine", "N/A"),
            **{f"Top{r+1}_Feature": n for r, (n, _) in enumerate(top5)},
            **{f"Top{r+1}_Score": s for r, (_, s) in enumerate(top5)},
        })

    # Save summary CSV
    summary_path = os.path.join(paths.ig_dir, "anomalies_explained_summary.csv")
    pd.DataFrame(summary_records).to_csv(summary_path, index=False)

    # Copy worst-anomaly files to ig_dir root for backward compat
    if summary_records:
        worst_row_id = summary_records[0]["Excel_Row"]
        for ext, dst in [(".txt", "anomaly_explanation.txt"), (".png", "anomaly_attribution.png")]:
            src = os.path.join(explanations_dir, f"anomaly_row_{worst_row_id}{ext}")
            if os.path.exists(src):
                shutil.copy(src, os.path.join(paths.ig_dir, dst))

    if verbose:
        print(f"\n  Done. Summary -> {summary_path}")
        print(f"  Per-sequence files -> {explanations_dir}/")


# ===========================================================================
# Public: generate_dashboard
# ===========================================================================
def generate_dashboard(
    limit: int = 30,
    steps: int = 50,
    hp: HyperParams = HyperParams(),
    paths: Paths = Paths(),
    verbose: bool = True,
) -> None:
    """
    Generate ``dashboard_data.js`` for the web dashboard.

    Produces a JavaScript file containing a ``const ANOMALIES_DATA = [...]``
    assignment with per-sequence IG attributions and raw log fields.

    Args:
        limit   : Maximum number of anomalies to include.
        steps   : IG integration steps.
        hp      : Hyperparameter configuration.
        paths   : File-path configuration.
        verbose : Print progress when True.
    """
    if verbose:
        print("=" * 60)
        print("  GENERATING DASHBOARD DATA (Keras)")
        print("=" * 60)

    if not os.path.exists(paths.detected_anomalies):
        raise FileNotFoundError(
            f"Anomaly list not found at {paths.detected_anomalies}. Run detect_anomalies.py first."
        )

    df_anom = pd.read_csv(paths.detected_anomalies, low_memory=False)
    df_anom = df_anom.sort_values("Reconstruction_Error", ascending=False).head(limit)

    model = keras.models.load_model(paths.model, custom_objects={"LSTMAutoencoder": LSTMAutoencoder})
    wrapper = _ReconstructionLossWrapper(model)

    scaler = joblib.load(paths.scaler)
    feature_names = list(scaler.feature_names_in_) if hasattr(scaler, "feature_names_in_") else FEATURE_COLUMNS
    pp = Preprocessor.load(FEATURE_COLUMNS, paths.encoders, paths.scaler)

    data_path = paths.eval_data
    col_names = list(pd.read_csv(data_path, nrows=0).columns)

    records = []
    for count, (_, row) in enumerate(df_anom.iterrows()):
        orig_idx = int(row["Original_Row_Index"])
        seq_start = orig_idx - hp.seq_len + 1
        seq_err = float(row["Reconstruction_Error"])
        label = int(row.get("Label", -1))

        if verbose:
            print(f"  [{count+1}/{len(df_anom)}] seq_start={seq_start}  (trigger row={orig_idx+2})")

        try:
            df_seq = pd.read_csv(
                data_path, skiprows=range(1, seq_start + 1), nrows=hp.seq_len,
                names=col_names, low_memory=False,
            )
        except Exception:
            continue

        if len(df_seq) < hp.seq_len:
            continue

        X = pp.transform(df_seq)
        input_seq = np.expand_dims(X, axis=0)
        
        # Predict to get reconstructed values and compute feature-level reconstruction errors
        reconstructed = model.predict(input_seq, verbose=0)
        reconstructed_seq = np.squeeze(reconstructed, axis=0)
        recon_errors = np.square(X - reconstructed_seq)  # (seq_len, n_features)

        # Compute Integrated Gradients
        ig = _compute_ig(wrapper, input_seq, np.zeros_like(input_seq), steps=steps)

        log_scores = np.sum(ig, axis=1)
        worst_t = int(np.argmax(log_scores))
        top5_idx = np.argsort(ig[worst_t])[::-1][:5]
        top5 = [{"feature_name": feature_names[i], "score": float(ig[worst_t, i])} for i in top5_idx]

        logs = []
        for t in range(hp.seq_len):
            raw = df_seq.iloc[t].to_dict()
            cleaned = {}
            for k, v in raw.items():
                if pd.isna(v):
                    cleaned[k] = "N/A"
                elif isinstance(v, (np.integer,)):
                    cleaned[k] = int(v)
                elif isinstance(v, (np.floating,)):
                    cleaned[k] = float(v)
                else:
                    cleaned[k] = str(v)
            cleaned["step"] = t + 1
            cleaned["excel_row"] = seq_start + t + 2
            cleaned["attribution_score"] = float(log_scores[t])
            
            # Store reconstruction errors for each feature
            cleaned["feature_errors"] = {
                feature_names[i]: float(recon_errors[t, i]) for i in range(len(feature_names))
            }
            logs.append(cleaned)

        records.append({
            "id": count + 1,
            "orig_row_idx": orig_idx,
            "excel_row": orig_idx + 2,
            "reconstruction_error": seq_err,
            "label": label,
            "trigger_time": str(df_seq.iloc[-1].get("SystemTime", "N/A")),
            "trigger_image": str(df_seq.iloc[-1].get("Image", "N/A")),
            "trigger_command": str(df_seq.iloc[-1].get("CommandLine", "N/A")),
            "worst_log_index": worst_t,
            "worst_log_excel_row": seq_start + worst_t + 2,
            "worst_log_score": float(log_scores[worst_t]),
            "worst_log_top_features": top5,
            "logs": logs,
        })

    os.makedirs(os.path.dirname(paths.dashboard_js), exist_ok=True)
    with open(paths.dashboard_js, "w", encoding="utf-8") as f:
        f.write("const ANOMALIES_DATA = ")
        json.dump(records, f, indent=2, ensure_ascii=False)
        f.write(";\n")

    if verbose:
        print(f"\n  Dashboard data saved -> {paths.dashboard_js}")


# ===========================================================================
# Private helpers
# ===========================================================================
def _write_text_report(path, seq_start, orig_idx, seq_err, label, seq_len,
                        worst_t, worst_score, worst_row, trigger_row,
                        feat_scores, feature_names):
    top10_idx = np.argsort(feat_scores)[::-1][:10]
    with open(path, "w", encoding="utf-8") as f:
        f.write("=== INTEGRATED GRADIENTS -- ANOMALY EXPLANATION REPORT ===\n")
        f.write(f"Sequence start index : {seq_start}\n")
        f.write(f"Trigger row (Excel)  : {orig_idx + 2}\n")
        f.write(f"Reconstruction error : {seq_err:.6f}\n")
        f.write(f"Label                : {'Anomalous (1)' if label == 1 else 'Normal (0)'}\n\n")
        f.write("--- WORST LOG IN SEQUENCE ---\n")
        f.write(f"Step in sequence     : {worst_t + 1}/{seq_len}\n")
        f.write(f"Row (Excel)          : {seq_start + worst_t + 2}\n")
        f.write(f"Attribution score    : {worst_score:.6f}\n")
        f.write(f"SystemTime           : {worst_row.get('SystemTime', 'N/A')}\n")
        f.write(f"Image                : {worst_row.get('Image', 'N/A')}\n")
        f.write(f"CommandLine          : {worst_row.get('CommandLine', 'N/A')}\n")
        f.write(f"EventID              : {worst_row.get('EventID', 'N/A')}\n\n")
        f.write("--- TOP 10 CONTRIBUTING FEATURES AT WORST LOG ---\n")
        for rank, i in enumerate(top10_idx):
            f.write(f"  #{rank+1:02d}  {feature_names[i]:<30} {feat_scores[i]:.6f}\n")
        f.write("\n--- TRIGGER EVENT (last log in sequence) ---\n")
        f.write(f"SystemTime   : {trigger_row.get('SystemTime', 'N/A')}\n")
        f.write(f"Image        : {trigger_row.get('Image', 'N/A')}\n")
        f.write(f"CommandLine  : {trigger_row.get('CommandLine', 'N/A')}\n")


def _plot_attribution_bar(path, top_features, worst_score, excel_row):
    names = [n for n, _ in reversed(top_features)]
    scores = [s for _, s in reversed(top_features)]
    colors = ["#d62728" if v >= 0 else "#1f77b4" for v in scores]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.barh(names, scores, color=colors, height=0.5)
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_title(
        f"Top {len(top_features)} Features -- Worst Log (Excel row {excel_row})\n"
        f"Worst log attribution score: {worst_score:.4f}",
        fontsize=11, fontweight="bold",
    )
    ax.set_xlabel("Attribution Score", fontsize=10)
    ax.grid(True, axis="x", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
