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
from .model import LSTMAutoencoder, EventSpecificMaskedMSE
from .preprocess import Preprocessor
from .dataset import build_tree_dataset, vectorize_tree_sequences


# ===========================================================================
# Loss wrapper for gradient computation
# ===========================================================================
class _ReconstructionLossWrapper(keras.Model):
    """Wraps the autoencoder so that its output is the per-sample MSE scalar."""

    def __init__(self, model: keras.Model, mask_matrix: np.ndarray, event_col_indices: list, **kwargs):
        super().__init__(**kwargs)
        self._ae = model
        self.mask_matrix = keras.ops.convert_to_tensor(mask_matrix, dtype="float32")
        self.event_col_indices = list(event_col_indices)

    def call(self, x, training=False):
        reconstructed = self._ae(x, training=training)
        
        # 1. Step-wise mask: Mask out zero-padded steps
        step_mask = keras.ops.any(x != 0, axis=2)
        step_mask_float = keras.ops.cast(step_mask, "float32")
        
        # 2. Event-specific feature mask
        one_hot_events = keras.ops.take(x, self.event_col_indices, axis=2)
        feature_mask = keras.ops.matmul(one_hot_events, self.mask_matrix)
        
        # 3. Compute masked squared error
        sq_err = keras.ops.square(reconstructed - x)
        masked_sq_err = sq_err * feature_mask
        
        # Average over active features
        active_feature_counts = keras.ops.sum(feature_mask, axis=2, keepdims=True)
        active_feature_counts = keras.ops.maximum(active_feature_counts, 1.0)
        mse_per_step = keras.ops.sum(masked_sq_err, axis=2) / keras.ops.squeeze(active_feature_counts, axis=2)
        
        # Average over active steps
        total_active_steps = keras.ops.sum(step_mask_float, axis=1)
        total_active_steps = keras.ops.maximum(total_active_steps, 1.0)
        loss = keras.ops.sum(mse_per_step * step_mask_float, axis=1) / total_active_steps
        
        return loss


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
        path_t = tf.convert_to_tensor(path, dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(path_t)
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
    limit: int = 10,
    steps: int = 50,
    hp: HyperParams = HyperParams(),
    paths: Paths = Paths(),
    verbose: bool = True,
    stride: int = 3,
) -> None:
    """
    Compute Integrated Gradients for the top-N most anomalous sequences and
    save per-sequence text + PNG reports.
    """
    if verbose:
        print("=" * 60)
        print("  INTEGRATED GRADIENTS -- EXPLANATION PIPELINE (Keras)")
        print("=" * 60)

    if not os.path.exists(paths.detected_anomalies):
        raise FileNotFoundError(
            f"Anomaly list not found at {paths.detected_anomalies}. Run detect_anomalies.py first."
        )

    df_anom = pd.read_csv(paths.detected_anomalies, low_memory=False)
    df_anom = df_anom.sort_values("Reconstruction_Error", ascending=False)
    if limit > 0:
        df_anom = df_anom.head(limit)

    if verbose:
        print(f"[1/3] Explaining {len(df_anom)} sequence(s) ...")

    pp = Preprocessor.load(FEATURE_COLUMNS, paths.encoders, paths.scaler)
    scaler = joblib.load(paths.scaler)
    feature_names = list(scaler.feature_names_in_) if hasattr(scaler, "feature_names_in_") else FEATURE_COLUMNS

    event_col_indices = [feature_names.index(f"EventID_{eid}") for eid in pp.event_ids]
    mask_matrix = pp.get_event_feature_mask()

    model = keras.models.load_model(
        paths.model, 
        custom_objects={
            "LSTMAutoencoder": LSTMAutoencoder, 
            "EventSpecificMaskedMSE": EventSpecificMaskedMSE
        }
    )
    wrapper = _ReconstructionLossWrapper(model, mask_matrix, event_col_indices)

    raw_seqs, seq_labels, metadatas = build_tree_dataset(paths.eval_data, stride=stride)
    sequences = vectorize_tree_sequences(raw_seqs, pp, fit=False, seq_len=hp.seq_len)

    explanations_dir = os.path.join(paths.ig_dir, "explanations")
    os.makedirs(explanations_dir, exist_ok=True)

    if verbose:
        print("[2/3] Loading model ... done")
        print("[3/3] Computing attributions ...")

    summary_records = []
    for count, (_, row) in enumerate(df_anom.iterrows()):
        seq_id = int(row["Sequence_ID"]) if "Sequence_ID" in row and pd.notna(row["Sequence_ID"]) else count
        if seq_id >= len(raw_seqs):
            continue

        raw_seq = raw_seqs[seq_id]
        meta = metadatas[seq_id]
        input_seq = sequences[seq_id : seq_id + 1]
        baseline = np.zeros_like(input_seq)
        seq_err = float(row["Reconstruction_Error"])
        label = int(row.get("Label", 0))
        orig_idx = int(row.get("Original_Row_Index", meta.get("trigger_event", {}).get("_row_idx", -1)))

        if verbose:
            print(f"  [{count+1}/{len(df_anom)}] seq_id={seq_id}  (trigger row={orig_idx+2})")

        # Compute Integrated Gradients
        ig = _compute_ig(wrapper, input_seq, baseline, steps=steps)  # (seq_len, n_features)

        n_active = len(raw_seq)
        log_scores = np.sum(ig[:n_active, :], axis=1)
        worst_t = int(np.argmax(log_scores))
        worst_score = float(log_scores[worst_t])
        worst_row = raw_seq[worst_t]
        feat_scores_at_worst = ig[worst_t]

        # Top features across all active steps - FILTER OUT EventID_* to highlight real values!
        all_seq_features = []
        for t in range(n_active):
            step_role = "Cha" if t == 0 else ("Hiện tại" if t == 1 else f"Hành vi #{t-1}")
            for i, f_name in enumerate(feature_names):
                if f_name.startswith("EventID_"):
                    continue
                all_seq_features.append((f"[{step_role}] {f_name}", float(ig[t, i])))

        all_seq_features.sort(key=lambda x: x[1], reverse=True)
        top5 = all_seq_features[:5]

        step_indices = [ev.get("_row_idx", -1) for ev in raw_seq]

        # --- Save text report ---
        txt_path = os.path.join(explanations_dir, f"anomaly_row_{orig_idx+2}.txt")
        _write_text_report(txt_path, step_indices, orig_idx, seq_err, label, n_active,
                           worst_t, worst_score, worst_row, meta.get("trigger_event", raw_seq[-1]),
                           feat_scores_at_worst, feature_names)

        # --- Save bar chart (top 5 sequences only) ---
        if count < 5:
            png_path = os.path.join(explanations_dir, f"anomaly_row_{orig_idx+2}.png")
            _plot_attribution_bar(png_path, top5, worst_score, orig_idx + 2)

        summary_records.append({
            "Sequence_ID": seq_id,
            "Original_Row_Index": orig_idx,
            "Excel_Row": orig_idx + 2,
            "Reconstruction_Error": seq_err,
            "Label": label,
            "Worst_Log_Step": worst_t + 1,
            "Worst_Log_Excel_Row": (step_indices[worst_t] + 2) if step_indices[worst_t] != -1 else "N/A",
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
    limit: int = 100,
    steps: int = 50,
    hp: HyperParams = HyperParams(),
    paths: Paths = Paths(),
    verbose: bool = True,
    stride: int = 3,
) -> None:
    """
    Generate ``dashboard_data.js`` for the web dashboard.
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
    df_anom = df_anom.sort_values("Reconstruction_Error", ascending=False)
    if limit and limit > 0:
        df_anom = df_anom.head(limit)

    pp = Preprocessor.load(FEATURE_COLUMNS, paths.encoders, paths.scaler)
    scaler = joblib.load(paths.scaler)
    feature_names = list(scaler.feature_names_in_) if hasattr(scaler, "feature_names_in_") else FEATURE_COLUMNS

    event_col_indices = [feature_names.index(f"EventID_{eid}") for eid in pp.event_ids]
    mask_matrix = pp.get_event_feature_mask()

    model = keras.models.load_model(
        paths.model, 
        custom_objects={
            "LSTMAutoencoder": LSTMAutoencoder, 
            "EventSpecificMaskedMSE": EventSpecificMaskedMSE
        }
    )
    wrapper = _ReconstructionLossWrapper(model, mask_matrix, event_col_indices)

    raw_seqs, seq_labels, metadatas = build_tree_dataset(paths.eval_data, stride=stride)
    sequences = vectorize_tree_sequences(raw_seqs, pp, fit=False, seq_len=hp.seq_len)

    records = []
    for count, (_, row) in enumerate(df_anom.iterrows()):
        seq_id = int(row["Sequence_ID"]) if "Sequence_ID" in row and pd.notna(row["Sequence_ID"]) else count
        if seq_id >= len(raw_seqs):
            continue

        raw_seq = raw_seqs[seq_id]
        meta = metadatas[seq_id]
        input_seq = sequences[seq_id : seq_id + 1]
        baseline = np.zeros_like(input_seq)
        seq_err = float(row["Reconstruction_Error"])
        label = int(row.get("Label", 0))
        threshold = float(row.get("Threshold", 0.0485))
        orig_idx = int(row.get("Original_Row_Index", meta.get("trigger_event", {}).get("_row_idx", -1)))

        if verbose and (count % 100 == 0 or count == len(df_anom) - 1):
            print(f"  [{count+1}/{len(df_anom)}] seq_id={seq_id}  (trigger row={orig_idx+2})")

        # Predict reconstructed sequence
        reconstructed = keras.ops.convert_to_numpy(model(input_seq, training=False))
        recon_errors = np.square(input_seq[0] - reconstructed[0])  # (seq_len, n_features)

        # Compute Integrated Gradients
        ig = _compute_ig(wrapper, input_seq, baseline, steps=steps)  # (seq_len, n_features)

        n_active = len(raw_seq)
        log_scores = np.sum(ig[:n_active, :], axis=1)
        worst_t = int(np.argmax(log_scores))

        # Top features across all active steps, EXCLUDING EventID_*
        all_seq_features = []
        for t in range(n_active):
            step_role = "Cha" if t == 0 else ("Hiện tại" if t == 1 else f"Hành vi #{t-1}")
            for i, f_name in enumerate(feature_names):
                if f_name.startswith("EventID_"):
                    continue
                all_seq_features.append({
                    "feature_name": f"[{step_role}] {f_name}",
                    "score": float(ig[t, i]),
                    "error": float(recon_errors[t, i]),
                    "step": t + 1
                })

        all_seq_features.sort(key=lambda x: x["score"], reverse=True)
        top5 = all_seq_features[:5]

        logs = []
        for t in range(n_active):
            raw = raw_seq[t]
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
            cleaned["step_role"] = "Cha" if t == 0 else ("Hiện tại" if t == 1 else f"Hành vi #{t-1}")
            actual_idx = raw.get("_row_idx", -1)
            cleaned["excel_row"] = (actual_idx + 2) if actual_idx != -1 else "N/A"
            cleaned["attribution_score"] = float(log_scores[t])
            cleaned["feature_errors"] = {
                feature_names[i]: float(recon_errors[t, i])
                for i in range(len(feature_names))
                if not feature_names[i].startswith("EventID_")
            }
            logs.append(cleaned)

        trigger_event = meta.get("trigger_event", raw_seq[-1])
        p_dict = meta.get("parent", {})
        c_dict = meta.get("current", {})
        proc_guid = str(c_dict.get("ProcessGuid", ""))
        proc_pid = str(c_dict.get("ProcessId", ""))
        proc_img = str(c_dict.get("Image", ""))
        proc_time = str(c_dict.get("SystemTime", ""))
        proc_comp = str(c_dict.get("Computer") or trigger_event.get("Computer") or "")
        proc_key = f"{proc_comp}_{proc_guid or proc_pid or proc_img}_{proc_time}"

        window_times = [l.get("SystemTime") for l in logs if l.get("SystemTime") and l.get("SystemTime") != "N/A"]
        start_time = window_times[0] if window_times else "N/A"
        end_time = window_times[-1] if window_times else "N/A"

        records.append({
            "id": count + 1,
            "orig_row_idx": orig_idx,
            "excel_row": orig_idx + 2 if orig_idx != -1 else "N/A",
            "computer": proc_comp,
            "process_key": proc_key,
            "process_image": proc_img,
            "process_pid": proc_pid,
            "parent_image": str(p_dict.get("Image", "")),
            "start_time": str(start_time),
            "end_time": str(end_time),
            "reconstruction_error": seq_err,
            "threshold": threshold,
            "label": label,
            "trigger_time": str(trigger_event.get("SystemTime", "N/A")),
            "trigger_image": str(trigger_event.get("Image", "N/A")),
            "trigger_command": str(trigger_event.get("CommandLine", "N/A")),
            "worst_log_index": worst_t,
            "worst_log_excel_row": (raw_seq[worst_t].get("_row_idx", -1) + 2) if raw_seq[worst_t].get("_row_idx", -1) != -1 else "N/A",
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
def _write_text_report(path, step_indices, orig_idx, seq_err, label, seq_len,
                        worst_t, worst_score, worst_row, trigger_row,
                        feat_scores, feature_names):
    valid_indices = [i for i, fn in enumerate(feature_names) if not fn.startswith("EventID_")]
    valid_scores = [feat_scores[i] for i in valid_indices]
    sorted_order = np.argsort(valid_scores)[::-1][:10]
    top10_idx = [valid_indices[k] for k in sorted_order]
    with open(path, "w", encoding="utf-8") as f:
        f.write("=== INTEGRATED GRADIENTS -- ANOMALY EXPLANATION REPORT ===\n")
        f.write(f"Sequence indices     : {step_indices}\n")
        f.write(f"Trigger row (Excel)  : {orig_idx + 2}\n")
        f.write(f"Reconstruction error : {seq_err:.6f}\n")
        f.write(f"Label                : {'Anomalous (1)' if label == 1 else 'Normal (0)'}\n\n")
        f.write("--- WORST LOG IN SEQUENCE ---\n")
        f.write(f"Step in sequence     : {worst_t + 1}/{seq_len}\n")
        worst_actual_idx = step_indices[worst_t]
        f.write(f"Row (Excel)          : {worst_actual_idx + 2 if worst_actual_idx != -1 else 'N/A'}\n")
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
