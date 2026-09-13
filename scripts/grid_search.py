"""Grid search script to find the optimal LSTM Autoencoder hyperparameters."""

import os
import shutil
import json
import argparse
from typing import List

from anomaly_detection.config import HyperParams, Paths, DEFAULT_HP
from anomaly_detection.trainer import train
from anomaly_detection.evaluator import evaluate


def run_grid_search(hp_candidates: HyperParams, paths: Paths) -> None:
    # Get candidate lists
    seq_lens = hp_candidates.grid_seq_lens
    batch_sizes = hp_candidates.grid_batch_sizes
    epochs_list = hp_candidates.grid_epochs
    
    print("=" * 60)
    print("  LSTM AUTOENCODER -- GRID SEARCH HYPERPARAMETER OPTIMIZATION")
    print("=" * 60)
    print(f"  Candidate Seq Lens   : {seq_lens}")
    print(f"  Candidate Batch Sizes: {batch_sizes}")
    print(f"  Candidate Epochs     : {epochs_list}")
    print("=" * 60)
    print()

    best_auc = -1.0
    best_val_loss = float("inf")
    best_hp = None
    
    results = []
    
    # Paths for backup
    best_model_backup = paths.model + ".best"
    
    total_combinations = len(seq_lens) * len(batch_sizes) * len(epochs_list)
    combination_idx = 0
    
    for seq_len in seq_lens:
        for batch_size in batch_sizes:
            for epochs in epochs_list:
                combination_idx += 1
                print(f"\n--- Run {combination_idx}/{total_combinations}: seq_len={seq_len}, batch_size={batch_size}, epochs={epochs} ---")
                
                hp = HyperParams(
                    seq_len=seq_len,
                    batch_size=batch_size,
                    epochs=epochs,
                    lr=hp_candidates.lr
                )
                
                try:
                    # 1. Train model (verbose=False to avoid cluttering stdout)
                    train_res = train(hp=hp, paths=paths, verbose=False)
                    val_loss = train_res.best_val_loss
                    
                    # 2. Evaluate model
                    eval_res = evaluate(hp=hp, paths=paths, verbose=False)
                    auc = eval_res.auc
                    
                    print(f"    Result: Best Val Loss = {val_loss:.6f} | ROC AUC = {auc:.4f}")
                    
                    results.append({
                        "seq_len": seq_len,
                        "batch_size": batch_size,
                        "epochs": epochs,
                        "val_loss": val_loss,
                        "auc": auc
                    })
                    
                    # We look for highest ROC AUC. If AUC is equal, we look for lower val loss.
                    is_better = False
                    if auc > best_auc:
                        is_better = True
                    elif abs(auc - best_auc) < 1e-7 and val_loss < best_val_loss:
                        is_better = True
                        
                    if is_better:
                        best_auc = auc
                        best_val_loss = val_loss
                        best_hp = hp
                        print(f"    *** New best model found! (AUC: {best_auc:.4f}) ***")
                        # Backup the best model files
                        if os.path.exists(paths.model):
                            shutil.copy2(paths.model, best_model_backup)
                            
                except Exception as e:
                    print(f"    Error during this run: {e}")
                    continue

    print("\n" + "=" * 60)
    print("  GRID SEARCH COMPLETED")
    print("=" * 60)
    
    # Save grid search summary
    summary_path = os.path.join(paths.root, "artifacts", "results", "grid_search_summary.json")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "best_hp": best_hp.to_dict() if best_hp else None,
            "best_auc": best_auc,
            "best_val_loss": best_val_loss,
            "all_runs": results
        }, f, indent=2)
        
    # Restore the best model to the main model path
    if os.path.exists(best_model_backup):
        if os.path.exists(paths.model):
            os.remove(paths.model)
        shutil.move(best_model_backup, paths.model)
        print(f"  Best model restored to: {paths.model}")
        
    print("\n  Summary of all runs:")
    print("  SeqLen | Batch | Epochs | Val Loss | ROC AUC")
    print("  " + "-" * 44)
    for r in results:
        marker = "  <-- BEST" if r["auc"] == best_auc else ""
        print(f"  {r['seq_len']:6d} | {r['batch_size']:5d} | {r['epochs']:6d} | {r['val_loss']:.6f} | {r['auc']:.4f}{marker}")
        
    if best_hp:
        print("\n  Optimal hyperparameters:")
        print(f"    - seq_len   : {best_hp.seq_len}")
        print(f"    - batch_size: {best_hp.batch_size}")
        print(f"    - epochs    : {best_hp.epochs}")
        print(f"    - ROC AUC   : {best_auc:.4f}")
    else:
        print("\n  No successful run found.")


def main():
    p = argparse.ArgumentParser(description="Grid Search LSTM Autoencoder")
    # Allow overriding lists from command line
    p.add_argument("--seq_lens", type=int, nargs="+", default=DEFAULT_HP.grid_seq_lens)
    p.add_argument("--batch_sizes", type=int, nargs="+", default=DEFAULT_HP.grid_batch_sizes)
    p.add_argument("--epochs", type=int, nargs="+", default=DEFAULT_HP.grid_epochs)
    p.add_argument("--lr", type=float, default=DEFAULT_HP.lr)
    
    args = p.parse_args()
    
    hp_candidates = HyperParams(
        grid_seq_lens=args.seq_lens,
        grid_batch_sizes=args.batch_sizes,
        grid_epochs=args.epochs,
        lr=args.lr
    )
    
    run_grid_search(hp_candidates, Paths())


if __name__ == "__main__":
    main()
