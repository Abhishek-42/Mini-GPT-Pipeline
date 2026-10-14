"""
Model Evaluation & Research Metrics Script for Rix-17M
=======================================================
Evaluates a trained checkpoint on real validation data and computes
all 8 classic ML research metrics:

  1. Accuracy
  2. Precision
  3. Recall
  4. F1-Score
  5. Sensitivity (True Positive Rate)
  6. Specificity (True Negative Rate)
  7. Confusion Matrix (TP, FP, TN, FN)
  8. ROC Curve & AUC Score (saved as PNG to logs/)

Usage:
  python scripts/evaluate_model.py --checkpoint models/checkpoints/final_model.pt
  python scripts/evaluate_model.py --checkpoint models/checkpoints/checkpoint_step_500.pt
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (no GUI needed)
import matplotlib.pyplot as plt

import yaml
import torch
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
from tqdm import tqdm

# Fix imports so it works from any directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.prepare_data import prepare_dataloaders, RAW_DATA_DIR, TOKENIZER_DIR

CONFIG_PATH = PROJECT_ROOT / "configs" / "model_config.yaml"
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def load_trained_model(config, tokenizer, checkpoint_path, device):
    """Loads the GPT-2 architecture and fills it with trained weights."""
    model_cfg = config["model"]

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else eos_id

    hf_config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_embd=model_cfg["hidden_size"],
        n_layer=model_cfg["num_hidden_layers"],
        n_head=model_cfg["num_attention_heads"],
        n_inner=model_cfg["intermediate_size"],
        n_positions=model_cfg["max_position_embeddings"],
        resid_pdrop=0.0,   # No dropout during evaluation
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        activation_function=model_cfg.get("activation", "gelu_new"),
        bos_token_id=bos_id,
        eos_token_id=eos_id,
        pad_token_id=pad_id,
    )

    model = GPT2LMHeadModel(hf_config)

    # Load checkpoint weights
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        step = ckpt.get("step", "N/A")
        epoch = ckpt.get("epoch", "N/A")
        print(f"[INFO] Loaded checkpoint from Step {step}, Epoch {epoch}")
    elif isinstance(ckpt, dict):
        model.load_state_dict(ckpt)
        print(f"[INFO] Loaded raw state_dict checkpoint")
    else:
        raise ValueError(f"Unknown checkpoint format in {checkpoint_path}")

    model.to(device)
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Model loaded: {num_params / 1e6:.2f}M parameters on {device}")
    return model


@torch.no_grad()
def evaluate_on_validation(model, val_loader, device, max_batches=200):
    """
    Runs the trained model on validation data and collects:
    - y_true: For each token position, 1 if the model's TOP prediction matches
              the actual next token, 0 otherwise.
    - y_scores: The model's confidence (softmax probability) for the correct
                next token at each position.
    """
    all_y_true = []
    all_y_scores = []

    device_type = "cuda" if "cuda" in str(device) else "cpu"

    print(f"\n[INFO] Evaluating model on {min(max_batches, len(val_loader))} validation batches...")

    for batch_idx, batch in enumerate(tqdm(val_loader, desc="Evaluating", total=min(max_batches, len(val_loader)))):
        if batch_idx >= max_batches:
            break

        input_ids = batch["input_ids"].to(device)  # (B, seq_len)

        # Forward pass
        if device_type == "cuda":
            with torch.amp.autocast("cuda"):
                outputs = model(input_ids)
        else:
            outputs = model(input_ids)

        # logits shape: (B, seq_len, vocab_size)
        logits = outputs.logits

        # Shift: predict token[t+1] from position[t]
        # predictions come from logits[:, :-1, :], targets are input_ids[:, 1:]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        # Get softmax probabilities
        probs = torch.softmax(shift_logits, dim=-1)  # (B, seq_len-1, vocab_size)

        # Top-1 prediction
        top1_preds = torch.argmax(shift_logits, dim=-1)  # (B, seq_len-1)

        # Was the top prediction correct? (Binary: 1=correct, 0=wrong)
        correct = (top1_preds == shift_labels).cpu().numpy().flatten()

        # Model's confidence score for the ACTUAL correct token
        # Gather the probability the model assigned to the true next token
        true_token_probs = probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
        confidence = true_token_probs.cpu().numpy().flatten()

        all_y_true.append(correct)
        all_y_scores.append(confidence)

    y_true = np.concatenate(all_y_true)
    y_scores = np.concatenate(all_y_scores)

    print(f"[INFO] Total token predictions evaluated: {len(y_true):,}")
    return y_true, y_scores


def compute_confusion_matrix(y_true, y_pred):
    """Computes TP, FP, TN, FN from binary arrays."""
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tp, fp, tn, fn


def compute_roc_curve(y_true, y_scores):
    """Calculates FPR, TPR, and AUC across confidence thresholds."""
    thresholds = np.linspace(0.0, 1.0, 201)
    tprs = []
    fprs = []

    for thresh in thresholds:
        y_pred = (y_scores >= thresh).astype(int)
        tp, fp, tn, fn = compute_confusion_matrix(y_true, y_pred)
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        tprs.append(tpr)
        fprs.append(fpr)

    # Sort for monotonic curve
    sorted_idx = np.argsort(fprs)
    fprs = np.array(fprs)[sorted_idx]
    tprs = np.array(tprs)[sorted_idx]

    # AUC via trapezoidal rule
    auc = np.trapezoid(tprs, fprs) if hasattr(np, "trapezoid") else np.trapz(tprs, fprs)
    return fprs, tprs, abs(auc)


def compute_all_metrics(y_true, y_scores, threshold=0.5):
    """Calculates all 8 research metrics."""
    y_pred = (y_scores >= threshold).astype(int)
    tp, fp, tn, fn = compute_confusion_matrix(y_true, y_pred)
    total = len(y_true)

    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    sensitivity = recall  # Same as recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    fprs, tprs, roc_auc = compute_roc_curve(y_true, y_scores)

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "ROC_AUC": roc_auc,
        "FPR": fprs,
        "TPR": tprs,
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
    }


def plot_roc_curve(fprs, tprs, roc_auc, output_path):
    """Generates and saves a publication-quality ROC Curve graph."""
    plt.figure(figsize=(7, 6))
    plt.plot(fprs, tprs, color="#1f77b4", lw=2.5, label=f"Rix-17M ROC Curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="grey", lw=1.5, linestyle="--", label="Random Classifier (AUC = 0.5000)")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=12)
    plt.title("ROC Curve — Rix-17M Next-Token Prediction", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved ROC Curve plot to: {output_path}")


def plot_confusion_matrix(tp, fp, tn, fn, output_path):
    """Generates and saves a visual confusion matrix."""
    matrix = np.array([[tn, fp], [fn, tp]])
    labels = ["Negative\n(Wrong Prediction)", "Positive\n(Correct Prediction)"]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="Blues", interpolation="nearest")

    # Add text annotations
    for i in range(2):
        for j in range(2):
            color = "white" if matrix[i, j] > matrix.max() / 2 else "black"
            ax.text(j, i, f"{matrix[i, j]:,}", ha="center", va="center",
                    fontsize=16, fontweight="bold", color=color)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Predicted", fontsize=12, fontweight="bold")
    ax.set_ylabel("Actual", fontsize=12, fontweight="bold")
    ax.set_title("Confusion Matrix — Rix-17M", fontsize=14, fontweight="bold")

    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved Confusion Matrix plot to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained Rix-17M model")
    parser.add_argument("--checkpoint", "-c", type=str, required=True,
                        help="Path to trained .pt checkpoint file")
    parser.add_argument("--batches", "-b", type=int, default=200,
                        help="Number of validation batches to evaluate (default: 200)")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"[ERROR] Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 65)
    print("  MODEL RESEARCH EVALUATION REPORT (Rix-17M)")
    print("=" * 65)
    print(f"  Checkpoint : {ckpt_path}")
    print(f"  Device     : {device}")

    # Load tokenizer
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(TOKENIZER_DIR))

    # Load trained model
    model = load_trained_model(config, tokenizer, ckpt_path, device)

    # Prepare validation data
    raw_text_file = RAW_DATA_DIR / "fineweb_edu_100000.txt"
    if not raw_text_file.exists():
        raw_text_file = RAW_DATA_DIR / "fineweb_edu_10000.txt"
    if not raw_text_file.exists():
        print(f"[ERROR] No dataset found in {RAW_DATA_DIR}. Run prepare_data.py first.")
        sys.exit(1)

    print(f"  Dataset    : {raw_text_file.name}")

    _, val_loader = prepare_dataloaders(
        raw_text_file=raw_text_file,
        tokenizer=tokenizer,
        max_length=config["model"]["max_position_embeddings"],
        batch_size=config["training"]["batch_size"]
    )

    # Run evaluation on real validation data
    y_true, y_scores = evaluate_on_validation(model, val_loader, device, max_batches=args.batches)

    # Compute all 8 metrics
    metrics = compute_all_metrics(y_true, y_scores, threshold=0.5)

    # Print results
    print("\n" + "=" * 65)
    print("[EVALUATION PARAMETERS]")
    print(f"  1. Accuracy    : {metrics['Accuracy']:.4f}  ({metrics['Accuracy']*100:.2f}%)")
    print(f"  2. Precision   : {metrics['Precision']:.4f}")
    print(f"  3. Recall      : {metrics['Recall']:.4f}")
    print(f"  4. F1-Score    : {metrics['F1-Score']:.4f}")
    print(f"  5. Sensitivity : {metrics['Sensitivity']:.4f}  (True Positive Rate)")
    print(f"  6. Specificity : {metrics['Specificity']:.4f}  (True Negative Rate)")
    print(f"  7. ROC-AUC     : {metrics['ROC_AUC']:.4f}")

    print(f"\n[CONFUSION MATRIX]")
    print(f"  True Positives  (TP) : {metrics['TP']:,}")
    print(f"  False Positives (FP) : {metrics['FP']:,}")
    print(f"  True Negatives  (TN) : {metrics['TN']:,}")
    print(f"  False Negatives (FN) : {metrics['FN']:,}")

    # Save ROC Curve plot
    roc_path = LOG_DIR / "roc_curve_model.png"
    plot_roc_curve(metrics["FPR"], metrics["TPR"], metrics["ROC_AUC"], roc_path)

    # Save Confusion Matrix plot
    cm_path = LOG_DIR / "confusion_matrix_model.png"
    plot_confusion_matrix(metrics["TP"], metrics["FP"], metrics["TN"], metrics["FN"], cm_path)

    print("\n" + "=" * 65)
    print("  Evaluation complete! Results and graphs saved to logs/ folder.")
    print("=" * 65)


if __name__ == "__main__":
    main()
