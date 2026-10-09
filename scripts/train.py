"""
LLM Training Loop for Rix-17M
=========================================================
Trains the language model on the prepared dataset using settings
optimized for a 4GB VRAM GPU (RTX 3050). Includes TensorBoard,
progress bars, auto-resume checkpointing, and validation tracking.
"""

import os
import sys
import yaml
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
from transformers import get_cosine_schedule_with_warmup

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import our data pipeline
from scripts.prepare_data import prepare_dataloaders, RAW_DATA_DIR, TOKENIZER_DIR
CONFIG_PATH = PROJECT_ROOT / "configs" / "model_config.yaml"
CHECKPOINT_DIR = PROJECT_ROOT / "models" / "checkpoints"
LOG_DIR = PROJECT_ROOT / "logs"

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

def setup_model(config, tokenizer):
    print("[INFO] Initializing GPT-2 architecture...")
    model_cfg = config["model"]
    
    # Use tokenizer's actual IDs
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    bos_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else eos_token_id
    
    hf_config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_embd=model_cfg["hidden_size"],
        n_layer=model_cfg["num_hidden_layers"],
        n_head=model_cfg["num_attention_heads"],
        n_inner=model_cfg["intermediate_size"],
        n_positions=model_cfg["max_position_embeddings"],
        resid_pdrop=model_cfg.get("dropout", 0.1),
        embd_pdrop=model_cfg.get("dropout", 0.1),
        attn_pdrop=model_cfg.get("dropout", 0.1),
        activation_function=model_cfg.get("activation", "gelu_new"),
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
    )
    
    model = GPT2LMHeadModel(hf_config)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Model Parameter Count: {num_params / 1e6:.2f} M")
    return model

@torch.no_grad()
def evaluate(model, val_loader, device, eval_iters=50):
    """Calculates validation loss without keeping gradients to save VRAM."""
    model.eval()
    losses = []
    
    device_type = "cuda" if "cuda" in str(device) else "cpu"
    
    for idx, batch in enumerate(val_loader):
        if idx >= eval_iters:
            break
        
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        
        # Use autocast for eval to save memory and speed up (only on CUDA)
        if device_type == "cuda":
            with torch.amp.autocast("cuda"):
                outputs = model(input_ids, labels=labels)
        else:
            outputs = model(input_ids, labels=labels)
            
        losses.append(outputs.loss.item())
        
    model.train()
    return sum(losses) / len(losses)

def main(verify_only=False, resume_checkpoint=None):
    print("+" * 60)
    print("  Initializing Rix-17M Training Pipeline")
    print("+" * 60)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    config = load_config()
    train_cfg = config["training"]
    
    device_name = config.get("hardware", {}).get("device", "cuda")
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    device_type = "cuda" if "cuda" in str(device) else "cpu"
    print(f"[INFO] Using device: {device}")
    
    if not TOKENIZER_DIR.exists():
        print(f"[ERROR] Tokenizer not found at {TOKENIZER_DIR}. Run prepare_data.py first.")
        sys.exit(1)
        
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(TOKENIZER_DIR))
    raw_text_file = RAW_DATA_DIR / "fineweb_edu_10000.txt"
    
    if not raw_text_file.exists():
        print(f"[ERROR] Data file not found: {raw_text_file}")
        sys.exit(1)
        
    train_loader, val_loader = prepare_dataloaders(
        raw_text_file=raw_text_file,
        tokenizer=tokenizer,
        max_length=config["model"]["max_position_embeddings"],
        batch_size=train_cfg["batch_size"]
    )
    
    model = setup_model(config, tokenizer)
    model.to(device)
    
    optimizer = AdamW(
        model.parameters(), 
        lr=float(train_cfg["learning_rate"]), 
        weight_decay=float(train_cfg["weight_decay"])
    )
    
    epochs = train_cfg["epochs"]
    grad_accum_steps = train_cfg["gradient_accumulation_steps"]
    total_steps = (len(train_loader) // grad_accum_steps) * epochs
    warmup_steps = train_cfg["warmup_steps"]
    
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    scaler = torch.amp.GradScaler(device_type, enabled=(device_type == "cuda" and train_cfg.get("fp16", True)))

    global_step = 0
    start_epoch = 0
    batch_idx_start = 0

    # Resume from checkpoint if requested
    if resume_checkpoint is not None:
        print(f"\n[INFO] Loading checkpoint state from: {resume_checkpoint}")
        ckpt = torch.load(resume_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "scaler_state_dict" in ckpt and hasattr(scaler, "load_state_dict"):
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        if "step" in ckpt:
            global_step = ckpt["step"]
        if "epoch" in ckpt:
            start_epoch = ckpt["epoch"]
            # Calculate where in the epoch we were
            steps_per_epoch = len(train_loader) // grad_accum_steps
            steps_completed_in_epoch = global_step % steps_per_epoch
            batch_idx_start = steps_completed_in_epoch * grad_accum_steps
            
        print(f"[INFO] Successfully resumed! Starting from Step {global_step}, Epoch {start_epoch + 1}")

    # Setup TensorBoard
    tb_writer = SummaryWriter(log_dir=str(LOG_DIR))
    
    print("\n" + "=" * 60)
    print("  TRAINING STARTED")
    print("=" * 60)
    
    model.train()
    save_steps = config.get("checkpointing", {}).get("save_steps", 500)
    eval_steps = 250  # Evaluate validation loss every 250 steps
    
    for epoch in range(start_epoch, epochs):
        epoch_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", dynamic_ncols=True)
        
        for batch_idx, batch in enumerate(epoch_iterator):
            # Fast-forward if resuming mid-epoch
            if epoch == start_epoch and batch_idx < batch_idx_start:
                continue
                
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            if device_type == "cuda" and train_cfg.get("fp16", True):
                with torch.amp.autocast("cuda"):
                    outputs = model(input_ids, labels=labels)
                    loss = outputs.loss / grad_accum_steps
            else:
                outputs = model(input_ids, labels=labels)
                loss = outputs.loss / grad_accum_steps
            
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["max_grad_norm"])
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                
                global_step += 1
                actual_loss = loss.item() * grad_accum_steps
                
                # Update progress bar and TensorBoard
                epoch_iterator.set_postfix({"Loss": f"{actual_loss:.4f}", "LR": f"{scheduler.get_last_lr()[0]:.2e}"})
                tb_writer.add_scalar("Train/Loss", actual_loss, global_step)
                tb_writer.add_scalar("Train/LearningRate", scheduler.get_last_lr()[0], global_step)
                
                # Periodic Validation
                if global_step % eval_steps == 0:
                    val_loss = evaluate(model, val_loader, device)
                    tb_writer.add_scalar("Validation/Loss", val_loss, global_step)
                    print(f"\n[Step {global_step}] Validation Loss: {val_loss:.4f}")
                
                # Periodic Checkpointing
                if global_step % save_steps == 0:
                    ckpt_path = CHECKPOINT_DIR / f"checkpoint_step_{global_step}.pt"
                    save_dict = {
                        'step': global_step,
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'loss': actual_loss
                    }
                    if hasattr(scaler, "state_dict"):
                        save_dict['scaler_state_dict'] = scaler.state_dict()
                    torch.save(save_dict, ckpt_path)
                    print(f"\n[INFO] Checkpoint successfully saved to {ckpt_path.name}")
                
                if verify_only and global_step >= 5:
                    print("\n[SUCCESS] Verification complete. Stopping early.")
                    tb_writer.close()
                    return
                    
        # Reset batch_idx_start for next epoch
        batch_idx_start = 0

    tb_writer.close()
    
    # Final save (Raw PyTorch Checkpoint)
    final_path = CHECKPOINT_DIR / "final_model.pt"
    save_dict = {
        'step': global_step,
        'epoch': epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }
    torch.save(save_dict, final_path)
    
    # Final save (Hugging Face Format - Needed for Ollama/GGUF Conversion)
    hf_save_dir = CHECKPOINT_DIR / "Rix-17M-HF"
    model.save_pretrained(str(hf_save_dir))
    tokenizer.save_pretrained(str(hf_save_dir))
    
    print(f"\n[INFO] Training complete!")
    print(f"  -> Raw PyTorch checkpoint saved to: {final_path}")
    print(f"  -> Hugging Face format (for Ollama conversion) saved to: {hf_save_dir}")

def find_latest_checkpoint():
    """Finds the most recent checkpoint in models/checkpoints/"""
    ckpts = list(CHECKPOINT_DIR.glob("checkpoint_step_*.pt"))
    if not ckpts:
        return None
    # Sort by step number
    return max(ckpts, key=lambda p: int(p.stem.split("_")[-1]))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true", help="Run for 5 steps")
    parser.add_argument("--resume", "-r", type=str, nargs="?", const="latest", default=None,
                        help="Resume training from a checkpoint file, or 'latest' for the newest checkpoint")
    args = parser.parse_args()

    # Pass resume path if requested
    resume_path = None
    if args.resume:
        if args.resume == "latest":
            resume_path = find_latest_checkpoint()
            if resume_path is None:
                print("[WARNING] No previous checkpoints found in models/checkpoints/. Starting from scratch.")
        else:
            resume_path = Path(args.resume)
            if not resume_path.exists():
                print(f"[ERROR] Specified checkpoint not found: {resume_path}")
                sys.exit(1)

    main(verify_only=args.verify, resume_checkpoint=resume_path)
