"""
Text Generation Script (Pre-training vs Post-training)
======================================================
Tests prompt generation on an untrained (random weights) or a saved checkpoint.

Usage:
  # 1. Untrained (random weights test):
  python scripts/generate.py "The universe is"

  # 2. From a trained checkpoint:
  python scripts/generate.py --checkpoint models/checkpoints/checkpoint_step_500.pt "The universe is"
  python scripts/generate.py --checkpoint models/checkpoints/final_model.pt "Photosynthesis is"
"""

import sys
import yaml
import argparse
from pathlib import Path
import torch
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "model_config.yaml"
TOKENIZER_DIR = PROJECT_ROOT / "models" / "tokenizer"


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def generate_text(prompt: str, checkpoint_path: str = None, max_new_tokens: int = 60, temperature: float = 0.8, top_k: int = 50):
    config = load_config()
    model_cfg = config["model"]

    # 1. Load Tokenizer
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(TOKENIZER_DIR))

    # 2. Build model config
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    bos_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else eos_token_id
    
    hf_config = GPT2Config(
        vocab_size=model_cfg["vocab_size"],
        n_embd=model_cfg["hidden_size"],
        n_layer=model_cfg["num_hidden_layers"],
        n_head=model_cfg["num_attention_heads"],
        n_inner=model_cfg["intermediate_size"],
        n_positions=model_cfg["max_position_embeddings"],
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
    )
    model = GPT2LMHeadModel(hf_config)

    # 3. Load checkpoint if provided
    is_trained = False
    if checkpoint_path:
        ckpt_file = Path(checkpoint_path)
        if not ckpt_file.exists():
            print(f"[ERROR] Checkpoint file not found: {ckpt_file}")
            sys.exit(1)
        print(f"[INFO] Loading trained weights from: {ckpt_file}")
        state = torch.load(ckpt_file, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        elif isinstance(state, dict):
            model.load_state_dict(state)
        is_trained = True
    else:
        print("[INFO] No checkpoint specified -> Using untrained model (random weights)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # 4. Encode prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    print("=" * 60)
    print(f"PROMPT : '{prompt}'")
    status_label = "TRAINED CHECKPOINT" if is_trained else "UNTRAINED (RANDOM WEIGHTS)"
    print(f"MODE   : {status_label}")
    print("=" * 60)

    # 5. Generate next tokens
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_k=top_k,
            pad_token_id=tokenizer.pad_token_id or 0
        )

    # 6. Decode safely for Windows console
    generated_text = tokenizer.decode(output[0].tolist(), skip_special_tokens=True)
    safe_output = generated_text.encode(sys.stdout.encoding or "utf-8", errors="replace").decode(sys.stdout.encoding or "utf-8")
    print(safe_output)
    print("\n" + "=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text with custom GPT model")
    parser.add_argument("prompt", type=str, nargs="?", default="The universe is", help="Starting text prompt")
    parser.add_argument("--checkpoint", "-c", type=str, default=None, help="Path to .pt checkpoint file")
    parser.add_argument("--tokens", "-t", type=int, default=60, help="Number of new tokens to generate")
    parser.add_argument("--temp", type=float, default=0.8, help="Creativity temperature (0.1 = robotic, 1.0 = creative)")
    parser.add_argument("--top_k", type=int, default=50, help="Top-K sampling cutoff")

    args = parser.parse_args()
    generate_text(
        prompt=args.prompt,
        checkpoint_path=args.checkpoint,
        max_new_tokens=args.tokens,
        temperature=args.temp,
        top_k=args.top_k
    )
