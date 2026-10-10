"""
Data Preprocessing & Tokenization Pipeline for ~20M LLM
=========================================================
Target:
  - Dataset: FineWeb-Edu (streamed sample)
  - Tokenizer: Byte-Pair Encoding (BPE), 16,000 vocabulary
  - Target Context: 512 tokens
  - DataLoaders: Batch size 8, 90/10 train/validation split
"""

import os
import sys
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from transformers import PreTrainedTokenizerFast
from tqdm import tqdm


import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
TOKENIZER_DIR = PROJECT_ROOT / "models" / "tokenizer"

def load_config():
    with open(PROJECT_ROOT / "configs" / "model_config.yaml", "r") as f:
        return yaml.safe_load(f)

config = load_config()
VOCAB_SIZE = config["model"]["vocab_size"]
CONTEXT_LENGTH = config["model"]["max_position_embeddings"]
BATCH_SIZE = config["training"]["batch_size"]
TARGET_NUM_SAMPLES = 100000  # Scaled up for better language modeling & generalization


def ensure_dirs():
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)


def download_or_stream_fineweb_edu(num_samples=TARGET_NUM_SAMPLES):
    """
    Streams samples from FineWeb-Edu (sample-10BT split) and caches raw text locally.
    """
    raw_text_file = RAW_DATA_DIR / f"fineweb_edu_{num_samples}.txt"
    if raw_text_file.exists() and raw_text_file.stat().st_size > 0:
        print(f"[INFO] Found existing cached text: {raw_text_file}")
        return raw_text_file

    print(f"[INFO] Streaming {num_samples} documents from HuggingFaceFW/fineweb-edu (sample-10BT)...")
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True
    )

    count = 0
    with open(raw_text_file, "w", encoding="utf-8") as f:
        for item in tqdm(dataset, total=num_samples, desc="Streaming FineWeb-Edu"):
            text = item.get("text", "").strip()
            if len(text) > 200:  # Skip tiny fragments
                f.write(text + "\n<|endoftext|>\n")
                count += 1
                if count >= num_samples:
                    break

    print(f"[SUCCESS] Downloaded and cached {count} articles to {raw_text_file}")
    return raw_text_file


def train_bpe_tokenizer(raw_text_file, vocab_size=VOCAB_SIZE):
    """
    Trains a Byte-Level BPE tokenizer (GPT-2 style) with specified vocab size.
    """
    tokenizer_file = TOKENIZER_DIR / "tokenizer.json"
    if tokenizer_file.exists():
        print(f"[INFO] Using existing tokenizer from {tokenizer_file}")
        fast_tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=str(tokenizer_file),
            bos_token="<|endoftext|>",
            eos_token="<|endoftext|>",
            unk_token="<unk>",
            pad_token="<pad>"
        )
        return fast_tokenizer

    print(f"[INFO] Training Byte-Level BPE Tokenizer (vocab_size={vocab_size})...")
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    special_tokens = ["<pad>", "<unk>", "<|endoftext|>"]
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=True
    )

    def text_iterator():
        with open(raw_text_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield line

    tokenizer.train_from_iterator(text_iterator(), trainer=trainer)
    tokenizer.save(str(tokenizer_file))

    # Wrap into Hugging Face fast tokenizer
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_file),
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        unk_token="<unk>",
        pad_token="<pad>"
    )
    fast_tokenizer.save_pretrained(str(TOKENIZER_DIR))
    print(f"[SUCCESS] Tokenizer trained and saved to {TOKENIZER_DIR}")
    return fast_tokenizer


class TextChunkDataset(Dataset):
    """
    Splits token stream into fixed-size context chunks (e.g., 512 tokens).
    Prepares input_ids and labels for causal language modeling.
    """
    def __init__(self, token_ids, max_length=CONTEXT_LENGTH):
        self.max_length = max_length
        # Chunk tokens into sequences of max_length
        num_chunks = len(token_ids) // max_length
        self.chunks = [
            token_ids[i * max_length : (i + 1) * max_length]
            for i in range(num_chunks)
        ]

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        x = torch.tensor(chunk, dtype=torch.long)
        # In causal LM, labels are typically equal to input_ids (shifted in loss calc)
        return {
            "input_ids": x,
            "labels": x.clone()
        }


def prepare_dataloaders(raw_text_file, tokenizer, max_length=CONTEXT_LENGTH, batch_size=BATCH_SIZE):
    """
    Tokenizes raw text, creates fixed chunks, and splits 90/10 into PyTorch DataLoaders.
    """
    print(f"[INFO] Tokenizing text data into {max_length}-token sequences...")
    all_tokens = []
    eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    with open(raw_text_file, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Tokenizing"):
            text = line.strip()
            if text:
                tokens = tokenizer.encode(text)
                all_tokens.extend(tokens)
                all_tokens.append(eos_token_id)

    print(f"[INFO] Total tokens generated: {len(all_tokens):,}")
    full_dataset = TextChunkDataset(all_tokens, max_length=max_length)
    print(f"[INFO] Total 512-token chunks available: {len(full_dataset):,}")

    # 90 / 10 train/validation split
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"[INFO] Train chunks: {len(train_dataset):,} | Validation chunks: {len(val_dataset):,}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader


def test_and_verify(tokenizer, train_loader):
    """
    Verifies tokenization with sample strings, inspects token IDs,
    decodes them back to text, and inspects DataLoader batches.
    """
    print("\n" + "=" * 60)
    print("  VERIFICATION & SANITY CHECKS")
    print("=" * 60)

    # 1. Test encoding/decoding on sample text
    sample_sentence = "The theory of relativity explains how gravity affects spacetime."
    encoded = tokenizer(sample_sentence)
    tokens = encoded["input_ids"]
    decoded = tokenizer.decode(tokens)

    print(f"\n[SAMPLE TEST]")
    print(f"  Original Text : '{sample_sentence}'")
    print(f"  Token IDs     : {tokens}")
    print(f"  Token Count   : {len(tokens)}")
    print(f"  Decoded Text  : '{decoded}'")
    assert decoded.strip() == sample_sentence.strip(), "Decoded text doesn't match original!"
    print("  Status        : [MATCH] Perfect reconstruction!")

    # 2. Inspect a DataLoader batch
    first_batch = next(iter(train_loader))
    input_ids = first_batch["input_ids"]
    labels = first_batch["labels"]

    print(f"\n[DATALOADER BATCH TEST]")
    print(f"  Batch Input Shape : {input_ids.shape} (Expected: [{BATCH_SIZE}, {CONTEXT_LENGTH}])")
    print(f"  Batch Labels Shape: {labels.shape}")
    print(f"  First 10 tokens   : {input_ids[0][:10].tolist()}")
    print(f"  Decoded excerpt   : '{tokenizer.decode(input_ids[0][:20].tolist())}...'")
    print("\n[SUCCESS] Pipeline is fully functional and ready for training!")


def main():
    print("+" * 60)
    print("  FineWeb-Edu Pipeline & BPE Tokenizer Setup")
    print("+" * 60)

    ensure_dirs()
    # 1. Download/Stream
    raw_file = download_or_stream_fineweb_edu(num_samples=TARGET_NUM_SAMPLES)

    # 2. Train Tokenizer
    tokenizer = train_bpe_tokenizer(raw_file, vocab_size=VOCAB_SIZE)

    # 3. Create DataLoaders
    train_loader, val_loader = prepare_dataloaders(raw_file, tokenizer)

    # 4. Verify & Sample Output
    test_and_verify(tokenizer, train_loader)


if __name__ == "__main__":
    main()
