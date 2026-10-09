# ⚡ Rix-17M

**Train a custom 17-million parameter LLM from scratch on a 4GB gaming laptop.**

Rix-17M is a radically simple, zero-bloat PyTorch pipeline built for "hardware-poor" developers. Go from raw data to a chatting AI in just 2 hours on an RTX 3050, complete with auto-resume training and instant Ollama export.

---
## 🔬 Model Specifications & Parameters

| Parameter / Spec | Value |
| :--- | :--- |
| **Total Parameters** | **16.99 Million (~17M)** |
| **Architecture** | GPT-style (Decoder-only Transformer) |
| **Vocabulary Size** | **16,000 tokens** (Byte-Level BPE) |
| **Hidden Dimensions ($d_{\text{model}}$)** | **384** |
| **Transformer Layers** | **6** |
| **Attention Heads** | **6** (64 dimension per head) |
| **Intermediate Size (MLP)** | **1,536** |
| **Context Window (Max Sequence Length)** | **512 tokens** |
| **Activation Function** | `gelu_new` |
| **Dataset** | **FineWeb-Edu** (~11.1 Million tokens) |
| **Train / Validation Split** | **90% / 10%** |
| **Micro-Batch Size** | **8** |
| **Gradient Accumulation Steps** | **4** (Effective Batch Size = 32) |
| **Precision** | **FP16 Mixed Precision** |
| **Optimizer** | **AdamW** ($\text{lr} = 6.0 \times 10^{-4}$, Cosine Schedule) |
| **Peak VRAM Usage** | **~2.1 GB – 2.3 GB** (Safe for 4GB VRAM GPUs) |

---

## 🚀 Setup & Training Guide for New Machines

Follow these steps on any new computer (e.g., an RTX 3050 laptop) to get it running and training:

### 1. Create & Activate Virtual Environment
```powershell
python -m venv venv
.\venv\Scripts\activate
```

### 2. Install PyTorch with CUDA Support
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 3. Install Requirements
```powershell
pip install -r requirements.txt
```

### 4. Prepare Data & Build Tokenizer
Downloads the FineWeb-Edu dataset slice and trains the 16k BPE tokenizer:
```powershell
python scripts/prepare_data.py
```

### 5. Start or Resume Training
* **Start from scratch:**
  ```powershell
  python scripts/train.py
  ```
* **Resume from where you stopped (e.g. after shutting down your laptop):**
  ```powershell
  python scripts/train.py --resume
  ```
  *(Automatically finds your latest checkpoint in `models/checkpoints/` and resumes seamlessly).*

---

## 💬 Test / Generate Prompts

### Test Before Training (Random Weights)
```powershell
python scripts/generate.py "The universe is"
```

### Test After Training (From Saved Checkpoint)
```powershell
python scripts/generate.py --checkpoint models/checkpoints/final_model.pt "The universe is"
```
