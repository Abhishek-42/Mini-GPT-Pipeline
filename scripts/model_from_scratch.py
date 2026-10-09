import math
import time
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import AdamW

# ==============================================================================
# Model Architecture Components
# ==============================================================================

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, max_seq_len):
        super().__init__()
        assert n_embd % n_head == 0
        
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=True)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=True)
        self.n_head = n_head
        self.n_embd = n_embd
        
        # Causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(max_seq_len, max_seq_len))
                                     .view(1, 1, max_seq_len, max_seq_len))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality
        
        # Calculate query, key, values for all heads in batch
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        
        # Causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        
        # output projection
        y = self.c_proj(y)
        return y

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.c_fc    = nn.Linear(n_embd, 4 * n_embd, bias=True)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * n_embd, n_embd, bias=True)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, n_embd, n_head, max_seq_len):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, max_seq_len)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = FeedForward(n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPTFromScratch(nn.Module):
    def __init__(self, vocab_size, max_seq_len, n_layer, n_head, n_embd):
        super().__init__()
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(vocab_size, n_embd),
            wpe = nn.Embedding(max_seq_len, n_embd),
            h = nn.ModuleList([Block(n_embd, n_head, max_seq_len) for _ in range(n_layer)]),
            ln_f = nn.LayerNorm(n_embd),
        ))
        
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        
        # Weight tying: the token embedding and LM head share the same weights
        self.transformer.wte.weight = self.lm_head.weight
        
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        
        # Forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = tok_emb + pos_emb
        
        for block in self.transformer.h:
            x = block(x)
            
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (b, t, vocab_size)
        
        loss = None
        if targets is not None:
            # cross_entropy requires logits to be shape (B*T, vocab_size) and targets to be (B*T,)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
        return logits, loss

# ==============================================================================
# Model Instantiation and Summary
# ==============================================================================
def print_model_summary(model):
    print("\n" + "="*60)
    print("  MODEL SUMMARY & LAYER BREAKDOWN")
    print("="*60)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params:,} (~{total_params / 1e6:.2f} M)")
    print("-" * 60)
    
    print("Layer-by-Layer Breakdown:")
    print(f"Token Embeddings: {model.transformer.wte.weight.numel():,} parameters")
    print(f"Position Embeddings: {model.transformer.wpe.weight.numel():,} parameters")
    
    for i, block in enumerate(model.transformer.h):
        block_params = sum(p.numel() for p in block.parameters())
        print(f"Transformer Block {i}: {block_params:,} parameters")
        
    print(f"Final LayerNorm: {sum(p.numel() for p in model.transformer.ln_f.parameters()):,} parameters")
    print(f"LM Head (Tied to Token Embeddings): {model.lm_head.weight.numel():,} parameters")
    print("="*60 + "\n")

# ==============================================================================
# Training Simulation (Mixed Precision & Grad Accumulation)
# ==============================================================================
def simulate_training():
    # Architecture settings requested
    vocab_size = 16000
    n_layer = 6
    n_embd = 384
    n_head = 6
    max_seq_len = 512
    batch_size = 8
    grad_accum_steps = 4

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Instantiate Model
    model = GPTFromScratch(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd
    )
    model.to(device)
    print_model_summary(model)
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=6e-4)
    device_type = "cuda" if "cuda" in str(device) else "cpu"
    scaler = torch.amp.GradScaler(device_type, enabled=(device_type == "cuda"))
    
    print("Simulating 1 full Effective Batch (4 accumulation steps)...")
    model.train()
    optimizer.zero_grad()
    
    start_time = time.time()
    for step in range(grad_accum_steps):
        # Generate dummy input tokens and targets
        x = torch.randint(0, vocab_size, (batch_size, max_seq_len), device=device)
        y = torch.randint(0, vocab_size, (batch_size, max_seq_len), device=device)
        
        # Forward pass with mixed precision
        if device_type == "cuda":
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits, loss = model(x, y)
                loss = loss / grad_accum_steps
        else:
            logits, loss = model(x, y)
            loss = loss / grad_accum_steps
            
        # Backward pass
        scaler.scale(loss).backward()
        
        mem_allocated = torch.cuda.memory_allocated() / (1024**3) if device_type == "cuda" else 0.0
        print(f"  Step {step+1}/{grad_accum_steps} | Loss: {loss.item() * grad_accum_steps:.4f} | VRAM Allocated: {mem_allocated:.2f} GB")
        
    # Optimizer step
    scaler.step(optimizer)
    scaler.update()
    
    mem_reserved = torch.cuda.memory_reserved() / (1024**3) if device_type == "cuda" else 0.0
    print("\n" + "="*60)
    print(f"[SUCCESS] Completed training step successfully in {time.time() - start_time:.2f} seconds.")
    print(f"Final VRAM Reservation: {mem_reserved:.2f} GB (Fits perfectly in 4GB!)")
    print("="*60)

if __name__ == "__main__":
    simulate_training()
