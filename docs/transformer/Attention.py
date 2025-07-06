def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Expands the KV heads to match the number of query heads by repeating the KV heads.
    """
    bsz, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (x[:, :, :, None, :].expand(bsz, seq_len, n_kv_heads, n_rep,
                                       head_dim).reshape(bsz, seq_len, n_kv_heads * n_rep, head_dim))


class Attention(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.dim = args.dim  # Embedding dimension
        self.n_heads = args.n_heads  # Query heads
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads  # KV heads
        self.head_dim = self.dim // self.n_heads  # Per-head dimension
        self.n_rep = self.n_heads // self.n_kv_heads  # KV replication factor

        # Linear projections for Q, K, V, and output
        self.wq = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=False, device=device)
        self.wk = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False, device=device)
        self.wv = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False, device=device)
        self.wo = nn.Linear(self.n_heads * self.head_dim, self.dim, bias=False, device=device)

        # KV cache for inference
        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim),
                                   device=args.device)
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim),
                                   device=args.device)

    def forward(self, x: torch.Tensor, start_pos, inference):
        bsz, seq_len, _ = x.shape
        mask = None

        # Project to Q, K, V
        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)

        # Reshape to [bsz, seq_len, heads, head_dim]
        xq = xq.view(bsz, seq_len, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_kv_heads, self.head_dim)

        if inference:
            # Compute RoPE embeddings and apply to Q, K
            freqs_cis = precompute_freqs_cis(dim=self.head_dim, seq_len=self.args.max_seq_len * 2)
            freqs_cis = freqs_cis[start_pos:start_pos + seq_len]
            xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

            # Update KV cache
            self.cache_k = self.cache_k.to(xq)
            self.cache_v = self.cache_v.to(xq)
            self.cache_k[:bsz, start_pos:start_pos + seq_len] = xk
            self.cache_v[:bsz, start_pos:start_pos + seq_len] = xv

            # Retrieve all cached keys/values up to current position
            keys = self.cache_k[:bsz, :start_pos + seq_len]
            values = self.cache_v[:bsz, :start_pos + seq_len]

            # Expand KV heads to match Q heads
            keys = repeat_kv(keys, self.n_rep)
            values = repeat_kv(values, self.n_rep)

        else:
            # Training mode: compute RoPE for full seq_len
            freqs_cis = precompute_freqs_cis(dim=self.head_dim, seq_len=self.args.max_seq_len)
            xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

            keys = repeat_kv(xk, self.n_rep)
            values = repeat_kv(xv, self.n_rep)

            # Causal mask for training
            mask = torch.full((seq_len, seq_len), float("-inf"), device=self.args.device)
            mask = torch.triu(mask, diagonal=1)

        # Transpose to [bsz, heads, seq_len, head_dim] for attention computation
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)

        # Attention output
        output = torch.matmul(scores, values)

        # Merge heads back to [bsz, seq_len, dim]
        output = output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.wo(output)


n_rep = ModelArgs.n_heads // ModelArgs.n_kv_heads
keys = repeat_kv(xk, n_rep)

print(f"xk.shape: {xk.shape}")  # [bsz, seq_len, n_kv_heads, head_dim]
print(f"keys.shape: {keys.shape}")  # After repeating to match n_heads

attention = Attention(ModelArgs)
x_out = attention(x_norm, start_pos=0, inference=False)

print(f"x_out.shape: {x_out.shape}")  # [bsz, seq_len, dim]
"""
xk.shape: torch.Size([10, 256, 4, 64])
keys.shape: torch.Size([10, 256, 8, 64])
x_out.shape: torch.Size([10, 256, 512])
"""
