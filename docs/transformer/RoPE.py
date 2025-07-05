def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0):
    """
    For each dimension index k (stepping by 2 for pairs) and each position index t:
        freq_k = 1 / theta^{k / dim}
        angle_{t,k} = t * freq_k

    The output is:
        freqs_cis[t, k] = cos(angle_{t,k}) + i * sin(angle_{t,k})

    This is efficiently computed via:
        freqs = outer(t, freq_k)
        freqs_cis = polar(1, freqs)
    """
    device = ModelArgs.device
    freqs = 1.0 / (theta**(torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(t, freqs).to(device)
    return torch.polar(torch.ones_like(freqs), freqs).to(device)


def reshape_for_broadcast(freqs_cis, x):
    """
    Given:
        x.shape = [bsz, seq_len, n_heads, head_dim//2]
        freqs_cis.shape = [seq_len, head_dim//2]

    The function reshapes freqs_cis to:
        [1, seq_len, 1, head_dim//2]
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1]), "The last two dimensions of freqs_cis and x must match."
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    """
    Let x = x_real + i x_imag be the complex representation of input tensors (from interleaved real tensors).
    Let freqs_cis = cos(theta) + i sin(theta) be the precomputed complex rotations.

    Then:
        x_rotated = x * freqs_cis
    """
    device = ModelArgs.device

    # Convert to complex tensors for rotation
    # [bsz, seq_len, n_heads, head_dim/2]
    xq_complex = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2)).to(device)
    xk_complex = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2)).to(device)

    freqs_cis = reshape_for_broadcast(freqs_cis, xq_complex)

    # Apply rotation and convert back to real tensors
    # [bsz, seq_len, n_heads, head_dim]
    xq_rotated = torch.view_as_real(xq_complex * freqs_cis).flatten(3).to(device)
    xk_rotated = torch.view_as_real(xk_complex * freqs_cis).flatten(3).to(device)

    return xq_rotated.type_as(xq), xk_rotated.type_as(xk)


device = ModelArgs.device
head_dim = ModelArgs.dim // ModelArgs.n_heads

wq = nn.Linear(ModelArgs.dim, ModelArgs.n_heads * head_dim, bias=False, device=device)
wk = nn.Linear(ModelArgs.dim, ModelArgs.n_kv_heads * head_dim, bias=False, device=device)

xq = wq(x_norm)
xk = wk(x_norm)

print(f"xq.shape: {xq.shape}")
print(f"xk.shape: {xk.shape}")

xq = xq.view(xq.shape[0], xq.shape[1], ModelArgs.n_heads, head_dim)
xk = xk.view(xk.shape[0], xk.shape[1], ModelArgs.n_kv_heads, head_dim)

print(f"xq reshaped: {xq.shape}")
print(f"xk reshaped: {xk.shape}")

freqs_cis = precompute_freqs_cis(dim=head_dim, seq_len=ModelArgs.max_seq_len)
print(f"freqs_cis.shape: {freqs_cis.shape}")

xq_rotated, xk_rotated = apply_rotary_emb(xq, xk, freqs_cis)

print(f"xq_rotated.shape: {xq_rotated.shape}")
print(f"xk_rotated.shape: {xk_rotated.shape}")
"""
Expected Output:
xq.shape: torch.Size([10, 256, 512])
xk.shape: torch.Size([10, 256, 256])
xq reshaped: torch.Size([10, 256, 8, 64])
xk reshaped: torch.Size([10, 256, 4, 64])
freqs_cis.shape: torch.Size([256, 32])
xq_rotated.shape: torch.Size([10, 256, 8, 64])
xk_rotated.shape: torch.Size([10, 256, 4, 64])
"""
