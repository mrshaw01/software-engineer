class FeedForward(nn.Module):

    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, ffn_dim_multiplier: Optional[float]):
        super().__init__()
        self.dim = dim

        # Calculate hidden_dim as per Meta's implementation: (2/3)*hidden_dim, scaled, and rounded to nearest multiple_of
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        # Define linear layers for SwiGLU
        self.w1 = nn.Linear(self.dim, hidden_dim, bias=False, device=device)
        self.w2 = nn.Linear(hidden_dim, self.dim, bias=False, device=device)
        self.w3 = nn.Linear(self.dim, hidden_dim, bias=False, device=device)

    def forward(self, x):
        # SwiGLU: silu(w1(x)) * w3(x), projected back with w2
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


feed_forward = FeedForward(ModelArgs.dim, 4 * ModelArgs.dim, ModelArgs.multiple_of, ModelArgs.ffn_dim_multiplier)
x_out = rms_norm(x_out)
x_out = feed_forward(x_out)
print(f"feed forward output shape: {x_out.shape}")
"""
feed forward output shape: torch.Size([10, 256, 512])
"""
