@dataclass
class ModelArgs:
    dim: int = 512  # Embedding dimension
    n_layers: int = 8  # Number of decoder blocks
    n_heads: int = 8  # Number of query heads
    n_kv_heads: int = 4  # Number of key/value heads
    vocab_size: int = len(vocab)  # Vocabulary size
    multiple_of: int = 256  # Used to calculate the feedforward network dimension
    ffn_dim_multiplier: Optional[float] = None  # Multiplier for feedforward network dimension
    norm_eps: float = 1e-5  # Epsilon for RMSNorm
    rope_theta: float = 10000.0  # Theta for RoPE
    max_batch_size: int = 10  # Maximum batch size
    max_seq_len: int = 256  # Maximum sequence length
    epochs: int = 2500  # Total training iterations
    log_interval: int = 10  # Interval for logging loss and metrics
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'  # Device: CUDA if available, else CPU


class RMSNorm(nn.Module):

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        device = ModelArgs.device
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, device=device))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        # x shape: [batch_size, seq_len, dim]
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


x = torch.randn((ModelArgs.max_batch_size, ModelArgs.max_seq_len, ModelArgs.dim), device=ModelArgs.device)
rms_norm = RMSNorm(dim=ModelArgs.dim)
x_norm = rms_norm(x)

print(f"Shape of x: {x.shape}")
print(f"Shape of x_norm: {x_norm.shape}")
"""
Shape of x: torch.Size([10, 256, 512])
Shape of x_norm: torch.Size([10, 256, 512])
"""
