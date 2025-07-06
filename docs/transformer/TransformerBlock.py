class TransformerBlock(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        # RMSNorm before attention
        self.attention_norm = RMSNorm(dim=args.dim, eps=args.norm_eps)
        self.attention = Attention(args)
        # RMSNorm before feedforward
        self.ff_norm = RMSNorm(dim=args.dim, eps=args.norm_eps)
        self.feedforward = FeedForward(args.dim, 4 * args.dim, args.multiple_of, args.ffn_dim_multiplier)

    def forward(self, x, start_pos, inference):
        # Apply attention with residual connection
        h = x + self.attention(self.attention_norm(x), start_pos, inference)
        # Apply feedforward with residual connection
        out = h + self.feedforward(self.ff_norm(h))
        return out  # [bsz, seq_len, dim]


x = torch.randn((ModelArgs.max_batch_size, ModelArgs.max_seq_len, ModelArgs.dim), device=device)
transformer_block = TransformerBlock(ModelArgs)
transformer_block_out = transformer_block(x, start_pos=0, inference=False)
print(f"transformer_block_out.shape: {transformer_block_out.shape}")
"""
transformer_block_out.shape: torch.Size([10, 64, 128])
"""
