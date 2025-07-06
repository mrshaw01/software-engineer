class Transformer(nn.Module):

    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params

        # Token embedding layer
        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)

        # Stack of decoder blocks (e.g. 4 here; 32 in official Llama 3)
        self.layers = nn.ModuleList([TransformerBlock(params) for _ in range(params.n_layers)])

        # Final RMSNorm and output projection
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

    def forward(self, x, start_pos=0, targets=None):
        # x: token IDs [bsz, seq_len]
        h = self.tok_embeddings(x)

        # Determine mode
        inference = targets is None

        # Pass through decoder blocks
        for layer in self.layers:
            h = layer(h, start_pos, inference)

        # Normalize and project to logits
        h = self.norm(h)
        logits = self.output(h).float()

        # Compute loss in training mode
        loss = None
        if not inference:
            loss = F.cross_entropy(logits.view(-1, self.params.vocab_size), targets.view(-1))

        return logits, loss


model = Transformer(ModelArgs).to(ModelArgs.device)
print(model)
"""
Transformer(
  (tok_embeddings): Embedding(68, 512)
  (layers): ModuleList(
    (0-7): 8 x TransformerBlock(
      (attention_norm): RMSNorm()
      (attention): Attention(
        (wq): Linear(in_features=512, out_features=512, bias=False)
        (wk): Linear(in_features=512, out_features=256, bias=False)
        (wv): Linear(in_features=512, out_features=256, bias=False)
        (wo): Linear(in_features=512, out_features=512, bias=False)
      )
      (ff_norm): RMSNorm()
      (feedforward): FeedForward(
        (w1): Linear(in_features=512, out_features=1536, bias=False)
        (w2): Linear(in_features=1536, out_features=512, bias=False)
        (w3): Linear(in_features=512, out_features=1536, bias=False)
      )
    )
  )
  (norm): RMSNorm()
  (output): Linear(in_features=512, out_features=68, bias=False)
)
"""
