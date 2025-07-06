# Encode the full Tiny Shakespeare dataset into token IDs
dataset = torch.tensor(encode(data), dtype=torch.int32).to(ModelArgs.device)
print(f"Dataset shape: {dataset.shape}")


def get_dataset_batch(data, split: str, args: ModelArgs):
    seq_len = args.max_seq_len
    batch_size = args.max_batch_size
    device = args.device

    # Partition dataset
    n = len(data)
    split_indices = {
        "train": (0, int(0.8 * n)),
        "val": (int(0.8 * n), int(0.9 * n)),
        "test": (int(0.9 * n), n),
    }
    start, end = split_indices[split]
    batch_data = data[start:end]

    # Sample random starting indices
    ix = torch.randint(0, len(batch_data) - seq_len - 1, (batch_size,), device=device)

    # Compose input (x) and target (y) batches with BOS/EOS if desired
    x = torch.stack([torch.cat([token_bos, batch_data[i:i + seq_len - 1]]) for i in ix]).long().to(device)
    y = torch.stack([torch.cat([batch_data[i + 1:i + seq_len], token_eos]) for i in ix]).long().to(device)

    return x, y


@torch.no_grad()
def evaluate_loss(model, args: ModelArgs):
    """
    Evaluate average loss for both training and validation splits.
    Returns:
        dict: {'train': float, 'val': float}
    """
    model.eval()
    losses = {"train": [], "val": []}
    for split in ["train", "val"]:
        for _ in range(10):
            xb, yb = get_dataset_batch(dataset, split, args)
            _, loss = model(x=xb, targets=yb)
            losses[split].append(loss.item())
    model.train()
    return {k: np.mean(v) for k, v in losses.items()}


def train(model, optimizer, args: ModelArgs):
    epochs = args.epochs
    log_interval = args.log_interval
    device = args.device
    losses = []
    start_time = time.time()

    for epoch in range(epochs):
        optimizer.zero_grad()
        xs, ys = get_dataset_batch(dataset, 'train', args)
        logits, loss = model(x=xs.to(device), targets=ys.to(device))
        loss.backward()
        optimizer.step()

        if epoch % log_interval == 0:
            elapsed = time.time() - start_time
            eval_loss = evaluate_loss(model, args)
            losses.append(eval_loss)
            print(f"Epoch {epoch:>4} | Val loss: {eval_loss['val']:.4f} | Time: {elapsed:.2f}s")
            start_time = time.time()

    print(f"Final validation loss: {losses[-1]['val']:.4f}")
    pd.DataFrame(losses).plot()


model = Transformer(ModelArgs).to(ModelArgs.device)
optimizer = torch.optim.Adam(model.parameters())
train(model, optimizer, ModelArgs)
