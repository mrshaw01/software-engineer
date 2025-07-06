def generate(model,
             prompts: str,
             params: ModelArgs,
             max_gen_len: int = 500,
             temperature: float = 0.6,
             top_p: float = 0.9):
    """
    Generate text sequences from prompts using the trained model.
    """
    bsz = 1  # Inference assumes single prompt per batch
    prompt_tokens = token_bos.tolist() + encode(prompts)
    assert len(prompt_tokens) <= params.max_seq_len, "Prompt exceeds max_seq_len"

    total_len = min(len(prompt_tokens) + max_gen_len, params.max_seq_len)

    # Initialize token buffer with pad tokens
    tokens = torch.full((bsz, total_len), fill_value=token_pad.item(), dtype=torch.long, device=params.device)
    tokens[:, :len(prompt_tokens)] = torch.tensor(prompt_tokens, dtype=torch.long, device=params.device)

    # Mask to track prompt vs generated tokens
    input_text_mask = tokens != token_pad.item()

    prev_pos = 0
    for cur_pos in range(1, total_len):
        with torch.no_grad():
            logits, _ = model(x=tokens[:, prev_pos:cur_pos], start_pos=prev_pos)

        if temperature > 0:
            probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
            next_token = sample_top_p(probs, top_p)
        else:
            next_token = torch.argmax(logits[:, -1], dim=-1)

        next_token = next_token.reshape(-1)
        # Only replace if not a prompt token
        next_token = torch.where(input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token)
        tokens[:, cur_pos] = next_token

        prev_pos = cur_pos

        if next_token == token_eos.item():
            break

    # Decode generated tokens to text
    output_tokens, output_texts = [], []
    for toks in tokens.tolist():
        if token_eos.item() in toks:
            toks = toks[:toks.index(token_eos.item())]
        output_tokens.append(toks)
        output_texts.append(decode(toks))

    return output_tokens, output_texts


def sample_top_p(probs, p):
    """
    Top-p (nucleus) sampling from probability distribution.
    Selects smallest set of tokens with cumulative prob > p, then samples from it.
    """
    probs_sort, prob_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    return torch.gather(prob_idx, -1, next_token)


prompts = "Consider you what services he has done"
output_tokens, output_texts = generate(model, prompts, ModelArgs)

# Remove special BOS token for cleaner output
output_text = output_texts[0].replace("<|begin_of_text|>", "")
print(output_text)
"""
Consider you what services he has done o eretrane
adetranytnn i eey i ade hs rcuh i eey,ad hsatsTns rpae,T
eon o i hseflns o i eee ee hs ote i ocal ersl,Bnnlnface
o i hmr a il nwye ademto nt i a ere
h i ees.
Frm oe o etrane o oregae,alh,t orede i oeral
"""
