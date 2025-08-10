# BERTScore — A Practical, Engineer-Focused Guide

> A comprehensive README for using, tuning, and interpreting BERTScore in modern LLM evaluation.

## TL;DR

BERTScore compares generated text to a reference by aligning **contextual token embeddings** (from models like RoBERTa/DeBERTa/XLM-R) and computing **soft precision/recall/F1** via **cosine similarity**. It correlates better with human judgments than n-gram metrics (BLEU/ROUGE), especially for paraphrases. Use **DeBERTa-v3-large** (English) or **XLM-R-large** (multilingual), enable **IDF weighting**, and **rescale with baseline** when reporting. Keep sequences ≤ model max length or chunk them. BERTScore is **semantic**, not factual—pair it with task-specific checks for hallucinations, formatting, or exactness.

## What problem does BERTScore solve?

Classic n-gram metrics reward exact word overlap and miss legitimate paraphrases. BERTScore uses **pretrained language models** to embed each token in context, then measures **semantic alignment** between hypothesis (model output) and reference(s). It’s robust to synonymy and phrasing variation—crucial for summarization, dialogue, translation, captioning, and instruction following.

## How it works (math without pain)

Let hypothesis tokens $c_1,\dots,c_n$ and reference tokens $r_1,\dots,r_m$ be embedded with a contextual model. Define cosine similarities

$$
s_{ij} = \cos\big(e(c_j), e(r_i)\big).
$$

Compute **precision** and **recall** using best matches:

$$
P = \frac{1}{n}\sum_{j=1}^{n} \max_{i} s_{ij}, \quad
R = \frac{1}{m}\sum_{i=1}^{m} \max_{j} s_{ij}, \quad
F_1 = \frac{2PR}{P+R+\varepsilon}.
$$

**IDF weighting (recommended):** weight rarer tokens higher. Replace uniform averages with weighted sums, e.g. $P=\sum_j v_j \max_i s_{ij}$ where $v_j$ is normalized IDF for $c_j$; similarly for $R$ with reference weights.

**Baseline rescaling (recommended for reporting):** subtract a model/language-specific baseline so scores are more comparable across runs:

$$
\hat{F}_1 = F_1 - b_{\text{model,lang}}.
$$

## Choosing the embedding model

- **English (quality first):** `microsoft/deberta-v3-large` or `microsoft/deberta-xlarge-mnli`
  (Great semantic sensitivity; needs more VRAM.)
- **English (balanced):** `roberta-large`
- **Multilingual:** `xlm-roberta-large` (broad coverage; good default)
- **Resource-constrained:** `bert-base-uncased` (acceptable for smoke tests; lower correlation)

Heuristics:

- Short, well-formed text → RoBERTa/DeBERTa.
- Cross-lingual or mixed language → XLM-R.
- Keep the same model across experiments for comparability.

## Practical usage

### Quickstart (Python)

```python
pip install bert-score torch transformers
```

```python
from bert_score import score

cands = ["The quick brown fox jumps over the lazy dog."]
refs  = ["A fast brown fox leaped over a lazy dog."]

P, R, F1 = score(
    cands, refs,
    model_type="microsoft/deberta-v3-large",
    lang="en",
    idf=True,
    rescale_with_baseline=True,
    batch_size=64,
    device="cuda:0"  # or "cpu"
)

print(float(F1.mean()))
```

### With `evaluate` (Hugging Face)

```python
pip install evaluate bert-score
```

```python
import evaluate
bertscore = evaluate.load("bertscore")
res = bertscore.compute(
    predictions=cands,
    references=refs,
    model_type="xlm-roberta-large",
    lang="en",
    idf=True,
    rescale_with_baseline=True,
    device="cuda:0"
)
print(res["f1"])  # list per example
```

### CLI

```bash
bert-score -r refs.txt -c cands.txt \
  -m microsoft/deberta-v3-large -l en --idf --rescale_with_baseline \
  --batch_size 64 --device cuda:0
```

## Handling multiple references

Common practice is to compute per-reference scores and **aggregate** (often **max** or **average**) per example. Max is stricter on “at least one good reference,” average rewards consistency across references. Be explicit in your reporting.

## Interpreting scores

- Range is roughly $[0, 1]$ after rescaling (can be <0 without it).
- **Rules of thumb (English, strong model):**

  - **≥ 0.95:** near-paraphrase
  - **0.90–0.95:** good semantic match with minor differences
  - **0.85–0.90:** acceptable but noticeable drift or omissions
  - **< 0.85:** substantive divergence

- Domain, length, and model choice shift these bands—define your own thresholds empirically on a **human-rated dev set**.

## Strengths vs. other metrics

- **Pros:** semantic matching, paraphrase-aware, less brittle than BLEU/ROUGE; language-agnostic with the right encoder; strong human correlation on many NLG tasks.
- **Cons:** depends on the encoder’s language/domain coverage; blind to factual correctness and structure; quadratic token matching per pair (though batched efficiently).

**When to pair with other checks**

- **Factuality / hallucination:** QAGS, FactCC, token-level entailment, QA-based checks.
- **Exactness / formal constraints (e.g., JSON/SQL):** exact match, schema validation, execution-based metrics.
- **Style/form:** task-specific regex or AST checks, toxicity/formality classifiers.

## Engineering tips & pitfalls

- **Preprocessing:** Don’t stem or lowercase; feed raw text (remove leading/trailing whitespace). The tokenizer handles subwords.
- **Length limits:** Most encoders cap at 512 tokens. For long outputs, **chunk** (sentence/paragraph) and average, or use long-context encoders. Be explicit about your strategy.
- **Batching & memory:** Control with `batch_size`. Similarity matrices are $n \times m$ per pair; long texts increase memory/time.
- **Determinism:** BERTScore itself is deterministic. Pin package versions, model name, and baseline setting in your reports.
- **IDF weighting:** Compute IDF over your **reference corpus** (default behavior in the library). Helps down-weight boilerplate and emphasize content words.
- **Baseline rescaling:** Turn it **on** (`rescale_with_baseline=True`) for cross-experiment comparability; report whether you used it.
- **Multilingual caveats:** Tokenization and encoder quality vary by language. Validate against a small human-rated set before locking thresholds.
- **Numbers, entities, code:** Semantic encoders may treat “\$1.2M” vs “\$12M” as similar. Add **numeric/entity diff checks** or exact-match constraints if these are critical.
- **Gaming the metric:** Repetition or generic paraphrases can inflate scores. Combine with **diversity** metrics (distinct-n) or task-specific guards.

## Recommended defaults (sane, reproducible)

- **Model:** `microsoft/deberta-v3-large` (English) / `xlm-roberta-large` (multilingual)
- **Flags:** `idf=True`, `rescale_with_baseline=True`
- **Batch size:** 32–128 (GPU-dependent)
- **Aggregation:** mean of F1 across examples; for multi-ref, **max** per example (document your choice)

## Reference implementation notes

- **Token alignment:** Soft, via max cosine matches—not hard word alignment.
- **Layers:** Libraries default to a good layer or aggregate internally; if you expose `num_layers`, keep it constant across experiments.
- **Special tokens:** Implementations exclude `[CLS]`, `[SEP]`, etc., from scoring.

## Example: building a robust eval harness

- Compute **BERTScore (F1)** with the defaults above.
- Also compute **Exact Match** (when applicable), **distinct-1/2**, **length**, and a **numeric/entity diff rate**.
- For factual tasks, add a **QA-based factuality check** or an **NLI classifier** (hypothesis → reference entails?).
- Store per-example metrics; monitor **mean** and **distribution** (P5/P50/P95) across test sets.

## FAQs

**Q: Should I use precision, recall, or F1?**
A: Report **F1** by default. If your task penalizes omissions more than additions (summarization), also report **recall**.

**Q: Why do my scores differ from another paper?**
A: Check model type, language, IDF, baseline rescaling, version pins, and multi-reference aggregation. Any of these will shift results.

**Q: Can I compare scores across languages or domains?**
A: Only with the **same encoder** and **rescaled baselines**—and even then, prefer **relative deltas** within the same setting.

## Citation

- Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q. Weinberger, Yoav Artzi. **“BERTScore: Evaluating Text Generation with BERT.”** ICLR 2020.
