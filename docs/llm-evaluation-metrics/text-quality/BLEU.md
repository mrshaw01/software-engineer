# BLEU — A Practical, No-Nonsense README (for MT & LLM Evaluation)

> **TL;DR**
>
> - **What it is:** BLEU (Papineni et al., 2002) measures **n-gram overlap** between system output and human references, combining **clipped n-gram precision** with a **brevity penalty**.
> - **Great for:** Machine translation and tasks where _lexical fidelity to references_ matters.
> - **Weak for:** Open-ended LLM outputs (summaries, QA, instruction following) with many valid phrasings—use alongside semantic metrics and human evals.
> - **How to report:** Use **sacreBLEU** and include the **signature** (tokenizer, case, smoothing).
> - **One-liner:** `sacrebleu.corpus_bleu(sys, [refs], tokenize="13a", smooth_method="exp").score`

## Table of Contents

- [1. What BLEU Measures](#1-what-bleu-measures)
- [2. Formal Definition](#2-formal-definition)
- [3. Worked Mini-Example](#3-worked-mini-example)
- [4. Smoothing (Critical for Sentence-Level)](#4-smoothing-critical-for-sentence-level)
- [5. Corpus vs Sentence BLEU](#5-corpus-vs-sentence-bleu)
- [6. Tokenization, Casing, Effective Order](#6-tokenization-casing-effective-order)
- [7. Reproducibility with sacreBLEU](#7-reproducibility-with-sacrebleu)
- [8. Multiple References](#8-multiple-references)
- [9. Strengths & Limitations](#9-strengths--limitations)
- [10. BLEU in LLM Workflows](#10-bleu-in-llm-workflows)
- [11. Best Practices & Pitfalls](#11-best-practices--pitfalls)
- [12. Quickstart Code](#12-quickstart-code)
- [13. Self-BLEU (Diversity Check)](#13-self-bleu-diversity-check)
- [14. Reporting Template](#14-reporting-template)
- [15. Related Metrics & When to Prefer Them](#15-related-metrics--when-to-prefer-them)
- [16. FAQ](#16-faq)
- [17. Minimal Checklist](#17-minimal-checklist)
- [18. References](#18-references)

## 1. What BLEU Measures

BLEU quantifies **how many of your candidate’s n-grams (1…N, typically 4) appear in the references**, discounting repeats (“**clipping**”), and penalizes **short** candidates (brevity penalty). It is fundamentally a **corpus-level** metric; sentence-level BLEU is unstable without smoothing.

**Intuition**

- **Precision over n-grams:** Did I use the same word sequences as the reference(s)?
- **No recall:** BLEU doesn’t directly reward covering _all_ content—only overlap with what you said.
- **Brevity penalty:** Discourages gaming via ultra-short hypotheses that match frequent n-grams.

## 2. Formal Definition

Let:

- $p_n$: **clipped** n-gram precision for order $n$
- $w_n$: weight for order $n$ (usually $w_n = 1/N$)
- $c$: total candidate length (tokens)
- $r$: effective reference length (closest per segment, summed)

**Clipped precision**

$$
p_n = \frac{\sum_{\text{cand n-gram}} \min(\text{count}_\text{cand},\ \max_{\text{refs}} \text{count}_\text{ref})}{\sum_{\text{cand n-gram}} \text{count}_\text{cand}}
$$

**Brevity penalty (BP)**

$$
\text{BP}=
\begin{cases}
1 & \text{if } c>r \\
e^{(1 - r/c)} & \text{if } c \le r
\end{cases}
$$

**BLEU score**

$$
\text{BLEU} = \text{BP} \cdot \exp\left(\sum_{n=1}^{N} w_n \log p_n\right)
$$

## 3. Worked Mini-Example

Candidate: `the cat is on the mat` (6 tokens)
Reference: `there is a cat on the mat` (7 tokens)

- 1-gram matches (clipped): `the, cat, is, on, mat` → $p_1 = 5/6$
- 2-gram matches: `on the`, `the mat` → $p_2 = 2/5$
- 3-gram matches: `on the mat` → $p_3 = 1/4$
- 4-gram matches: none → $p_4 = 0/3 = 0$

Without smoothing, any $p_n = 0$ makes BLEU **0**.
Brevity penalty: $c=6, r=7 \Rightarrow \text{BP}=\exp(1-7/6) \approx 0.846$.
👉 **Use smoothing** for sentence-level (and short) texts.

## 4. Smoothing (Critical for Sentence-Level)

Common options (names vary by library):

- **Add-k** (e.g., add-1)
- **Exponential** (replace zeros with progressively smaller values)
- **Flooring** (replace zeros with a tiny floor)
- **Chen & Cherry (2014) methods 1–7** (widely implemented in sacreBLEU)

For **corpus-level** BLEU with sufficiently long texts, smoothing often has little effect; for **sentence-level** BLEU, it’s essential.

## 5. Corpus vs Sentence BLEU

- **BLEU was designed for corpus evaluation.**
- **Sentence BLEU** is noisy, frequently zero without smoothing, and can be misleading. If you must report it, apply robust smoothing and state the method.

## 6. Tokenization, Casing, Effective Order

BLEU is **tokenization-sensitive**; choices materially affect scores.

- **Tokenizer:** `13a` (WMT English), `intl`, `zh` (Chinese), or **SPM**-based tokenization (spBLEU) for multilingual use.
- **Case:** cased vs lowercased (`-lc`).
- **Effective order:** on very short sentences, compute up to the **max feasible n** to avoid unfair zeros at higher orders.

**Rule of thumb:** Fix tokenizer, case, and N—use identical settings across systems and **report them**.

## 7. Reproducibility with sacreBLEU

Use **sacreBLEU** to:

- Standardize tokenization and references.
- Emit a **signature** encoding config (tokenizer, case, smoothing, version).
- Prevent silent score drift from custom preprocessing.

**Example signature**
`BLEU+c.mixed+l.en+numrefs.1+smooth.exp+tok.13a+version.2.4.0`

## 8. Multiple References

If multiple references exist, BLEU uses the **max** reference count per n-gram (for clipping) and the **closest** reference length for BP. More references usually **raise** BLEU and reduce variance.

## 9. Strengths & Limitations

**Strengths**

- Simple, fast, widely understood; great for **MT regression** and **A/B** comparisons.
- Reasonable correlation with human MT ratings in **narrow domains** when tokenization is standardized.

**Limitations**

- **Lexical overlap only:** penalizes valid paraphrases/synonyms.
- **No recall:** incomplete outputs can score well.
- **Brevity quirks:** without BP and consistent settings, scores can be gamed.
- **LLM mismatch:** for open-ended tasks (instruction following, summarization, reasoning), BLEU is often **misleading**—pair with **semantic** metrics and human evals.

## 10. BLEU in LLM Workflows

- **Use BLEU for:** MT fine-tuning/serving, constrained paraphrasing, data filtering, and quick sanity checks.
- **Don’t rely solely on BLEU for:** instruction following, multi-sentence summarization, dialog, or reasoning. Prefer **COMET/BLEURT/BERTScore**, reference-free QE, or carefully designed **LLM-as-judge** protocols.
- **Diversity:** **Self-BLEU** (BLEU of each sample vs the rest) — high self-BLEU ⇒ low diversity.

## 11. Best Practices & Pitfalls

**Do**

- Use **sacreBLEU** and record the **signature**.
- Fix **tokenizer** (13a/zh/spm), **case**, **smoothing**, and **N**.
- Report **corpus-level** BLEU (or clearly state sentence-level with smoothing).
- Keep **references fixed** and versioned; share outputs for replication.
- Pair BLEU with **qualitative** inspection and error analyses.

**Avoid**

- Mixing tokenizers/case across experiments.
- Cross-paper comparisons without matching exact settings/datasets.
- Averaging raw sentence BLEU **without** smoothing.
- Treating **absolute** BLEU as quality—focus on **relative deltas** under identical configs.

## 12. Quickstart Code

### Python (sacreBLEU)

```python
# pip install sacrebleu
import sacrebleu

sys_out = [
    "the cat is on the mat",
    "there is a dog outside"
]
refs = [
    "there is a cat on the mat",
    "a dog is outside"
]

# sacreBLEU expects: sys_out: List[str]; refs: List[List[str]] (one list per reference set)
score = sacrebleu.corpus_bleu(sys_out, [refs], tokenize="13a", smooth_method="exp")
print(score.score)          # numeric BLEU
print(score.signature)      # include this in your report for reproducibility
```

### Python (NLTK sentence BLEU with smoothing)

```python
# pip install nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

ref = [["there", "is", "a", "cat", "on", "the", "mat"]]
cand = ["the", "cat", "is", "on", "the", "mat"]
ch = SmoothingFunction()
bleu4 = sentence_bleu(ref, cand, weights=(0.25, 0.25, 0.25, 0.25),
                      smoothing_function=ch.method1)
print(bleu4)
```

### Hugging Face `evaluate` (sacreBLEU under the hood)

```python
# pip install evaluate sacrebleu
import evaluate
bleu = evaluate.load("sacrebleu")

preds = ["the cat is on the mat"]
refs  = [["there is a cat on the mat", "a cat is on the mat"]]
res = bleu.compute(predictions=preds, references=refs, smooth_method="exp", tokenize="13a")
print(res["score"])
```

### CLI (sacreBLEU)

```bash
# refs.txt: one reference sentence per line
# sys.txt:  one system output per line
cat sys.txt | sacrebleu refs.txt -tok 13a -lc
```

## 13. Self-BLEU (Diversity Check)

```python
import sacrebleu

def self_bleu(samples):
    vals = []
    for i, hyp in enumerate(samples):
        refs = samples[:i] + samples[i+1:]
        vals.append(
            sacrebleu.corpus_bleu([hyp], [refs], tokenize="13a", smooth_method="exp").score
        )
    return sum(vals) / len(vals)

samples = [
    "a quick brown fox jumps over the lazy dog",
    "a swift brown fox leaps over a lazy dog",
    "fast brown fox jumps over the lazy dog"
]
print(self_bleu(samples))  # higher -> less diverse
```

## 14. Reporting Template

Copy-paste and fill in:

```
Metric: BLEU-4 (corpus-level)
Tool: sacreBLEU vX.Y.Z
Signature: <paste score.signature here>
Tokenizer: 13a (or zh/spm/intl)
Case: cased or -lc
Smoothing: exp (or methodN)
References: dataset/version, provenance
Test set: e.g., WMT20 en-de newstest
Notes: exact preprocessing (if any), decoding settings, seed
```

## 15. Related Metrics & When to Prefer Them

- **chrF / chrF++** — character n-grams; robust to morphology; strong MT correlation.
- **spBLEU** — BLEU over SentencePiece tokens; good for multilingual settings.
- **METEOR** — stems/synonyms alignment; better recall; slower.
- **ROUGE** — recall-oriented (summarization).
- **BERTScore / BLEURT / COMET** — **semantic** similarity; better for open-ended LLM outputs.
- **Human evals / LLM-as-judge** — the gold standard for instruction-following and reasoning tasks (design carefully!).

## 16. FAQ

**Why is my BLEU lower than a paper’s?**
Different tokenization, casing, smoothing, or test set. Match their **signature**.

**How many references should I use?**
As many as are appropriate/available. More references typically increase BLEU and reduce variance.

**Should I average sentence BLEU?**
Prefer **corpus-level** BLEU. If you must average sentences, specify smoothing and report variance.

**Can I compare BLEU across languages?**
Not meaningfully. Absolute values aren’t comparable across languages/datasets/tokenizers.

## 17. Minimal Checklist

- [ ] Use **sacreBLEU** and record the **signature**
- [ ] Fix **tokenizer** (13a/zh/spm), **case**, **smoothing**, and **N**
- [ ] Prefer **corpus-level** BLEU; if sentence-level, **state smoothing**
- [ ] Keep **references fixed** and **versioned**
- [ ] Share outputs/scripts for reproducibility
- [ ] Pair BLEU with **semantic metrics** + **human evals** for LLMs

## 18. References

- **Papineni, K., Roukos, S., Ward, T., & Zhu, W. J.** (2002). _BLEU: a Method for Automatic Evaluation of Machine Translation._ ACL.
- **Chen, B., & Cherry, C.** (2014). _A Systematic Comparison of Smoothing Techniques for Sentence-Level BLEU._ WMT.
- **Post, M.** (2018). _A Call for Clarity in Reporting BLEU Scores._ WMT — introduces **sacreBLEU**.
