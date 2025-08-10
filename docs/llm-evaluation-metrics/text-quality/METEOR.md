# METEOR

## Overview

**METEOR** (Metric for Evaluation of Translation with Explicit ORdering) is a **reference-based** evaluation metric for assessing the quality of generated text, such as translations, summarizations, or responses from large language models (LLMs).

It was developed to address limitations of **BLEU**, particularly its insensitivity to recall, synonymy, and word ordering. METEOR computes a score based on **harmonized precision and recall**, enriched with stemming, synonym matching, and fragmentation penalties.

While originally designed for **machine translation**, METEOR has become widely adopted for **NLG evaluation**, including summarization, dialogue generation, and LLM output benchmarking.

## Key Characteristics

1. **Precision–Recall Balance**
   Unlike BLEU (which focuses on precision), METEOR uses a **weighted harmonic mean** (F-score) of precision and recall, giving recall slightly more weight (default β = 9).

2. **Linguistic Flexibility**

   - **Exact matches**: identical tokens.
   - **Stem matches**: match after stemming.
   - **Synonym matches**: match using synonym dictionaries (e.g., WordNet).
   - **Paraphrase matches**: optional, for multi-word equivalences.

3. **Fragmentation Penalty**
   Measures how well matched words are in the same order. More fragmented alignments yield higher penalties.

4. **Sentence-Level Evaluation**
   METEOR operates at the sentence level before averaging, which makes it sensitive to individual sentence errors.

## How It Works

### 1. Tokenization

Both reference and candidate sentences are tokenized into words.

Example:

```
Candidate: "The cat sat on the mat."
Reference: "A cat was sitting on a mat."
```

### 2. Matching

Matches are computed in the following priority:

1. **Exact match** → same surface form.
2. **Stem match** → match after stemming (e.g., "sitting" → "sit").
3. **Synonym match** → match based on synonym sets (e.g., "cat" ↔ "feline").
4. **Paraphrase match** → optional (e.g., "on top of" ↔ "on").

### 3. Precision & Recall

- **Precision (P)** = matched words / candidate words.
- **Recall (R)** = matched words / reference words.

Example:
Matched = 5 words, Candidate length = 6, Reference length = 7

```
P = 5 / 6  = 0.833
R = 5 / 7  ≈ 0.714
```

### 4. F-Score

METEOR uses a weighted harmonic mean of precision and recall:

$$
F_{mean} = \frac{10 \cdot P \cdot R}{R + 9P}
$$

Recall is weighted higher than precision by default (β = 9).

### 5. Fragmentation Penalty

A penalty is applied based on **chunks** — contiguous matched words in both sentences.
More chunks → higher penalty.

Penalty formula:

$$
Penalty = \gamma \left( \frac{\text{chunks}}{\text{matches}} \right)^{\theta}
$$

Typical parameters: γ = 0.5, θ = 3.

### 6. Final METEOR Score

$$
\text{METEOR} = (1 - \text{Penalty}) \times F_{mean}
$$

## Example Calculation

Candidate:

```
the cat is on the mat
```

Reference:

```
there is a cat on the mat
```

- Matches: `the`(exact), `cat`(exact), `is`(exact), `on`(exact), `the`(exact), `mat`(exact) → 6 matches.
- Candidate length = 6, Reference length = 7
- **P** = 6/6 = 1.0, **R** = 6/7 ≈ 0.857
- **F_mean** = 0.882
- Chunks = 2 → Penalty = 0.5 \* (2/6)^3 ≈ 0.0185
- Final METEOR = 0.882 × (1 - 0.0185) ≈ **0.866**

## Advantages

- **Better correlation with human judgment** than BLEU, especially for sentence-level evaluation.
- Handles **synonyms, stemming, and paraphrases**.
- More **recall-aware**, penalizing omissions.

## Limitations

- **Language dependency**: Needs stemmer and synonym resources (e.g., WordNet).
- Slower than BLEU for large-scale evaluation.
- Not as widely used in modern LLM benchmarking (BERTScore, BLEURT often preferred).

## Usage in LLM Evaluation

For **LLM-generated outputs**, METEOR can:

- Measure **semantic similarity** with human-written references.
- Complement **BLEU** by incorporating recall and synonyms.
- Provide **fine-grained sentence-level insights**.

However:

- For open-ended tasks with multiple valid responses, **reference-free** metrics (e.g., BERTScore, MAUVE, QLORA-based scoring) may be more suitable.

## Python Example

```python
import evaluate

meteor = evaluate.load("meteor")

predictions = ["the cat is on the mat"]
references = ["there is a cat on the mat"]

results = meteor.compute(predictions=predictions, references=references)
print(results)  # {'meteor': 0.866...}
```

## References

- Banerjee, Satanjeev, and Alon Lavie. ["METEOR: An automatic metric for MT evaluation with improved correlation with human judgments."](https://aclanthology.org/W05-0909/) ACL Workshop, 2005.
- HuggingFace Evaluate: [meteor](https://huggingface.co/spaces/evaluate-metric/meteor)
