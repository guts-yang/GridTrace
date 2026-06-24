# GridTrace Algorithm

> A formal walkthrough of grid quantization, two-phase retrieval,
> exact unlearning, and joint semantic representation.

## 1. Grid Quantization

### 1.1 Definition

For a vector \( v \in \mathbb{R}^d \) and a granularity parameter
\( \varepsilon > 0 \), the GridTrace quantized vector is:

\[
Q_\varepsilon(v) = \mathrm{round}\!\left(\frac{v}{\varepsilon}\right) \cdot \varepsilon
\]

where `round` is the standard "round half to even" operation applied
componentwise. The quantized vector lies on the grid

\[
G_\varepsilon = \varepsilon \cdot \mathbb{Z}^d.
\]

### 1.2 Why grid quantization (and not K-means / PQ)?

| Property                | K-means       | PQ            | **Grid (GridTrace)** |
|-------------------------|---------------|---------------|------------------|
| Training step required  | yes           | yes           | **no**           |
| Time per vector         | O(K·d)        | O(m·d)        | **O(d)**         |
| Determinism             | random init   | random init   | **exact**        |
| Codebook size           | K · d         | m · 2^b       | **0**            |
| Hyperparameters         | K             | m, b          | **ε**            |
| Distortion              | minimized     | minimized     | bounded by ε/2   |

The key insight: for *routing* (not final similarity), we don't need
the lowest-distortion quantizer. We need a quantizer that is
**O(d)**, **deterministic**, and **stable across machines**. Grid
quantization satisfies all three.

### 1.3 Distortion bound

The error vector satisfies

\[
\| Q_\varepsilon(v) - v \|_\infty \le \frac{\varepsilon}{2},
\]

so

\[
\| Q_\varepsilon(v) - v \|_2 \le \frac{\sqrt{d}}{2}\,\varepsilon.
\]

For `d=512, ε=0.02`, the worst-case L2 distortion is `≈ 0.226`, but
typical real embeddings concentrate on a small sub-grid and the
empirical error is far smaller (cosine drop < 0.01 in our benchmarks).

### 1.4 Quant key

Two vectors land in the same grid cell iff their quantized forms are
componentwise equal. The quant key is a SHA256 of the quantized
vector's bytes (with `precision=6` rounding to absorb floating-point
noise):

```python
quant_key = SHA256(round(Q_ε(v), 6).tobytes()).hexdigest()
```

## 2. Two-Phase Retrieval

### 2.1 Setup

Let
- \( A \in \mathbb{R}^{|A| \times d} \) be the anchor matrix
- \( E \in \mathbb{R}^{|E| \times d} \) be the entry embedding matrix
- \( q \in \mathbb{R}^d \) be the query vector

We precompute the L2-normalized versions \( \hat{A}, \hat{E}, \hat{q} \)
so that cosine reduces to a dot product.

### 2.2 L1 — Anchor routing

\[
\mathcal{A}^*(q) = \mathrm{argTopK}_{a \in A} \ \hat{a} \cdot \hat{q}
\]

Cost: \( O(|A| \cdot d) \). For pgvector this is a single index scan
with the `<=>` operator. For SQLite/Memory this is a single
`O(|A| d)` numpy matmul.

The L1 hit set is \( \mathcal{A}^* \) with \( |\mathcal{A}^*| = K \)
(default `K=8`).

### 2.3 L2 — Full-precision rerank

Gather all entries that point at any anchor in \( \mathcal{A}^* \):

\[
\mathcal{C}(q) = \bigcup_{a \in \mathcal{A}^*} \{ e \in E : \mathrm{anchor}(e) = a \}
\]

Then re-rank by full-precision cosine:

\[
\mathrm{score}(e) = \hat{e} \cdot \hat{q}, \quad e \in \mathcal{C}(q)
\]

Filter by `score >= threshold` (default `threshold=0.65`), then return
the top `RAG_TOP_K` (default 3) entries.

### 2.4 Complexity

| Phase | Operation            | Complexity       |
|-------|----------------------|------------------|
| L1    | Anchor cosine Top-K  | \( O(|A| d) \)   |
| L2    | Gather + rerank      | \( O(|\mathcal{C}| d) \) |
| Total | Per query            | \( O((|A| + |\mathcal{C}|) d) \) |

In production \( |A| \ll |E| \) (≈1/10 by default), so the L1 phase
provides a 10× speedup on the cosine-heavy part. The L2 phase operates
on a small candidate set of `O(K · \bar{b})` where `K=8` and
`\bar{b}` is the average entries-per-anchor.

## 3. Exact Unlearning

### 3.1 The problem

GDPR / right-to-be-forgotten requires that deleted content actually
disappear from the system. Approximate unlearning (e.g. "fine-tune to
forget") is not enough when the underlying index still contains the
deleted data.

### 3.2 The GridTrace solution

GridTrace achieves **exact unlearning** with `O(1)` overhead per
document by exploiting the fact that anchors are **identifiers**, not
data:

```
unlearn(doc_id):
    1. entries = SELECT * FROM kb_entries WHERE doc_id = :d
    2. DELETE FROM kb_entries WHERE doc_id = :d
    3. anchors_to_check = {e.anchor_id for e in entries}
    4. for a in anchors_to_check:
           if COUNT(kb_entries WHERE anchor_id = a) == 0:
               DELETE FROM kb_anchors WHERE id = a
    5. INSERT INTO kb_unlearn_log
```

### 3.3 Why it works

- An anchor contains **no payload**, only a quantized vector and a
  count. It cannot leak content on its own.
- A delete is idempotent: re-running the operation finds zero
  matching entries.
- Orphan pruning is safe: we only delete anchors whose `ref_count`
  has just dropped to 0 (and verified by a final COUNT).
- The entire flow runs in a single DB transaction with row-level
  locks; concurrent unlearns cannot race.

### 3.4 Comparison

| Method | Time to forget one doc | Correctness |
|---|---|---|
| Delete + rebuild index | O(N) | exact |
| K-means with empty bucket | O(N) | approximate |
| SISA / sharded retraining | O(log N) | approximate |
| **GridTrace (anchor-aware delete)** | **O(1)** | **exact** |

## 4. Joint Semantic Representation

### 4.1 The problem

Storing only the question's embedding loses the rich information in
the solution. Storing only the solution's embedding loses query
alignment. A common workaround is to concatenate `[CLS]` tokens, but
this changes the model's input distribution.

### 4.2 The GridTrace representation

At ingest time, encode the question and the solution independently,
then combine:

\[
\mathrm{joint} = \mathrm{normalize}\!\left( \frac{q_\mathrm{vec} + s_\mathrm{vec}}{2} \right)
\]

where `normalize` is L2-normalization. We store `joint` in
`kb_entries.embedding` and use it for L2 cosine search.

The joint vector preserves both:

- **query alignment** (the `q_vec` component pulls the embedding
  toward the question's region of embedding space), and
- **content richness** (the `s_vec` component adds the
  topic-distinguishing signal from the solution).

### 4.3 Self-check: `match_score`

Independently, we record the intrinsic consistency of the pair:

\[
\mathrm{match\_score} = \cos(q_\mathrm{vec}, s_\mathrm{vec}) \in [-1, 1]
\]

- `match_score ≈ 1` → question and solution are paraphrases /
  answers to the same topic; high-quality pair.
- `match_score ≈ 0` → orthogonal; possibly a noisy or out-of-scope
  pair.
- `match_score < 0` → question and solution contradict; almost
  certainly a data error.

This score is surfaced in `kb_stats.avg_match_score` and in the API
response (`refs[].match_score`) so operators can audit KB quality at
a glance.

## 5. Hyperparameters

| Symbol                | Env var                | Default | Effect of ↑ |
|-----------------------|------------------------|---------|-------------|
| `ε` (epsilon)         | `ANCHOR_QUANT_EPSILON` | 0.02    | More anchors, finer routing, more memory |
| `K` (anchor_top_k)    | `RAG_ANCHOR_TOP_K`     | 8       | Larger L1 candidate pool, slightly slower |
| `τ` (threshold)       | `RAG_SCORE_THRESHOLD`  | 0.65    | Stricter filtering, fewer false positives |
| `R` (top_k final)     | `RAG_TOP_K`            | 3       | More context for LLM, more noise |
| `d` (embedding dim)   | `EMBEDDING_DIM`        | 512     | Higher fidelity, more memory/compute |

A grid search is provided in `scripts/hyperparam_search.py` (planned
M4-4).
