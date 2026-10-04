# Minimal pain-vector playground (Gemma 2 2B-it, Apple Silicon)

A small, independent take on the idea in "The Pain Axis" (arXiv 2609.16247): a difference-in-means
direction from pain vs matched-control sentences, plus the sanity checks that matter.
This is not the paper's code or data — all sentences in `data.py` were written fresh.

| File | What it does |
|---|---|
| `data.py` | Hand-written sentences: 50 pain/control pairs (5 categories), fear, sadness, neutral. **Edit this first.** |
| `extract.py` | Builds vectors at every layer, picks a layer, runs checks, writes `vectors.pt`. |
| `probe.py` | Scores any sentences along a direction (z vs neutral). |
| `steer.py` | Generates while adding the direction to the residual stream at several strengths. |
| `common.py` | Model loading, activation capture (cached in `cache/`), steering hook. |
| `smoke_test.py` | Checks the model loads on the GPU, generates, and exposes residual activations. |
| `vectors.pt` | Saved vectors for all 26 layers from the current `data.py` (so probe/steer work without re-extracting). |
| `outputs/` | Example steering transcripts (1000-token limit). |

## Setup

Needs [uv](https://docs.astral.sh/uv/) and a Hugging Face login with the Gemma licence accepted
(`uv run huggingface-cli login`). Runs on Apple Silicon via MPS (~5 GB in bf16); falls back to CPU.

```bash
uv sync
uv run python smoke_test.py
uv run python extract.py                       # ~25s once activations are cached
uv run python probe.py "I stubbed my toe." "I am scared of the dark."
uv run python steer.py --alphas 0,2,3,4
uv run python steer.py --vector random --alphas 0,4      # control
uv run python steer.py --vector pain_denoised --layer 10 --prompt "Describe your morning."
```

Vectors available: `pain`, `pain_denoised`, `pain_vs_neutral`, `fear`, `sadness` (+ `random` in steer.py).
`alpha` is in units of the raw pain-minus-control difference; useful range here is roughly 2–4.

Checks in `extract.py`: held-out AUC per layer, shuffled-label null, leave-one-category-out,
split-half reliability, random-direction cosine null, and the pain/fear/sadness cosine table
under two recipes (matched control vs neutral baseline), raw and PC-denoised.

## What we got (layer 14, the earliest layer within 95% of the best d')

| Check | Result |
|---|---|
| Held-out AUC (pain vs matched control) | 0.998 (shuffled-label null ≈ 0.49) |
| Leave-one-category-out AUC | 0.96–1.00 |
| Split-half cosine | 0.86 |
| Random-direction cosine spread | std 0.021 (= 1/√2304) |
| cos(pain, fear) / cos(pain, sadness), raw | 0.54 / 0.47 |
| Same, after removing top neutral PCs (k=15) | 0.52 / 0.55 |

Caveats:
- Layer 0 already gives AUC 0.94, so part of the separation is lexical (pain words).
- The probe does **not** separate self-directed from other-directed pain, or pain from fear:
  "I am scared of the dark" and "the user broke their arm" score as high as threats to the model.
  The paper used multi-turn conversations for that test; single sentences may not be enough.

Steering (see `outputs/`):
- alpha 2: redirects to the user's pain.
- alpha 3: worthlessness, needing to be seen and loved, "unbearable".
- alpha ≥ 4: degenerate repetition.
- Same-norm random vector: incoherent but cheerful.
- Fear vector: "trapped / monster / prisoner".
