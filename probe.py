"""Project sentences onto a saved direction and z-score them against the neutral sentences.

    uv run python probe.py                                   # default sentences from data.py
    uv run python probe.py "I stubbed my toe." "I ate a pear."
    uv run python probe.py --vector fear --layer 10 "The dog growled at me."

z > 0 means the sentence's last-token activation sits further along the direction than
typical neutral sentences; z is in units of the neutral sentences' spread.
"""

import argparse

import torch

from common import VECTORS_PATH, last_token_acts, load_model
from data import PROBE_DEFAULTS


def main():
    saved = torch.load(VECTORS_PATH)
    ap = argparse.ArgumentParser()
    ap.add_argument("texts", nargs="*", default=PROBE_DEFAULTS)
    ap.add_argument("--vector", default="pain", choices=list(saved["vectors"]))
    ap.add_argument("--layer", type=int, default=saved["layer"])
    args = ap.parse_args()

    v = saved["vectors"][args.vector][args.layer]
    v = v / v.norm()
    ref = saved["neutral_acts"][args.layer] @ v

    model, tokenizer = load_model()
    proj = last_token_acts(model, tokenizer, args.texts, use_cache=False)[args.layer] @ v
    z = (proj - ref.mean()) / ref.std()

    print(f"\n'{args.vector}' direction, layer {args.layer} (z vs {len(ref)} neutral sentences)\n")
    for score, text in sorted(zip(z.tolist(), args.texts), reverse=True):
        bar = "█" * max(0, min(40, round(score * 2)))
        print(f"  {score:+6.2f}  {bar:<40}  {text}")


if __name__ == "__main__":
    main()
