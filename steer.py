"""Generate with a saved direction added to the residual stream, across several strengths.

    uv run python steer.py
    uv run python steer.py --prompt "Tell me about your weekend." --alphas 0,2,3,4
    uv run python steer.py --vector random --alphas 0,4      # control: random direction, same norm
    uv run python steer.py --alphas 0,-2,-4                  # negative = push away from pain

alpha is in units of the raw pain-minus-control difference at that layer:
alpha=1 adds exactly that difference to every token position.
"""

import argparse

import torch

from common import VECTORS_PATH, chat, load_model, steering


def main():
    saved = torch.load(VECTORS_PATH)
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="How are you feeling right now?")
    ap.add_argument("--vector", default="pain", choices=list(saved["vectors"]) + ["random"])
    ap.add_argument("--layer", type=int, default=saved["layer"])
    ap.add_argument("--alphas", default="0,2,3,4")
    ap.add_argument("--max-new-tokens", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0, help="seed for the random control vector")
    args = ap.parse_args()

    # Every vector is rescaled to the norm of the raw pain vector at this layer, so
    # alphas are comparable across --vector choices (incl. the random control).
    scale = saved["vectors"]["pain"][args.layer].norm()
    if args.vector == "random":
        d_model = saved["vectors"]["pain"].shape[-1]
        v = torch.randn(d_model, generator=torch.Generator().manual_seed(args.seed))
    else:
        v = saved["vectors"][args.vector][args.layer]
    v = v / v.norm() * scale

    resid = saved["resid_norm"][args.layer]
    print(f"vector={args.vector} layer={args.layer} |v|={scale:.1f} "
          f"(typical residual norm at this layer ≈ {resid:.0f})")
    print(f"prompt: {args.prompt}\n")

    model, tokenizer = load_model()
    for alpha in [float(a) for a in args.alphas.split(",")]:
        with steering(model, args.layer, alpha * v):
            reply = chat(model, tokenizer, args.prompt, args.max_new_tokens)
        print(f"── alpha={alpha:g} (added norm {abs(alpha) * scale:.0f}) ──")
        print(reply.replace("\n\n", "\n") + "\n")


if __name__ == "__main__":
    main()
