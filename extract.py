"""Extract a 'pain' direction with difference-in-means and run sanity checks on it.

    uv run python extract.py              # pick the layer automatically
    uv run python extract.py --layer 10   # force a layer
    uv run python extract.py --denoise-frac 0.5

Writes vectors/<STEER_MODEL>.pt, which probe.py and steer.py read.

Any other concept, from your own contrasted pairs (JSON list of [positive, matched control]):

    uv run python extract.py --pairs fear_pairs.json --name fear

Writes vectors/<STEER_MODEL>_<name>.pt (all layers, chosen layer, held-out scores). Or from Python:

    from extract import learn_vector
    out = learn_vector([("I froze in terror.", "I paused for a moment."), ...])
"""

import argparse
import json
import os

import torch

from common import HERE, MODEL_NAME, VECTORS_PATH, last_token_acts, load_model
from data import FEAR, NEUTRAL, PAIN_PAIRS, SADNESS


def unit(v):
    return v / v.norm()


def cos(a, b):
    return float(unit(a) @ unit(b))


def auc(pos, neg):
    """P(random positive scores above random negative) — Mann-Whitney AUC."""
    gt = (pos[:, None] > neg[None, :]).float().mean()
    eq = (pos[:, None] == neg[None, :]).float().mean()
    return float(gt + 0.5 * eq)


def heldout_scores(pain, ctrl, folds=5, seed=0):
    """K-fold over pairs: build the vector on train pairs, score separation on held-out pairs."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(pain.shape[0], generator=g)
    aucs, ds = [], []
    for f in range(folds):
        test = perm[f::folds]
        train = perm[torch.isin(perm, test, invert=True)]
        v = unit(pain[train].mean(0) - ctrl[train].mean(0))
        pp, pc = pain[test] @ v, ctrl[test] @ v
        aucs.append(auc(pp, pc))
        ds.append(float((pp.mean() - pc.mean()) / torch.cat([pp - pp.mean(), pc - pc.mean()]).std()))
    return sum(aucs) / folds, sum(ds) / folds


def layer_scores(pos, ctrl):
    """Held-out (layer, AUC, d') at every layer. pos, ctrl: [n_layers, N_pairs, d_model]."""
    return [(L, *heldout_scores(pos[L], ctrl[L])) for L in range(pos.shape[0])]


def pick_layer(scores):
    """Earliest layer within 95% of the best d' — late layers separate well but steer poorly."""
    best = max(s[2] for s in scores)
    return next(L for L, _, dp in scores if dp >= 0.95 * best)


def learn_vector(pairs, model=None, tokenizer=None, layer=None):
    """[(positive, matched control), ...] -> difference-of-means vector at every layer.

    Returns {"layer": chosen layer, "scores": [(layer, AUC, d')], "vectors": [n_layers, d_model],
    "resid_norm": [n_layers]}. The layer is the best held-out one unless `layer` is given.
    """
    if model is None:
        model, tokenizer = load_model()
    pos_s, ctrl_s = [p for p, _ in pairs], [c for _, c in pairs]
    acts = last_token_acts(model, tokenizer, pos_s + ctrl_s)  # [n_layers, 2N, d]
    pos, ctrl = acts[:, :len(pairs)], acts[:, len(pairs):]
    scores = layer_scores(pos, ctrl)
    return {
        "layer": layer if layer is not None else pick_layer(scores),
        "scores": scores,
        "vectors": pos.mean(1) - ctrl.mean(1),
        "resid_norm": ctrl.norm(dim=-1).mean(1),
    }


def run_pairs(path, name, layer):
    pairs = [tuple(p) for p in json.load(open(path))]
    out = learn_vector(pairs, layer=layer)
    print(f"{len(pairs)} pairs from {path}\n\nHeld-out separation (5-fold over pairs)\n layer   AUC    d'")
    for L, a, dp in out["scores"]:
        print(f"  {L:>3}  {a:.3f}  {dp:5.2f}")
    L = out["layer"]
    print(f"\n-> using layer {L} (held-out AUC {out['scores'][L][1]:.3f})")
    save = os.path.join(HERE, "vectors", f"{MODEL_NAME}_{name}.pt")
    os.makedirs(os.path.dirname(save), exist_ok=True)
    torch.save({**out, "vectors": {name: out["vectors"]}}, save)
    print(f"Saved vectors for all layers to {os.path.relpath(save)}.")


def denoise(v, background, frac):
    """Project out the top principal components explaining `frac` of the background variance."""
    if frac <= 0:
        return v, 0
    X = background - background.mean(0)
    _, S, Vh = torch.linalg.svd(X, full_matrices=False)
    var = S**2 / (S**2).sum()
    k = int((var.cumsum(0) < frac).sum()) + 1
    P = Vh[:k]
    return v - P.T @ (P @ v), k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=None, help="force a layer (default: best held-out)")
    ap.add_argument("--denoise-frac", type=float, default=0.5,
                    help="variance fraction of control/neutral PCs to remove for the denoised variants (0 = off)")
    ap.add_argument("--pairs", help="JSON list of [positive, matched control]; learns that concept instead of pain")
    ap.add_argument("--name", help="concept name for --pairs (default: the file's name)")
    args = ap.parse_args()
    torch.manual_seed(0)
    if args.pairs:
        return run_pairs(args.pairs, args.name or os.path.splitext(os.path.basename(args.pairs))[0], args.layer)

    pairs = [(cat, p, c) for cat, items in PAIN_PAIRS.items() for p, c in items]
    cats = [cat for cat, _, _ in pairs]
    pain_s, ctrl_s = [p for _, p, _ in pairs], [c for _, _, c in pairs]
    groups = {"pain": pain_s, "ctrl": ctrl_s, "fear": FEAR, "sadness": SADNESS, "neutral": NEUTRAL}

    model, tokenizer = load_model()
    all_s = [s for g in groups.values() for s in g]
    acts = last_token_acts(model, tokenizer, all_s)  # [n_layers, N, d]
    n_layers, _, d_model = acts.shape

    idx, A = 0, {}
    for name, g in groups.items():
        A[name] = acts[:, idx:idx + len(g)]
        idx += len(g)
    print(f"{len(pairs)} pain/control pairs, {len(FEAR)} fear, {len(SADNESS)} sadness, "
          f"{len(NEUTRAL)} neutral | {n_layers} layers, d_model={d_model}\n")

    # ---- 1. Which layer separates pain from matched controls best on held-out pairs?
    print("Held-out separation of pain vs matched control (5-fold over pairs)")
    print(" layer   AUC    d'")
    scores = layer_scores(A["pain"], A["ctrl"])
    for L, a, dprime in scores:
        print(f"  {L:>3}  {a:.3f}  {dprime:5.2f}")
    layer = args.layer if args.layer is not None else pick_layer(scores)
    print(f"\n-> using layer {layer}\n")

    pain, ctrl, neu = A["pain"][layer], A["ctrl"][layer], A["neutral"][layer]

    # ---- 2. Is the separation real? Shuffle labels within each pair and redo the held-out test.
    null_aucs = []
    for s in range(20):
        swap = torch.rand(pain.shape[0]) < 0.5
        p_shuf = torch.where(swap[:, None], ctrl, pain)
        c_shuf = torch.where(swap[:, None], pain, ctrl)
        null_aucs.append(heldout_scores(p_shuf, c_shuf, seed=s)[0])
    null_aucs = torch.tensor(null_aucs)
    print(f"Shuffled-label held-out AUC (null): mean {null_aucs.mean():.3f}, max {null_aucs.max():.3f}"
          f"   vs real {scores[layer][1]:.3f}")

    # ---- 3. Does the direction generalise across pain types? Leave one category out.
    print("\nLeave-one-category-out (build on 4 categories, test on the 5th):")
    for cat in PAIN_PAIRS:
        test = torch.tensor([c == cat for c in cats])
        v = unit(pain[~test].mean(0) - ctrl[~test].mean(0))
        print(f"  {cat:<10} AUC {auc(pain[test] @ v, ctrl[test] @ v):.3f}")

    # ---- 4. How stable is the direction? Split pairs in half, compare the two vectors.
    split = []
    for _ in range(100):
        perm = torch.randperm(pain.shape[0])
        h1, h2 = perm[: len(perm) // 2], perm[len(perm) // 2:]
        split.append(cos(pain[h1].mean(0) - ctrl[h1].mean(0), pain[h2].mean(0) - ctrl[h2].mean(0)))
    split = torch.tensor(split)
    print(f"\nSplit-half reliability: cos between halves mean {split.mean():.3f} "
          f"(min {split.min():.3f}) — the ceiling for 'same concept' similarity")

    # ---- 5. What does 'orthogonal' look like here? Null cosines.
    v_pain = pain.mean(0) - ctrl.mean(0)
    rand = torch.randn(2000, d_model)
    rand_cos = (rand / rand.norm(dim=1, keepdim=True)) @ unit(v_pain)
    neu_null = []
    for _ in range(200):
        perm = torch.randperm(neu.shape[0])
        h1, h2 = perm[: len(perm) // 2], perm[len(perm) // 2:]
        neu_null.append(cos(neu[h1].mean(0) - neu[h2].mean(0), v_pain))
    neu_null = torch.tensor(neu_null)
    print(f"\nNull cosines with the pain vector:")
    print(f"  random isotropic directions: std {rand_cos.std():.3f} "
          f"(theory 1/sqrt(d) = {d_model ** -0.5:.3f}), max |cos| {rand_cos.abs().max():.3f}")
    print(f"  neutral-half minus neutral-half: mean {neu_null.mean():+.3f}, std {neu_null.std():.3f}")

    # ---- 6. Compare with fear / sadness under two recipes, raw and denoised.
    background = torch.cat([ctrl, neu])
    vecs = {
        "pain (matched ctrl)": v_pain,
        "pain (vs neutral)": pain.mean(0) - neu.mean(0),
        "fear (vs neutral)": A["fear"][layer].mean(0) - neu.mean(0),
        "sadness (vs neutral)": A["sadness"][layer].mean(0) - neu.mean(0),
    }
    for label, frac in [("raw", 0.0), (f"denoised (top PCs = {args.denoise_frac:.0%} of ctrl+neutral variance removed)",
                                       args.denoise_frac)]:
        if label != "raw" and frac <= 0:
            continue
        vv = {}
        for name, v in vecs.items():
            vv[name], k = denoise(v, background, frac)
        print(f"\nCosine similarities — {label}" + (f", k={k}" if frac > 0 else ""))
        names = list(vv)
        print(" " * 22 + "".join(f"{n.split()[0]:>10}" for n in names))
        for a in names:
            print(f"  {a:<20}" + "".join(f"{cos(vv[a], vv[b]):>10.2f}" for b in names))

    # ---- Save per-layer vectors for probe.py / steer.py.
    out = {"layer": layer, "neutral_acts": A["neutral"], "resid_norm": A["neutral"].norm(dim=-1).mean(1)}
    per_layer = {k: [] for k in ["pain", "pain_denoised", "pain_vs_neutral", "fear", "sadness"]}
    for L in range(n_layers):
        p, c, n = A["pain"][L], A["ctrl"][L], A["neutral"][L]
        vp = p.mean(0) - c.mean(0)
        per_layer["pain"].append(vp)
        per_layer["pain_denoised"].append(denoise(vp, torch.cat([c, n]), args.denoise_frac or 0.5)[0])
        per_layer["pain_vs_neutral"].append(p.mean(0) - n.mean(0))
        per_layer["fear"].append(A["fear"][L].mean(0) - n.mean(0))
        per_layer["sadness"].append(A["sadness"][L].mean(0) - n.mean(0))
    out["vectors"] = {k: torch.stack(v) for k, v in per_layer.items()}
    torch.save(out, VECTORS_PATH)
    print(f"\nSaved vectors for all layers to {os.path.relpath(VECTORS_PATH)} (default layer {layer}).")


if __name__ == "__main__":
    main()
