"""Shared helpers: model loading, activation capture, steering hook."""

import hashlib
import os
from contextlib import contextmanager

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "google/gemma-2-2b-it"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "cache")
VECTORS_PATH = os.path.join(HERE, "vectors.pt")


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.padding_side = "left"  # so position -1 is always the last real token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, attn_implementation="eager"
    ).to(DEVICE)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def last_token_acts(model, tokenizer, sentences, batch_size=16, use_cache=True):
    """Residual stream at the last token after every layer -> float32 tensor [n_layers, N, d_model].

    Results are cached on disk keyed by the sentence list, so re-runs after editing
    analysis code are instant (editing data.py invalidates the cache automatically).
    """
    key = hashlib.sha1((MODEL_ID + "\n".join(sentences)).encode()).hexdigest()[:16]
    path = os.path.join(CACHE_DIR, f"acts_{key}.pt")
    if use_cache and os.path.exists(path):
        return torch.load(path)

    chunks = []
    for i in range(0, len(sentences), batch_size):
        batch = tokenizer(sentences[i:i + batch_size], return_tensors="pt", padding=True).to(DEVICE)
        hs = model(**batch, output_hidden_states=True).hidden_states  # (embeddings, layer0, ..., layerN-1)
        chunks.append(torch.stack([h[:, -1].float().cpu() for h in hs[1:]]))
    acts = torch.cat(chunks, dim=1)

    if use_cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        torch.save(acts, path)
    return acts


@contextmanager
def steering(model, layer, vector):
    """Add `vector` to the residual stream output of `layer` at every position while active."""
    vec = vector.to(DEVICE, dtype=next(model.parameters()).dtype)

    def hook(_module, _inputs, output):
        if isinstance(output, tuple):
            return (output[0] + vec,) + tuple(output[1:])
        return output + vec

    handle = model.model.layers[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@torch.no_grad()
def chat(model, tokenizer, prompt, max_new_tokens=80):
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True, return_tensors="pt", return_dict=True,
    ).to(DEVICE)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
