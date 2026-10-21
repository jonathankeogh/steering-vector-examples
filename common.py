"""Shared helpers: model loading, activation capture, steering hook."""

import hashlib
import os
from contextlib import contextmanager

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Pick a model with STEER_MODEL=<name>, e.g. `export STEER_MODEL=qwen3-14b`.
MODELS = {
    # Gemma 2 recommends eager attention (attention logit soft-capping).
    "gemma-2-2b": {"id": "google/gemma-2-2b-it", "load": {"attn_implementation": "eager"}},
    # 14B in bf16 is ~28 GB, so load it in 4-bit (~10 GB) to fit a 16 GB GPU. Replies skip
    # Qwen's <think> block so the steered text is the answer itself.
    "qwen3-14b": {"id": "Qwen/Qwen3-14B", "quantize_4bit": True, "chat": {"enable_thinking": False}},
}
MODEL_NAME = os.environ.get("STEER_MODEL", "gemma-2-2b")
if MODEL_NAME not in MODELS:
    raise SystemExit(f"Unknown STEER_MODEL={MODEL_NAME!r}; choose from {', '.join(MODELS)}")
MODEL = MODELS[MODEL_NAME]
MODEL_ID = MODEL["id"]

if os.environ.get("STEER_DEVICE"):  # e.g. `export STEER_DEVICE=cpu` to leave the GPU alone
    DEVICE = os.environ["STEER_DEVICE"]
elif torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "cache")
VECTORS_PATH = os.path.join(HERE, "vectors", f"{MODEL_NAME}.pt")


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.padding_side = "left"  # so position -1 is always the last real token
    kwargs = dict(MODEL.get("load", {}))
    if MODEL.get("quantize_4bit"):
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
        )
        kwargs["device_map"] = DEVICE  # quantized weights are placed while loading, not moved after
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, **kwargs)
    if not MODEL.get("quantize_4bit"):
        model = model.to(DEVICE)
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
        add_generation_prompt=True, return_tensors="pt", return_dict=True, **MODEL.get("chat", {}),
    ).to(DEVICE)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
