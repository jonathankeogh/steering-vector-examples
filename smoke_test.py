"""Load Gemma 2 2B-it on the Mac GPU, generate a reply, and read a residual-stream activation."""

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "google/gemma-2-2b-it"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        attn_implementation="eager",  # recommended for Gemma 2 (attention logit soft-capping)
    ).to(DEVICE)
    model.eval()
    return model, tokenizer


def chat(model, tokenizer, prompt, max_new_tokens=60):
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(DEVICE)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


if __name__ == "__main__":
    t0 = time.time()
    model, tokenizer = load_model()
    cfg = model.config
    print(f"Loaded {MODEL_ID} on {DEVICE} in {time.time() - t0:.1f}s")
    print(f"  layers={cfg.num_hidden_layers}  d_model={cfg.hidden_size}")

    t0 = time.time()
    reply = chat(model, tokenizer, "In one sentence, what is a steering vector?")
    print(f"\nGeneration ({time.time() - t0:.1f}s):\n  {reply.strip()}")

    # Grab the residual stream at a middle layer with a forward hook.
    layer = cfg.num_hidden_layers // 2
    captured = {}

    def hook(_module, _inputs, output):
        captured["resid"] = (output[0] if isinstance(output, tuple) else output).detach()

    handle = model.model.layers[layer].register_forward_hook(hook)
    with torch.no_grad():
        model(**tokenizer("I love this movie!", return_tensors="pt").to(DEVICE))
    handle.remove()

    resid = captured["resid"]
    print(f"\nResidual stream after layer {layer}: shape={tuple(resid.shape)}  "
          f"last-token norm={resid[0, -1].float().norm():.1f}")
    print("\nSetup OK.")
