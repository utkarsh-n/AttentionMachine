"""Capture a real Qwen3 attention recording for the offline iPhone exhibit.

Attention probabilities are reconstructed in float32 from the actual normalized,
rotary-positioned Q/K projections. The original fused attention path is untouched.
This script is validation tooling; the iPhone app performs inference on-device.
"""
import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.generate import generate_step

MODEL_ID = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
REVISION = "50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b"
WEIGHT_SHA = "2a73c6c248601ab904e035548abd8e6abb65ea27dcb5f342fb0a8910eb44173f"


class Capture(nn.Module):
    def __init__(self, original, head):
        super().__init__()
        self.original = original
        self.head = head
        self.matrix = None

    def __call__(self, x, mask=None, cache=None):
        if cache is None and x.shape[1] > 1:
            a = self.original
            batch, length, _ = x.shape
            q = a.q_norm(a.q_proj(x).reshape(batch, length, a.n_heads, -1))
            k = a.k_norm(a.k_proj(x).reshape(batch, length, a.n_kv_heads, -1))
            q = a.rope(q.transpose(0, 2, 1, 3))
            k = a.rope(k.transpose(0, 2, 1, 3))
            kv = self.head // (a.n_heads // a.n_kv_heads)
            scores = (q[0, self.head].astype(mx.float32) @
                      k[0, kv].astype(mx.float32).T) * a.scale
            causal = mx.arange(length)[:, None] >= mx.arange(length)[None, :]
            self.matrix = mx.softmax(mx.where(causal, scores, -float("inf")), axis=-1)
        return self.original(x, mask, cache)


def capture(model_directory, output, prompt, layer=17, head=7, max_new=32):
    start = time.perf_counter()
    model, tokenizer = load(str(model_directory))
    formatted = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    token_ids = tokenizer.encode(formatted, add_special_tokens=False)
    if len(token_ids) > 96:
        raise ValueError("Prompt must fit 96 model tokens, including template")
    inputs = mx.array([token_ids])
    baseline = model(inputs)[:, -1, :]
    mx.eval(baseline)
    original = model.layers[layer].self_attn
    probe = Capture(original, head)
    model.layers[layer].self_attn = probe
    try:
        logits = model(inputs)[:, -1, :]
        mx.eval(logits, probe.matrix)
        unchanged_error = float(mx.max(mx.abs(baseline - logits)).item())
        matrix = probe.matrix.tolist()
    finally:
        model.layers[layer].self_attn = original
    probabilities = mx.softmax(logits[0].astype(mx.float32))
    order = mx.argsort(probabilities)[-8:][::-1]
    mx.eval(probabilities, order)
    next_tokens = [{"id": int(i), "text": tokenizer.decode([int(i)]),
                    "probability": float(probabilities[int(i)].item())} for i in order.tolist()]
    generated = []
    for token, _ in generate_step(mx.array(token_ids), model, max_tokens=max_new):
        token_id = int(token)
        if token_id in tokenizer.eos_token_ids:
            break
        generated.append(token_id)
    row_error = max(abs(sum(row) - 1) for row in matrix)
    future_max = max(matrix[i][j] for i in range(len(matrix)) for j in range(i + 1, len(matrix)))
    assert row_error < 1e-5, row_error
    assert future_max == 0, future_max
    assert unchanged_error == 0, unchanged_error
    config = json.loads((model_directory / "config.json").read_text())
    trace = {
        "schemaVersion": 1, "mode": "recorded",
        "capturedAt": datetime.now(timezone.utc).isoformat(), "prompt": prompt,
        "model": {"id": MODEL_ID, "revision": REVISION, "architecture": "qwen3",
                  "quantization": "4-bit affine, group size 64",
                  "layers": config["num_hidden_layers"], "heads": config["num_attention_heads"],
                  "kvHeads": config["num_key_value_heads"], "headDimension": config["head_dim"],
                  "license": "Apache-2.0"},
        "tokens": [{"index": i, "id": token_id, "text": tokenizer.decode([token_id]),
                    "isSpecial": token_id in tokenizer.all_special_ids} for i, token_id in enumerate(token_ids)],
        "layer": layer, "head": head, "attention": matrix, "nextTokens": next_tokens,
        "generation": tokenizer.decode(generated, skip_special_tokens=True),
        "generatedTokenIDs": generated,
        "metrics": {"elapsedSeconds": time.perf_counter() - start,
                    "rowSumMaxError": row_error, "maskedFutureMax": future_max,
                    "instrumentationLogitMaxError": unchanged_error},
        "provenance": {"method": "float32 softmax(QK^T / sqrt(d_k)) reconstructed from actual Qwen3 Q/K after RMSNorm and RoPE, with causal mask; fused model path unchanged",
                       "source": "https://huggingface.co/" + MODEL_ID,
                       "runtime": "MLX Python on Apple M1 Max; prerecorded exhibit",
                       "weightSHA256": WEIGHT_SHA}
    }
    output.write_text(json.dumps(trace, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(output), "tokens": len(token_ids),
                      "generation": trace["generation"], "metrics": trace["metrics"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path(__file__).parent / "models/Qwen3-4B-Instruct-2507-4bit")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent.parent / "Resources/RecordedTrace.json")
    parser.add_argument("--prompt", default="Explain attention in one short sentence.")
    parser.add_argument("--layer", type=int, default=17)
    parser.add_argument("--head", type=int, default=7)
    args = parser.parse_args()
    capture(args.model, args.output, args.prompt, args.layer, args.head)
