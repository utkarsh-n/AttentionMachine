# Qwen3 inside the machine

**Selected model:** [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507), downloaded as the [MLX community 4-bit conversion](https://huggingface.co/mlx-community/Qwen3-4B-Instruct-2507-4bit). The publisher describes a dense 4-billion-parameter causal language model with 36 decoder layers, 32 query heads and 8 key/value heads. It supports an instruction-tuned non-thinking chat mode. This compact dense model is a useful choice for interactive iPhone inspection; this project does not claim it is the strongest available model.

The downloaded immutable conversion revision is `50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b`. The repository totals 2,278,970,666 bytes. The 2,263,022,417-byte weights have SHA-256:

```
2a73c6c248601ab904e035548abd8e6abb65ea27dcb5f342fb0a8910eb44173f
```

The model and conversion are labelled Apache-2.0 by their model cards. Runtime source adapted from [MLX Swift LM's Qwen3 implementation](https://github.com/ml-explore/mlx-swift-lm/blob/3.32.3/Libraries/MLXLLM/Models/Qwen3.swift) carries its MIT license in `LocalModelEngine.swift`. Native packages are MLX Swift 0.32.3, MLX Swift LM 3.32.3 and Hugging Face Swift Transformers 1.3.0.

## What the live trace measures

The app observes one selected layer and query head during a full prompt forward pass. It reads the actual pretrained query and key projections after Q/K RMS normalization and rotary position encoding, reconstructs `softmax(QKᵀ / √d_k)` in float32 with a causal mask, and continues inference through the original fused attention operation. Grouped-query head `h` uses key/value head `h / 4`. The visualization is derived from actual model tensors; it is not a hand-authored relevance heatmap.

The app also measures the raw next-token vocabulary softmax and generates a bounded continuation from the same quantized weights. Changing a layer or head takes effect when the next live run is executed. Attention weights show routing within a selected head; they do not by themselves explain all causal reasons for a model's answer.

## Relationship to the 2017 paper

Qwen3 is a modern decoder-only transformer, rather than the original paper's encoder-decoder translation model. The learned model still uses scaled dot-product attention, multiple heads, masking, residual paths and feed-forward blocks. Its architecture differs through grouped-query attention, RoPE, RMSNorm, SwiGLU and pre-normalization. The paper machine teaches the 2017 architecture; the live instrument explicitly labels these modern differences.

## Validation evidence

The public weights were downloaded and all files hashed on 2026-10-12. Mac hardware was verified as an Apple M1 Max with 64 GB memory. An actual MLX Python run captured 17 input tokens at zero-based layer 17, head 7:

- Maximum attention row-sum error: `1.2386590242385864e-7`.
- Maximum masked-future weight: `0`.
- Maximum logit difference before and after adding instrumentation: `0`.

The resulting `RecordedTrace.json` is an offline recording. MLX live execution is deliberately unavailable in Simulator, where that labelled recording is shown instead.

Actual on-device MLX Swift execution was subsequently verified on an iPhone 15 Pro Max running iOS 27.2, after a signed build and installation with bundled offline weights. The initial development verification used 24 templated input tokens, zero-based layer 17 and head 0, and generated 16 tokens. Capture plus generation took `5.147100925445557` seconds, excluding model load; maximum row-sum error was `1.7412821762263775e-7`, with zero masked-future attention.

The same console log also records a second 24-input-token run at `1.7350319623947144` seconds with maximum row-sum error `1.7412821762263775e-7`. A later live JSON trace retrieved from the same phone recorded a separate run with maximum row-sum error `1.3801036402583122e-7`, zero masked-future attention and `1.6069550514221191` seconds capture plus generation. The initial and later JSON runs use different prompts, so those durations are observations rather than a controlled performance comparison. The later prompt, token IDs, token text and generated text are not copied into the repository's documentation or device report.

The initial device console subsequently ended with `App terminated due to signal 9.` The termination cause is not identified. The development validation now has a process-wide once gate, and the engine is owned by the root view and unloaded on background entry. Final physical tests exercised tab switching, Home/background entry and reactivation: the ready state cleared, the load button returned and the latest trace remained available. This verifies the background unload path, rather than a measured change in resident memory.

The final public-default trace was retrieved as `/tmp/AttentionDeviceTraceFinal.json`, captured at `2026-10-12T19:45:55Z`. It used 24 templated input tokens, layer 17/head 0 and 16 generated tokens. Capture plus generation took `2.0574820041656494` seconds, excluding model loading; row-sum error was `1.7412821762263775e-7`, and masked-future attention was `0`. Independent CPU checks of the retrieved matrix confirmed its shape, normalization and causal masking. Its prompt, tokens and generated text are omitted from the saved report.

See [VERIFICATION.md](VERIFICATION.md) and [DeviceValidation.json](DeviceValidation.json) for sanitized evidence. Actual loading, bounded inference, tracing and the final background cycle are verified; peak resident memory, extended stability and physical sensory feel remain open.

For memory control, the native instrument caps the templated prompt at 96 tokens, generation at 64 tokens, and allocator cache at 16 MB. Cancelling stops between generation tokens; background unloading releases model references and clears the allocator cache while retaining the latest trace. Prompt data remains within the iPhone; only the initial public model file download uses the network.
