# Real-model validation tools

The iPhone app performs all live tokenization, inference, attention inspection, and generation on the iPhone with MLX Swift. It does not call a model server. This folder is optional Mac validation tooling and the local source for the bundled weights.

The model is `mlx-community/Qwen3-4B-Instruct-2507-4bit`, pinned to revision `50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b`. Twelve repository files were downloaded into `models/Qwen3-4B-Instruct-2507-4bit`, totaling 2,278,970,666 bytes. `model-manifest.json` records SHA-256 for every downloaded file. The weights and isolated `.venv` are ignored by Git.

On an Apple Silicon Mac with `uv`, run:

```zsh
zsh Runtime/setup.sh
```

To regenerate the real offline exhibit from already-downloaded weights:

```zsh
Runtime/.venv/bin/python Runtime/capture_trace.py \
  --prompt "Explain transformer attention in one short sentence."
```

The capture checks that each attention row sums to 1, future tokens have zero weight, and instrumenting Q/K does not change model logits. It emits the measured metrics and saves `Resources/RecordedTrace.json`. That file is a recording of an actual pretrained model forward pass on this Mac, and the app must label it as recorded. Changing the prompt, layer, or head requires a new inference run.

The current recording uses zero-based layer 17 and head 7. `attention[row][column]` means how much the query token at `row` attends to the key token at `column`. Tokens include the exact chat-template control tokens. Next-token probabilities are the raw vocabulary softmax, before sampling. Generation uses greedy decoding with a short 32-token cap for a reproducible exhibit.

For app bundling, use the folder name `Qwen3-4B-Instruct-2507-4bit` at bundle root or under `Model/`. Only `config.json`, `tokenizer_config.json`, `tokenizer.json`, and `model.safetensors` are required by the native engine. Do not package the hidden Hugging Face download cache.

The native engine can instead download those four immutable public files directly to Application Support. Completed files persist between attempts; cancelling an in-progress file restarts that file next time. SHA-256 checks cover both weights and tokenizer data. The app excludes its downloaded model directory from iCloud backup.
