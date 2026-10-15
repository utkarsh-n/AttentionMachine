# Attention Is All You Need

A native iPhone instrument for the complete 2017 Transformer paper, built against the installed iOS 27 SDK. The selected visual direction is a dark scientific machine: fine circuit-grid backdrop, warm white type, mint signal paths, coral queries, violet keys, and gold values. Real Liquid Glass belongs on floating controls and navigation; the numerical content has stable, opaque surfaces.

Three connected experiences:

1. **Machine:** a deterministic, untrained small encoder–decoder you can take apart. Tokens are inputs, each head is a lens, and an inspectable forward pass exposes embeddings, positions, projections, scaled scores, probabilities, residual normalization, feed-forward transformations, decoder masks/cross-attention and output distribution. Controls are honest ablation experiments, not reproductions of trained translation quality.
2. **Paper:** every substantive section becomes a manipulable experiment, with original numerical settings, source links and reported results.
3. **Local model:** a downloaded quantized Qwen dense Transformer runs on the physical iPhone through MLX and shows measured attention and generated tokens. Modern Qwen differences (decoder-only, RoPE, RMSNorm, grouped query attention, gated FFN) are named explicitly. Simulator and offline recordings must never be represented as live device inference.

Touch selects a query or an attention cell. Haptic magnitude follows selected attention strength. Optional pitched tones make distributions audible. Native accessible controls offer alternatives to the visual matrix; VoiceOver speaks each selected row. Motion and playback respect accessibility and scene inactivity. All inference after the initial model download stays on device.

Design reference generated with the built-in Imagegen tool; prompt: a straight-on native iPhone Transformer instrument with ink backdrop, luminous Q/K/V wiring, measured attention heatmap, mint/coral/violet/gold encoding, SF typography and native floating Liquid Glass controls.
