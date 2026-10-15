# Attention Is All You Need

A native SwiftUI iPhone app targeting iOS 27, with native Liquid Glass navigation and controls. Touch a token, trace a head, step through the original Transformer, and run a real pretrained model entirely on the phone.

- **Machine:** seven stages of a complete, deterministic tiny encoder–decoder. Actual CPU tensor calculations expose embeddings, sinusoidal positions, Q/K/V, scaled dot-product attention, residual normalization, ReLU FFN, causal masking, cross-attention and tied output probabilities. The weights are explicitly untrained.
- **Paper:** 21 chapter routes and 17 manipulable experiments cover every substantive part of Vaswani et al. (2017), including optimization, regularization, reported results, model variations, parsing and appendix attention maps. Historical measurements are labeled separately from teaching calculations.
- **Sound & touch:** optional weighted tones and soft haptics, a shared volume control, quiet/strong previews and a stoppable five-connection sequence. The user confirmed clean tones and distinct taps on the physical iPhone.
- **Offline assistant:** the shared `OfflineIntelligenceKit` runs Qwen3.5 2B by default, with an optional Qwen3.5 4B choice. The Local model tab starts in Text assistant mode, streams bounded paper explanations, and reports measured token timing. Models load from verified local folders; this path has no network fallback.
- **Measured attention:** Attention trace mode retains Qwen3 4B Instruct 2507 and its selected layer/head capture from real Q/K projections. Its existing 2.27 GB runtime weights and tokenizers remain bundled; the specialist engine can also download pinned, checksum-verified files when built without that bundle. Mode switching drains and unloads the current engine first.

The default build includes model weights under `Resources/Model` (ignored by Git). `Runtime/setup.sh` and `Runtime/capture_trace.py` reproduce the verified download and recorded exhibit. The recorder runs on the Mac for reference validation; the app's live inference runs inside the iPhone process, with no server or cloud API. An explicitly marked recorded trace is available on Simulator, where live MLX inference is disabled.

## Build

Open `AttentionMachine.xcodeproj` in Xcode 27, select `AttentionMachine`, and choose an iPhone on iOS 27 or newer. The project is iPhone-only. Regenerate it with `xcodegen generate` after changing `project.yml`. Dependencies are pinned: MLX Swift 0.32.3, MLX Swift LM 3.32.3, Tokenizers 1.3.0, and Swift Collections 1.5.0 (compatible with the installed Xcode 27 beta standard library).

Use the shared `OfflineIntelligenceKit/Tools/build_hosts.py` staging helper for new assistant builds. It stages the sibling package and clones the verified 2B/4B files into `Resources/Models` outside the synced workspace. Existing specialist weights remain under `Resources/Model`. The empty source `Resources/Models` folder is a resource reference; ordinary Xcode builds require locally supplied verified models to use the text assistant. The installed compiler is Swift 6.4 in Xcode 27.2 beta 2.

## Verification

Run `Tools/check-core.sh` for deterministic numerical checks. The physical-device diagnostics and private machine evidence are omitted from this source snapshot. See `PUBLICATION.md` for dependencies and assets required to run the app.

## Sources

[Original paper](https://arxiv.org/html/1706.03762v7) · [Official Qwen3 4B model card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) · [MLX conversion](https://huggingface.co/mlx-community/Qwen3-4B-Instruct-2507-4bit) · [MLX Swift LM](https://github.com/ml-explore/mlx-swift-lm) · [Apple Liquid Glass guidance](https://developer.apple.com/documentation/swiftui/applying-liquid-glass-to-custom-views)

See `Documentation/PAPER_MAP.md`, `Documentation/MODEL.md`, `Documentation/VERIFICATION.md`, and `Documentation/DIRECTION.md`. `Documentation/Design-Reference.png` is the visual concept produced with built-in Imagegen; the functioning app draws its diagrams and data natively.


## Public snapshot

See [PUBLICATION.md](PUBLICATION.md) for publication exclusions and setup requirements.
