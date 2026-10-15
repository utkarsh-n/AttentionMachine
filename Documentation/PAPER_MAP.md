# Attention Is All You Need: source and machine map

Source: Vaswani et al., [paper HTML, version 7](https://arxiv.org/html/1706.03762v7) and [PDF, including attention visualizations](https://arxiv.org/pdf/1706.03762). The paper first appeared in 2017; the source inspected for this implementation is the 2023 arXiv revision.

## What the native machine computes

`AttentionCore` is an inspectable CPU encoder–decoder. Every displayed tensor comes from arithmetic in a deterministic forward pass. Its weights are seeded random **untrained initializations**. It is not a translation model, does not learn during use, and is distinct from the separately downloaded pretrained local model.

The default model has 8 features, 32 feed-forward features, 2 heads and 2 layers per stack. Controls permit 1, 2, 4 or 8 heads and 1–6 layers. The original base configuration has 512 features, 2,048 feed-forward features, 8 heads and 6 layers per stack (§3). Smaller matrices make direct inspection practical on an iPhone.

## Paper coverage

| Paper | Native operation or experiment |
| --- | --- |
| Abstract, §§1–2 | Architecture overview and parallel versus sequential processing |
| §3, Figure 1 | Complete source → encoder → memory → decoder → output traversal |
| §3.1 | Each sublayer computes a residual sum followed by feature-wise normalization |
| §3.2.1, Equation 1 | Q/K/V projections, dot products, square-root scaling, masked scores, row softmax and value mixing |
| §3.2.2, Figure 2 | Independent heads, concatenation and output projection |
| §3.2.3 | Encoder self-attention, causal decoder self-attention and encoder-memory cross-attention |
| §3.3, Equation 2 | Position-wise linear → ReLU → linear computation |
| §3.4 | Shared embedding/output table; input embeddings scaled by √d_model |
| §3.5 | Paired sine/cosine positions and position-strength experiment |
| §4, Table 1 | Sequence length, dimension, interaction cost, sequential steps and dependency distance |
| §§5.1–5.2 | Historical datasets, batching and hardware in the paper reader |
| §5.3, Equation 3 | Exact learning-rate schedule, including warmup at 4,000 steps |
| §5.4 | Label-smoothing distribution/cross-entropy experiment; dropout explanation |
| §6.1, Table 2 | Translation results and original beam-search evaluation |
| §6.2, Table 3 | Architecture variation evidence; head and position ablations in the native machine |
| §6.3, Table 4 | Constituency parsing evidence in the reader |
| §7 | Conclusions and connection to a contemporary pretrained model |
| PDF appendix, Figures 3–5 | Attention routing examples and head comparison |

Table 2 reports big-model BLEU scores of **28.4 EN–DE and 41.8 EN–FR**. Version 7's §6.1 prose instead says 41.0 for EN–FR. Preserve this discrepancy when describing the historical result; the app uses the table value and links to the source.

## Deliberate laboratory differences

- Tokenization splits words and punctuation, not BPE or wordpieces. The output vocabulary is a small base vocabulary plus source tokens. Fixing it before reading the decoder prefix prevents future target vocabulary from leaking through the softmax denominator. Unknown target words map to `<unk>`.
- Target input is shifted: `<bos>` followed by the supplied prefix. `expectedTargets` is the prefix followed by `<eos>`.
- LayerNorm uses unit gain and zero bias. Dropout is disabled for this inference demonstration. Neither Adam nor backpropagation is implemented; the training laboratory computes its schedule, target distribution and loss.
- Attention temperature, position strength, disabling scaling/masking and stochastic output sampling are interactive experiments. They are not claims about the original evaluation procedure. The paper evaluates translation with beam search.
- Changing the number of heads changes projection shapes and deterministic initializations. The resulting display illustrates tensor organization, not trained model quality or a controlled benchmark.
- Output projection uses the unscaled embedding table. BOS is suppressed before generation probabilities because it is input-only in this laboratory vocabulary.
- The UI can display `−∞` in score/logit matrices. The softmax returns exact zero for these entries. An entirely masked row returns zeros as a defensive numerical rule.
- Source input is capped at 24 tokens and target prefix at 23 tokens. `inputWasTruncated` exposes this boundary. Temperatures, position strength and structural controls are sanitized before execution.
- Table 1 operation counts are proportional asymptotic comparisons. They do not include all projection/FFN work, constants, device speed or measured latency. Convolution path length uses the table's logarithmic comparison.
- The smoothing experiment assigns confidence `1−ε` to the true class and spreads `ε` among the other classes. It has no padding class. The paper specifies ε but does not spell out this distribution convention.

## Swift API

```swift
import AttentionCore

let controls = TransformerControls(
    attentionTemperature: 1,
    samplingTemperature: 1,
    useScaledAttention: true,
    causalMask: true,
    positionStrength: 1,
    headCount: 2,
    layerCount: 2
)
let trace = TransformerEngine().forward(
    source: "attention connects ideas",
    target: "we see",
    controls: controls
)
let selectedHead = trace.encoderLayers[0].selfAttention.heads[0]
let weight = selectedHead.weights[0, 1]
let nextToken = trace.sample(unitInterval: 0.37)
```

`Matrix` is row-major and exposes `rows`, `columns`, `values`, `row(_:)` and a two-index subscript. `AttentionHeadTrace` exposes `q`, `k`, `v`, `rawScores`, `scaledScores`, `maskedScores`, `weights` and `output`. The complete trace also exposes embedding components, concatenated head output, each residual sum/normalized result, FFN preactivation/ReLU/output, logits and probabilities. `nextTokenCandidates` sorts the last row for display; sampling itself uses the original vocabulary order.

`TransformerMath` supplies standalone `softmax`, `sinusoidalPositions`, `layerNormalize`, `attention`, `causalMask` and `entropy`. `PaperTraining` supplies `learningRate`, `smoothedTarget`, `crossEntropy` and optimizer/regularization constants. `PaperComplexity.estimate` returns the four Table 1 comparisons. `PaperChapter` supplies source-linked navigation metadata for 13 teaching chapters.

## Verification

22 XCTest checks passed on October 12, 2026 using Swift 6.4. They include independently computed attention values, large-logit stability, exact mask zeros, sinusoidal frequencies, postnorm feature statistics, tied output projection, head dimensions, rectangular cross-attention, full-stack causal invariance, reproducibility, sampling entropy, schedule peak, smoothing, complexity growth and sanitized input boundaries. These are CPU-core checks; they establish neither pretrained-model quality nor observed iPhone audio, haptics or foreground runtime.

```sh
env CLANG_MODULE_CACHE_PATH=/tmp/attention-clang-cache \
    SWIFTPM_MODULECACHE_OVERRIDE=/tmp/attention-swift-cache \
    swift test --disable-sandbox \
    --scratch-path /tmp/attention-core-build \
    --cache-path /tmp/attention-package-cache \
    --config-path /tmp/attention-package-config \
    --security-path /tmp/attention-package-security
```

Run from `AttentionMachine/AttentionCore`. The temporary caches avoid writes outside the managed workspace; `--disable-sandbox` avoids a nested SwiftPM `sandbox-exec` failure inside the host's existing sandbox.
