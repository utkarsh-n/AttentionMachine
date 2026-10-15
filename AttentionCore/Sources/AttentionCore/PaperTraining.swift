import Foundation

public enum PaperTraining {
    public static let adamBeta1 = 0.9
    public static let adamBeta2 = 0.98
    public static let adamEpsilon = 1e-9
    public static let baseDropout = 0.1
    public static let labelSmoothing = 0.1
    public static let warmupSteps = 4_000

    /// Equation 3. Step zero is explicitly defined as zero for plotting the origin.
    public static func learningRate(step: Int, dimension: Int = 512, warmup: Int = 4_000) -> Double {
        guard step > 0 else { return 0 }
        let step = Double(step)
        return pow(Double(max(1, dimension)), -0.5) * min(pow(step, -0.5), step * pow(Double(max(1, warmup)), -1.5))
    }

    /// Confidence 1-epsilon on the correct class; epsilon uniformly across other classes.
    /// This laboratory has no padding class. The paper states epsilon but does not specify this distribution convention.
    public static func smoothedTarget(index: Int, vocabularySize: Int, epsilon: Double = 0.1) -> [Double] {
        precondition(vocabularySize >= 2 && (0..<vocabularySize).contains(index))
        let epsilon = epsilon.isFinite ? min(1, max(0, epsilon)) : 0.1
        var distribution = Array(repeating: epsilon / Double(vocabularySize - 1), count: vocabularySize)
        distribution[index] = 1 - epsilon
        return distribution
    }

    public static func crossEntropy(probabilities: [Double], target: [Double]) -> Double {
        precondition(probabilities.count == target.count && !probabilities.isEmpty)
        return zip(probabilities, target).reduce(0) { $0 - $1.1 * log(max(1e-12, $1.0)) }
    }
}

public struct LayerComplexity: Identifiable, Sendable {
    public var id: String { name }
    public let name: String
    /// Proportional operation count from Table 1, not a measured CPU duration or a full Transformer FLOP count.
    public let operations: Double
    public let sequentialSteps: Double
    public let maximumPath: Double
    public let formula: String
}

public enum PaperComplexity {
    public static func estimate(sequenceLength: Int, dimension: Int, kernel: Int = 3,
                                neighborhood: Int = 4) -> [LayerComplexity] {
        let n = Double(max(1, sequenceLength))
        let d = Double(max(1, dimension))
        let k = Double(max(2, kernel))
        let r = Double(min(max(1, neighborhood), max(1, sequenceLength)))
        return [
            LayerComplexity(name: "Self-attention", operations: n * n * d, sequentialSteps: 1,
                            maximumPath: 1, formula: "O(n²d) • O(1) steps • O(1) path"),
            LayerComplexity(name: "Recurrent", operations: n * d * d, sequentialSteps: n,
                            maximumPath: n, formula: "O(nd²) • O(n) steps • O(n) path"),
            LayerComplexity(name: "Convolutional", operations: k * n * d * d, sequentialSteps: 1,
                            maximumPath: max(1, ceil(log(n) / log(k))), formula: "O(knd²) • O(1) steps • O(logₖ n) path"),
            LayerComplexity(name: "Restricted attention", operations: r * n * d, sequentialSteps: 1,
                            maximumPath: max(1, ceil(n / r)), formula: "O(rnd) • O(1) steps • O(n/r) path")
        ]
    }
}

/// Human-authored navigation through the whole paper, rather than a reproduction of its prose.
public enum PaperChapter: String, CaseIterable, Identifiable, Sendable {
    case motivation, architecture, embeddings, positions, attention, heads, encoder, decoder, output, complexity, training, results, appendix
    public var id: String { rawValue }
    public var title: String {
        switch self {
        case .motivation: "Parallel thought"
        case .architecture: "The whole machine"
        case .embeddings: "Tokens become vectors"
        case .positions: "Order becomes a wave"
        case .attention: "Query • key • value"
        case .heads: "Many views, one state"
        case .encoder: "Understand the source"
        case .decoder: "Generate with memory"
        case .output: "A distribution, then a token"
        case .complexity: "Distance and cost"
        case .training: "Learning the weights"
        case .results: "What the experiments showed"
        case .appendix: "Inspecting attention"
        }
    }
    public var section: String {
        switch self {
        case .motivation: "Abstract · §§1–2"
        case .architecture: "§3 · Figure 1"
        case .embeddings: "§3.4"
        case .positions: "§3.5"
        case .attention: "§3.2.1 · Equation 1"
        case .heads: "§3.2.2 · Figure 2"
        case .encoder: "§§3.1, 3.3 · Equation 2"
        case .decoder: "§§3.1, 3.2.3"
        case .output: "§§3.4, 6.1"
        case .complexity: "§4 · Table 1"
        case .training: "§5 · Equation 3"
        case .results: "§§6–7 · Tables 2–4"
        case .appendix: "Appendix · Figures 3–5"
        }
    }
    public var summary: String {
        switch self {
        case .motivation: "Attention connects distant tokens directly. Training positions can be processed together, while generation remains sequential."
        case .architecture: "A source encoder feeds a target decoder. The original uses six layers in each stack; this transparent machine uses smaller dimensions."
        case .embeddings: "Look up each token's vector, multiply by the square root of the feature count, and share the table with the output projection."
        case .positions: "Sine and cosine features give order to a model without recurrence. Add them to the embeddings before either stack."
        case .attention: "Project queries, keys and values. Score every permitted connection, normalize each row, then mix the value vectors."
        case .heads: "Separate projections create parallel attention views. Join their outputs and project the joined vector back into the model's feature space."
        case .encoder: "Each layer performs self-attention and a position-wise ReLU network. Add the incoming state around each operation, then normalize."
        case .decoder: "A causal mask blocks future target positions. Cross-attention then uses decoder queries to read the final encoder memory."
        case .output: "The final decoder state becomes vocabulary logits and probabilities. This lab samples; the paper's reported translation evaluation uses beam search."
        case .complexity: "Compare computation, sequential steps and dependency distance. Dense attention's token-pair matrix grows quadratically with sequence length."
        case .training: "Adam updates weights from smoothed target labels. A warmup ramp precedes inverse-square-root decay, and dropout regularizes training."
        case .results: "Read translation scores, controlled architecture variations and constituency parsing together. Historical benchmarks are evidence, not this toy's abilities."
        case .appendix: "The authors inspect heads with distant and syntactic connections. Attention weights describe routing, but alone do not establish an explanation."
        }
    }
    public var equation: String {
        switch self {
        case .motivation: "Attention path: O(1) · recurrent path: O(n)"
        case .architecture: "source → encoder → memory → decoder → logits"
        case .embeddings: "x = E[token] × √d_model"
        case .positions: "PE(pos,2i) = sin(pos / 10000^(2i/d))"
        case .attention: "Attention(Q,K,V) = softmax(QKᵀ / √d_k)V"
        case .heads: "MultiHead = Concat(head₁,…,headₕ)Wᴼ"
        case .encoder: "LayerNorm(x + Sublayer(x))"
        case .decoder: "future scores = −∞ before softmax"
        case .output: "p(token) = softmax(logits)"
        case .complexity: "dense attention: O(n²d)"
        case .training: "lr = d^−½ × min(step^−½, step × warmup^−1½)"
        case .results: "Table 2: 28.4 EN–DE · 41.8 EN–FR BLEU"
        case .appendix: "connection thickness ∝ attention weight"
        }
    }
    public var experiment: String {
        switch self {
        case .motivation: "Increase the token count and compare the number of sequential operations."
        case .architecture: "Follow a single token through both stacks and inspect each intermediate matrix."
        case .embeddings: "Repeat a word and compare its unscaled vectors with its position-dependent state."
        case .positions: "Turn the position strength to zero; repeated tokens now enter with identical states."
        case .attention: "Disable √d_k scaling and watch score magnitudes and row entropy change."
        case .heads: "Select another head, then change the number of heads while keeping eight total features."
        case .encoder: "Inspect the residual sum before normalization; compare the row's mean and variance."
        case .decoder: "Turn off the causal mask and inspect the newly permitted future connections."
        case .output: "Change output temperature and sample repeatedly from the same final state."
        case .complexity: "Double n: dense attention's proportional interaction cost quadruples."
        case .training: "Scrub past step 4,000 and redistribute confidence with label smoothing."
        case .results: "Compare one head with eight, base with big, and translation with parsing."
        case .appendix: "Compare routing from toy initialization with actual pretrained-model telemetry."
        }
    }
    public var sourceURL: String { "https://arxiv.org/html/1706.03762v7" }
}
