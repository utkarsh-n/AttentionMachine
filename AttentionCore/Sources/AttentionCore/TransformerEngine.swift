import Foundation

public struct TransformerControls: Equatable, Sendable {
    public var attentionTemperature: Double
    public var samplingTemperature: Double
    public var useScaledAttention: Bool
    public var causalMask: Bool
    public var positionStrength: Double
    public var headCount: Int
    public var layerCount: Int

    public init(attentionTemperature: Double = 1, samplingTemperature: Double = 1,
                useScaledAttention: Bool = true, causalMask: Bool = true,
                positionStrength: Double = 1, headCount: Int = 2, layerCount: Int = 2) {
        self.attentionTemperature = attentionTemperature
        self.samplingTemperature = samplingTemperature
        self.useScaledAttention = useScaledAttention
        self.causalMask = causalMask
        self.positionStrength = positionStrength
        self.headCount = headCount
        self.layerCount = layerCount
    }

    var sanitized: TransformerControls {
        var copy = self
        copy.attentionTemperature = attentionTemperature.isFinite ? min(5, max(0.05, attentionTemperature)) : 1
        copy.samplingTemperature = samplingTemperature.isFinite ? min(5, max(0.05, samplingTemperature)) : 1
        copy.positionStrength = positionStrength.isFinite ? min(3, max(0, positionStrength)) : 1
        copy.headCount = [1, 2, 4, 8].contains(headCount) ? headCount : 2
        copy.layerCount = min(6, max(1, layerCount))
        return copy
    }
}

public struct EmbeddingTrace: Sendable {
    public let raw: Matrix
    public let scaled: Matrix
    public let positions: Matrix
    public let combined: Matrix
}

public struct AttentionHeadTrace: Identifiable, Sendable {
    public var id: Int { index }
    public let index: Int
    public let q: Matrix
    public let k: Matrix
    public let v: Matrix
    public let rawScores: Matrix
    public let scaledScores: Matrix
    public let maskedScores: Matrix
    public let weights: Matrix
    public let output: Matrix
}

public struct MultiHeadTrace: Sendable {
    public let heads: [AttentionHeadTrace]
    public let concatenated: Matrix
    public let output: Matrix
}

public struct ResidualTrace: Sendable {
    public let input: Matrix
    public let sublayer: Matrix
    public let sum: Matrix
    public let normalized: Matrix
}

public struct FeedForwardTrace: Sendable {
    public let preActivation: Matrix
    public let relu: Matrix
    public let output: Matrix
}

public struct EncoderLayerTrace: Identifiable, Sendable {
    public var id: Int { index }
    public let index: Int
    public let input: Matrix
    public let selfAttention: MultiHeadTrace
    public let attentionResidual: ResidualTrace
    public let feedForward: FeedForwardTrace
    public let feedForwardResidual: ResidualTrace
    public var output: Matrix { feedForwardResidual.normalized }
}

public struct DecoderLayerTrace: Identifiable, Sendable {
    public var id: Int { index }
    public let index: Int
    public let input: Matrix
    public let selfAttention: MultiHeadTrace
    public let attentionResidual: ResidualTrace
    public let crossAttention: MultiHeadTrace
    public let crossResidual: ResidualTrace
    public let feedForward: FeedForwardTrace
    public let feedForwardResidual: ResidualTrace
    public var output: Matrix { feedForwardResidual.normalized }
}

public struct TokenProbability: Identifiable, Sendable {
    public var id: String { token }
    public let token: String
    public let probability: Double
}

public struct TransformerTrace: Sendable {
    public let sourceTokens: [String]
    /// Shifted decoder input: BOS followed by the already supplied target prefix.
    public let decoderTokens: [String]
    public let expectedTargets: [String]
    public let vocabulary: [String]
    public let controls: TransformerControls
    public let sourceEmbedding: EmbeddingTrace
    public let targetEmbedding: EmbeddingTrace
    public let encoderLayers: [EncoderLayerTrace]
    public let decoderLayers: [DecoderLayerTrace]
    public let logits: Matrix
    public let probabilities: Matrix
    public let inputWasTruncated: Bool
    public var dimension: Int { sourceEmbedding.raw.columns }
    public var headDimension: Int { dimension / controls.headCount }
    public var nextTokenCandidates: [TokenProbability] {
        zip(vocabulary, probabilities.row(probabilities.rows - 1))
            .map { TokenProbability(token: $0, probability: $1) }
            .sorted { $0.probability == $1.probability ? $0.token < $1.token : $0.probability > $1.probability }
    }
    public var nextTokenEntropy: Double { TransformerMath.entropy(probabilities.row(probabilities.rows - 1)) }

    /// Inverse-CDF sampling with a supplied random draw makes experiments reproducible.
    public func sample(unitInterval: Double) -> String {
        let draw = unitInterval.isFinite ? min(1 - Double.ulpOfOne, max(0, unitInterval)) : 0
        var cumulative = 0.0
        let row = probabilities.row(probabilities.rows - 1)
        for (index, probability) in row.enumerated() {
            cumulative += probability
            if draw < cumulative { return vocabulary[index] }
        }
        return vocabulary.last ?? "<eos>"
    }
}

/// A complete, small Transformer forward pass with explicit intermediate tensors.
/// Weights are reproducible pseudo-random INITIALIZATIONS, never trained or pretrained.
/// The 8-feature/32-feature model demonstrates the paper's arithmetic, not its translation quality.
public struct TransformerEngine: Sendable {
    public static let dimension = 8
    public static let feedForwardDimension = 32
    public static let maximumTokens = 24
    public static let provenance = "Deterministic untrained CPU Transformer • 8 features • postnorm encoder–decoder"
    private let seed: UInt64

    public init(seed: UInt64 = 2017) { self.seed = seed }

    /// Word/punctuation splitting is illustrative; it is not the paper's BPE tokenizer.
    public static func tokenize(_ text: String) -> [String] {
        var tokens: [String] = []
        var word = ""
        for character in text.lowercased() {
            if character.isLetter || character.isNumber || character == "'" {
                word.append(character)
            } else {
                if !word.isEmpty { tokens.append(word); word = "" }
                if !character.isWhitespace { tokens.append(String(character)) }
            }
        }
        if !word.isEmpty { tokens.append(word) }
        return tokens
    }

    public func forward(source: String, target: String = "", controls: TransformerControls = .init()) -> TransformerTrace {
        forward(sourceTokens: Self.tokenize(source), targetTokens: Self.tokenize(target), controls: controls)
    }

    public func forward(sourceTokens: [String], targetTokens: [String] = [],
                        controls: TransformerControls = .init()) -> TransformerTrace {
        let controls = controls.sanitized
        let source = sourceTokens.isEmpty ? ["<empty>"] : Array(sourceTokens.prefix(Self.maximumTokens))
        // Fix the output vocabulary from the source before reading the target prefix.
        // Including future target words here would leak them through the softmax denominator.
        let vocabulary = Array(Set(Self.baseVocabulary + source)).sorted()
        let vocabularySet = Set(vocabulary)
        let target = Array(targetTokens.prefix(Self.maximumTokens - 1)).map { vocabularySet.contains($0) ? $0 : "<unk>" }
        let decoder = ["<bos>"] + target
        let embeddingTable = Matrix(vocabulary.map { embedding(for: $0) })
        let sourceEmbedding = embed(source, strength: controls.positionStrength)
        let targetEmbedding = embed(decoder, strength: controls.positionStrength)
        var encoder = sourceEmbedding.combined
        var encoderLayers: [EncoderLayerTrace] = []
        for layer in 0..<controls.layerCount {
            let attention = multiHead(query: encoder, memory: encoder, causal: false,
                                      controls: controls, family: "encoder.\(layer).self")
            let residual = postNorm(input: encoder, sublayer: attention.output)
            let ffn = feedForward(residual.normalized, family: "encoder.\(layer).ffn")
            let ffnResidual = postNorm(input: residual.normalized, sublayer: ffn.output)
            encoderLayers.append(EncoderLayerTrace(index: layer, input: encoder, selfAttention: attention,
                                                  attentionResidual: residual, feedForward: ffn,
                                                  feedForwardResidual: ffnResidual))
            encoder = ffnResidual.normalized
        }
        var decoderState = targetEmbedding.combined
        var decoderLayers: [DecoderLayerTrace] = []
        for layer in 0..<controls.layerCount {
            let attention = multiHead(query: decoderState, memory: decoderState, causal: controls.causalMask,
                                      controls: controls, family: "decoder.\(layer).self")
            let residual = postNorm(input: decoderState, sublayer: attention.output)
            // Every decoder layer reads the FINAL encoder output, not the corresponding encoder layer.
            let cross = multiHead(query: residual.normalized, memory: encoder, causal: false,
                                  controls: controls, family: "decoder.\(layer).cross")
            let crossResidual = postNorm(input: residual.normalized, sublayer: cross.output)
            let ffn = feedForward(crossResidual.normalized, family: "decoder.\(layer).ffn")
            let ffnResidual = postNorm(input: crossResidual.normalized, sublayer: ffn.output)
            decoderLayers.append(DecoderLayerTrace(index: layer, input: decoderState, selfAttention: attention,
                                                  attentionResidual: residual, crossAttention: cross,
                                                  crossResidual: crossResidual, feedForward: ffn,
                                                  feedForwardResidual: ffnResidual))
            decoderState = ffnResidual.normalized
        }
        // The unscaled embedding table is tied to the output projection, as in §3.4.
        var logits = decoderState.multiplied(by: embeddingTable.transposed)
        // BOS is input-only in this teaching vocabulary; suppress it at generation time.
        if let bos = vocabulary.firstIndex(of: "<bos>") {
            for row in 0..<logits.rows { logits[row, bos] = -.infinity }
        }
        let probabilities = TransformerMath.softmax(logits, temperature: controls.samplingTemperature)
        return TransformerTrace(sourceTokens: source, decoderTokens: decoder, expectedTargets: target + ["<eos>"],
                                vocabulary: vocabulary, controls: controls, sourceEmbedding: sourceEmbedding,
                                targetEmbedding: targetEmbedding, encoderLayers: encoderLayers,
                                decoderLayers: decoderLayers, logits: logits, probabilities: probabilities,
                                inputWasTruncated: sourceTokens.count > Self.maximumTokens || targetTokens.count >= Self.maximumTokens)
    }

    private static let baseVocabulary = ["<bos>", "<eos>", "<unk>", "the", "a", "machine", "attention", "is", "all", "you",
                                         "need", "we", "see", "light", "connects", "ideas", "tokens", "learn", "from",
                                         "context", "it", "because", "can", "think", "."]

    private func embed(_ tokens: [String], strength: Double) -> EmbeddingTrace {
        let raw = Matrix(tokens.map { embedding(for: $0) })
        let scaled = raw.scaled(by: sqrt(Double(Self.dimension)))
        let positions = TransformerMath.sinusoidalPositions(count: tokens.count, dimension: Self.dimension)
        return EmbeddingTrace(raw: raw, scaled: scaled, positions: positions,
                              combined: scaled.adding(positions.scaled(by: strength)))
    }

    private func embedding(for token: String) -> [Double] {
        var generator = SeededGenerator(seed: seed ^ stableHash("embedding.\(token)"))
        return (0..<Self.dimension).map { _ in generator.uniformSigned() * 0.6 }
    }

    private func weights(rows: Int, columns: Int, family: String) -> Matrix {
        var generator = SeededGenerator(seed: seed ^ stableHash(family))
        let scale = sqrt(6 / Double(rows + columns))
        return Matrix(rows: rows, columns: columns, values: (0..<(rows * columns)).map { _ in
            generator.uniformSigned() * scale
        })
    }

    private func multiHead(query: Matrix, memory: Matrix, causal: Bool,
                           controls: TransformerControls, family: String) -> MultiHeadTrace {
        let headDimension = Self.dimension / controls.headCount
        let heads = (0..<controls.headCount).map { head in
            TransformerMath.attention(
                query: query.multiplied(by: weights(rows: Self.dimension, columns: headDimension, family: "\(family).\(head).q")),
                key: memory.multiplied(by: weights(rows: Self.dimension, columns: headDimension, family: "\(family).\(head).k")),
                value: memory.multiplied(by: weights(rows: Self.dimension, columns: headDimension, family: "\(family).\(head).v")),
                scale: controls.useScaledAttention, causal: causal, temperature: controls.attentionTemperature, index: head)
        }
        let concatenated = Matrix.concatenateColumns(heads.map(\.output))
        let output = concatenated.multiplied(by: weights(rows: Self.dimension, columns: Self.dimension, family: "\(family).out"))
        return MultiHeadTrace(heads: heads, concatenated: concatenated, output: output)
    }

    private func postNorm(input: Matrix, sublayer: Matrix) -> ResidualTrace {
        let sum = input.adding(sublayer)
        return ResidualTrace(input: input, sublayer: sublayer, sum: sum, normalized: TransformerMath.layerNormalize(sum))
    }

    private func feedForward(_ input: Matrix, family: String) -> FeedForwardTrace {
        let w1 = weights(rows: Self.dimension, columns: Self.feedForwardDimension, family: "\(family).w1")
        let b1 = weights(rows: 1, columns: Self.feedForwardDimension, family: "\(family).b1").scaled(by: 0.05).values
        let preActivation = input.multiplied(by: w1).addingRow(b1)
        let relu = Matrix(rows: input.rows, columns: Self.feedForwardDimension, values: preActivation.values.map { max(0, $0) })
        let w2 = weights(rows: Self.feedForwardDimension, columns: Self.dimension, family: "\(family).w2")
        let b2 = weights(rows: 1, columns: Self.dimension, family: "\(family).b2").scaled(by: 0.05).values
        return FeedForwardTrace(preActivation: preActivation, relu: relu, output: relu.multiplied(by: w2).addingRow(b2))
    }
}

extension TransformerMath {
    public static func attention(query: Matrix, key: Matrix, value: Matrix, scale: Bool = true,
                                 causal: Bool = false, temperature: Double = 1, index: Int = 0) -> AttentionHeadTrace {
        precondition(query.columns == key.columns && key.rows == value.rows)
        let raw = query.multiplied(by: key.transposed)
        let scaled = raw.scaled(by: scale ? 1 / sqrt(Double(query.columns)) : 1)
        let masked = causal ? causalMask(scaled) : scaled
        let weights = softmax(masked, temperature: temperature)
        return AttentionHeadTrace(index: index, q: query, k: key, v: value, rawScores: raw,
                                  scaledScores: scaled, maskedScores: masked,
                                  weights: weights, output: weights.multiplied(by: value))
    }
}

private func stableHash(_ string: String) -> UInt64 {
    string.utf8.reduce(UInt64(14_695_981_039_346_656_037)) { ($0 ^ UInt64($1)) &* 1_099_511_628_211 }
}

private struct SeededGenerator {
    var seed: UInt64
    mutating func uniformSigned() -> Double {
        seed &+= 0x9E3779B97F4A7C15
        var value = seed
        value = (value ^ (value >> 30)) &* 0xBF58476D1CE4E5B9
        value = (value ^ (value >> 27)) &* 0x94D049BB133111EB
        value ^= value >> 31
        return Double(value >> 11) / Double(UInt64(1) << 53) * 2 - 1
    }
}
