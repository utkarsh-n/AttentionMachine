import XCTest
@testable import AttentionCore

final class AttentionCoreTests: XCTestCase {
    func testMatrixProductAndConcatenationHaveIndependentGoldenValues() {
        let product = Matrix([[1, 2, 3], [4, 5, 6]]).multiplied(by: Matrix([[7, 8], [9, 10], [11, 12]]))
        XCTAssertEqual(product, Matrix([[58, 64], [139, 154]]))
        XCTAssertEqual(product.transposed, Matrix([[58, 139], [64, 154]]))
        XCTAssertEqual(Matrix.concatenateColumns([Matrix([[1], [2]]), Matrix([[3, 4], [5, 6]])]),
                       Matrix([[1, 3, 4], [2, 5, 6]]))
    }

    func testSoftmaxIsStableUnderLargeOffsets() {
        let probabilities = TransformerMath.softmax(Matrix([[1_000, 1_001, 999]]))
        XCTAssertEqual(probabilities[0, 0], 0.24472847105479764, accuracy: 1e-12)
        XCTAssertEqual(probabilities[0, 1], 0.6652409557748218, accuracy: 1e-12)
        XCTAssertEqual(probabilities[0, 2], 0.09003057317038046, accuracy: 1e-12)
        XCTAssertEqual(probabilities.row(0).reduce(0, +), 1, accuracy: 1e-12)
    }

    func testSoftmaxHasExactMaskZerosAndNoNaNsForFullyMaskedRows() {
        let probabilities = TransformerMath.softmax(Matrix([[2, -.infinity], [-.infinity, -.infinity]]))
        XCTAssertEqual(probabilities.row(0), [1, 0])
        XCTAssertEqual(probabilities.row(1), [0, 0])
        XCTAssertTrue(probabilities.values.allSatisfy(\.isFinite))
        XCTAssertEqual(TransformerMath.softmax(Matrix([[.infinity, 1, .infinity]])).row(0), [0.5, 0, 0.5])
    }

    func testSinusoidalPositionsPairFrequenciesAndStartAtZero() {
        let positions = TransformerMath.sinusoidalPositions(count: 3, dimension: 4)
        XCTAssertEqual(positions.row(0), [0, 1, 0, 1])
        XCTAssertEqual(positions[1, 0], sin(1), accuracy: 1e-12)
        XCTAssertEqual(positions[1, 1], cos(1), accuracy: 1e-12)
        XCTAssertEqual(positions[2, 2], sin(0.02), accuracy: 1e-12)
        XCTAssertEqual(positions[2, 3], cos(0.02), accuracy: 1e-12)
        XCTAssertEqual(TransformerMath.sinusoidalPositions(count: 1, dimension: 4, offset: 2).row(0), positions.row(2))
    }

    func testLayerNormUsesWithinTokenPopulationVariance() {
        let output = TransformerMath.layerNormalize(Matrix([[1, 2, 3], [5, 5, 5]]), epsilon: 1e-6)
        XCTAssertEqual(output[0, 0], -1 / sqrt(2.0 / 3 + 1e-6), accuracy: 1e-12)
        XCTAssertEqual(output[0, 1], 0)
        XCTAssertEqual(output[0, 2], 1 / sqrt(2.0 / 3 + 1e-6), accuracy: 1e-12)
        XCTAssertEqual(output.row(1), [0, 0, 0])
    }

    func testScaledAttentionMatchesHandComputedWeightedMixture() {
        let head = TransformerMath.attention(query: Matrix([[1, 0], [0, 1]]),
                                             key: Matrix([[1, 0], [0, 1]]), value: Matrix([[10, 1], [20, 5]]))
        let probability = exp(1 / sqrt(2.0)) / (exp(1 / sqrt(2.0)) + 1)
        XCTAssertEqual(head.rawScores, Matrix([[1, 0], [0, 1]]))
        XCTAssertEqual(head.weights[0, 0], probability, accuracy: 1e-12)
        XCTAssertEqual(head.output[0, 0], 10 * probability + 20 * (1 - probability), accuracy: 1e-12)
        XCTAssertEqual(head.output[1, 1], 1 * (1 - probability) + 5 * probability, accuracy: 1e-12)
    }

    func testCausalAttentionBlocksFutureValueEvenWhenItsScoreIsLarge() {
        let head = TransformerMath.attention(query: Matrix([[1], [2]]), key: Matrix([[1], [100]]),
                                             value: Matrix([[7], [999]]), causal: true)
        XCTAssertEqual(head.maskedScores[0, 1], -.infinity)
        XCTAssertEqual(head.weights[0, 1], 0)
        XCTAssertEqual(head.output[0, 0], 7)
        XCTAssertGreaterThan(head.output[1, 0], 998)
    }

    func testScalingAndTemperatureControlAttentionConcentration() {
        let q = Matrix([[4, 1, 2, 3]])
        let k = Matrix([[4, 1, 2, 3], [1, 1, 1, 1]])
        let v = Matrix([[1], [2]])
        let scaled = TransformerMath.attention(query: q, key: k, value: v)
        let unscaled = TransformerMath.attention(query: q, key: k, value: v, scale: false)
        let warmer = TransformerMath.attention(query: q, key: k, value: v, temperature: 3)
        XCTAssertGreaterThan(unscaled.weights[0, 0], scaled.weights[0, 0])
        XCTAssertGreaterThan(TransformerMath.entropy(warmer.weights.row(0)), TransformerMath.entropy(scaled.weights.row(0)))
    }

    func testCompleteForwardPassHasCorrectShapesAndNormalizedProbabilities() {
        let trace = TransformerEngine().forward(source: "attention connects ideas", target: "we see")
        XCTAssertEqual(trace.sourceTokens.count, 3)
        XCTAssertEqual(trace.decoderTokens, ["<bos>", "we", "see"])
        XCTAssertEqual(trace.expectedTargets, ["we", "see", "<eos>"])
        XCTAssertEqual(trace.encoderLayers.count, 2)
        XCTAssertEqual(trace.decoderLayers.count, 2)
        for encoder in trace.encoderLayers {
            XCTAssertEqual(encoder.selfAttention.heads.count, 2)
            XCTAssertEqual(encoder.selfAttention.concatenated.columns, 8)
            XCTAssertEqual(encoder.feedForward.relu.columns, 32)
            XCTAssertTrue(encoder.feedForward.relu.values.allSatisfy { $0 >= 0 })
        }
        for decoder in trace.decoderLayers {
            XCTAssertEqual(decoder.crossAttention.heads[0].q.rows, 3)
            XCTAssertEqual(decoder.crossAttention.heads[0].k.rows, 3)
            XCTAssertEqual(decoder.output.columns, 8)
            for head in decoder.selfAttention.heads {
                for r in 0..<3 {
                    XCTAssertEqual(head.weights.row(r).reduce(0, +), 1, accuracy: 1e-12)
                    for c in 0..<3 where c > r { XCTAssertEqual(head.weights[r, c], 0) }
                }
            }
        }
        for r in 0..<trace.probabilities.rows { XCTAssertEqual(trace.probabilities.row(r).reduce(0, +), 1, accuracy: 1e-12) }
        XCTAssertTrue(trace.probabilities.values.allSatisfy(\.isFinite))
        XCTAssertEqual(trace.probabilities.columns, trace.vocabulary.count)
    }

    func testDecoderCrossAttentionHasDifferentSourceAndTargetLengths() {
        let trace = TransformerEngine().forward(source: "we see the light", target: "we")
        let head = trace.decoderLayers[0].crossAttention.heads[0]
        XCTAssertEqual(head.q.rows, 2)
        XCTAssertEqual(head.k.rows, 4)
        XCTAssertEqual(head.weights.rows, 2)
        XCTAssertEqual(head.weights.columns, 4)
        XCTAssertEqual(head.output.rows, 2)
    }

    func testFullStackCausalityFutureTokensCannotChangeEarlierPredictions() {
        let engine = TransformerEngine()
        let first = engine.forward(source: "attention connects ideas", target: "we see light")
        let second = engine.forward(source: "attention connects ideas", target: "we see machine")
        XCTAssertEqual(first.vocabulary, second.vocabulary)
        for r in 0..<3 {
            for c in 0..<first.probabilities.columns {
                XCTAssertEqual(first.probabilities[r, c], second.probabilities[r, c], accuracy: 1e-12)
            }
        }
        XCTAssertNotEqual(first.probabilities.row(3), second.probabilities.row(3))
        let noMask = TransformerControls(causalMask: false)
        let unmaskedFirst = engine.forward(source: "attention connects ideas", target: "we see light", controls: noMask)
        let unmaskedSecond = engine.forward(source: "attention connects ideas", target: "we see machine", controls: noMask)
        XCTAssertNotEqual(unmaskedFirst.probabilities.row(0), unmaskedSecond.probabilities.row(0))
    }

    func testWeightsAreReproducibleAndNotTokenOrderHashRandomized() {
        let first = TransformerEngine().forward(source: "the machine sees light", target: "we")
        let second = TransformerEngine().forward(source: "the machine sees light", target: "we")
        XCTAssertEqual(first.probabilities, second.probabilities)
        XCTAssertEqual(first.encoderLayers[0].selfAttention.heads[0].q, second.encoderLayers[0].selfAttention.heads[0].q)
        XCTAssertNotEqual(first.probabilities, TransformerEngine(seed: 42).forward(source: "the machine sees light", target: "we").probabilities)
    }

    func testEmbeddingScalingPositionsAndTiedOutputProjection() {
        let trace = TransformerEngine().forward(source: "attention attention", target: "we")
        XCTAssertEqual(trace.sourceEmbedding.raw.row(0), trace.sourceEmbedding.raw.row(1))
        XCTAssertNotEqual(trace.sourceEmbedding.combined.row(0), trace.sourceEmbedding.combined.row(1))
        for c in 0..<8 {
            XCTAssertEqual(trace.sourceEmbedding.scaled[0, c], trace.sourceEmbedding.raw[0, c] * sqrt(8), accuracy: 1e-12)
            XCTAssertEqual(trace.sourceEmbedding.combined[0, c], trace.sourceEmbedding.scaled[0, c] + trace.sourceEmbedding.positions[0, c], accuracy: 1e-12)
        }
        let attentionIndex = trace.vocabulary.firstIndex(of: "attention")!
        let finalRow = trace.decoderLayers.last!.output.row(1)
        let expectedLogit = zip(finalRow, trace.sourceEmbedding.raw.row(0)).reduce(0) { $0 + $1.0 * $1.1 }
        XCTAssertEqual(trace.logits[1, attentionIndex], expectedLogit, accuracy: 1e-12)
        let bosIndex = trace.vocabulary.firstIndex(of: "<bos>")!
        XCTAssertEqual(trace.probabilities[1, bosIndex], 0)
    }

    func testRemovingPositionsMakesRepeatedTokensIndistinguishableInEncoder() {
        let trace = TransformerEngine().forward(source: "light light", controls: TransformerControls(positionStrength: 0))
        XCTAssertEqual(trace.sourceEmbedding.combined.row(0), trace.sourceEmbedding.combined.row(1))
        for layer in trace.encoderLayers { XCTAssertEqual(layer.output.row(0), layer.output.row(1)) }
    }

    func testEveryResidualIsPostNormAndFinite() {
        let trace = TransformerEngine().forward(source: "attention is all you need", target: "we see")
        let residuals = trace.encoderLayers.flatMap { [$0.attentionResidual, $0.feedForwardResidual] }
            + trace.decoderLayers.flatMap { [$0.attentionResidual, $0.crossResidual, $0.feedForwardResidual] }
        for residual in residuals {
            XCTAssertEqual(residual.sum, residual.input.adding(residual.sublayer))
            for r in 0..<residual.normalized.rows {
                let row = residual.normalized.row(r)
                XCTAssertEqual(row.reduce(0, +) / Double(row.count), 0, accuracy: 1e-12)
                let variance = row.reduce(0) { $0 + $1 * $1 } / Double(row.count)
                XCTAssertGreaterThan(variance, 0.999)
                XCTAssertLessThanOrEqual(variance, 1)
            }
            XCTAssertTrue(residual.normalized.values.allSatisfy(\.isFinite))
        }
    }

    func testOutputTemperatureOnlyChangesSamplingDistribution() {
        let cold = TransformerEngine().forward(source: "attention", controls: TransformerControls(samplingTemperature: 0.5))
        let warm = TransformerEngine().forward(source: "attention", controls: TransformerControls(samplingTemperature: 2))
        XCTAssertEqual(cold.logits, warm.logits)
        XCTAssertGreaterThan(warm.nextTokenEntropy, cold.nextTokenEntropy)
        XCTAssertEqual(cold.sample(unitInterval: 0.3), cold.sample(unitInterval: 0.3))
        XCTAssertTrue(cold.vocabulary.contains(cold.sample(unitInterval: -10)))
        XCTAssertTrue(cold.vocabulary.contains(cold.sample(unitInterval: 100)))
        XCTAssertNotEqual(cold.sample(unitInterval: 0), "<bos>")
        XCTAssertEqual(cold.nextTokenCandidates.first!.probability, cold.nextTokenCandidates.map(\.probability).max()!)
    }

    func testEmptyAndOversizedUserInputsAndNonfiniteControlsStaySafe() {
        let empty = TransformerEngine().forward(source: "", target: "")
        XCTAssertEqual(empty.sourceTokens, ["<empty>"])
        XCTAssertEqual(empty.decoderTokens, ["<bos>"])
        let controls = TransformerControls(attentionTemperature: .nan, samplingTemperature: .infinity,
                                            positionStrength: -.infinity, headCount: 3, layerCount: -1)
        let many = Array(repeating: "light", count: 100)
        let trace = TransformerEngine().forward(sourceTokens: many, targetTokens: many, controls: controls)
        XCTAssertEqual(trace.sourceTokens.count, 24)
        XCTAssertEqual(trace.decoderTokens.count, 24)
        XCTAssertTrue(trace.inputWasTruncated)
        XCTAssertEqual(trace.controls.headCount, 2)
        XCTAssertEqual(trace.controls.layerCount, 1)
        XCTAssertTrue(trace.probabilities.values.allSatisfy(\.isFinite))
    }

    func testAllSupportedHeadCountsMaintainModelDimension() {
        for headCount in [1, 2, 4, 8] {
            let trace = TransformerEngine().forward(source: "light", controls: TransformerControls(headCount: headCount, layerCount: 1))
            let layer = trace.encoderLayers[0]
            XCTAssertEqual(layer.selfAttention.heads.count, headCount)
            XCTAssertEqual(layer.selfAttention.heads[0].q.columns, 8 / headCount)
            XCTAssertEqual(layer.selfAttention.concatenated.columns, 8)
            XCTAssertEqual(layer.output.columns, 8)
        }
    }

    func testWarmupScheduleRisesThenDecaysWithKnownPeak() {
        XCTAssertEqual(PaperTraining.learningRate(step: 0), 0)
        let peak = PaperTraining.learningRate(step: 4_000)
        XCTAssertEqual(peak, 0.0006987712429686843, accuracy: 1e-15)
        XCTAssertEqual(PaperTraining.learningRate(step: 2_000), peak / 2, accuracy: 1e-15)
        XCTAssertEqual(PaperTraining.learningRate(step: 8_000), peak / sqrt(2), accuracy: 1e-15)
    }

    func testSmoothedTargetsNormalizeAndCrossEntropyMatchesGoldenValue() {
        let target = PaperTraining.smoothedTarget(index: 2, vocabularySize: 5)
        XCTAssertEqual(target, [0.025, 0.025, 0.9, 0.025, 0.025])
        XCTAssertEqual(target.reduce(0, +), 1, accuracy: 1e-12)
        XCTAssertEqual(PaperTraining.crossEntropy(probabilities: [0.25, 0.75], target: [0, 1]), -log(0.75), accuracy: 1e-12)
        XCTAssertEqual(PaperTraining.smoothedTarget(index: 1, vocabularySize: 3, epsilon: 0), [0, 1, 0])
    }

    func testComplexityDoublingSequenceHasQuadraticAndLinearEffects() {
        let small = PaperComplexity.estimate(sequenceLength: 8, dimension: 512)
        let large = PaperComplexity.estimate(sequenceLength: 16, dimension: 512)
        XCTAssertEqual(large[0].operations / small[0].operations, 4)
        XCTAssertEqual(large[1].operations / small[1].operations, 2)
        XCTAssertEqual(large[0].maximumPath, 1)
        XCTAssertEqual(large[1].sequentialSteps, 16)
        XCTAssertEqual(large[3].maximumPath, 4)
    }

    func testTeachingTokenizerSeparatesPunctuationAndUnknownTargets() {
        XCTAssertEqual(TransformerEngine.tokenize("Don't drop 42!"), ["don't", "drop", "42", "!"])
        let trace = TransformerEngine().forward(source: "light", target: "unseenword")
        XCTAssertEqual(trace.decoderTokens, ["<bos>", "<unk>"])
        XCTAssertFalse(trace.vocabulary.contains("unseenword"))
    }
}
