import SwiftUI
import Charts
import AttentionCore

private struct LabPanel<Content: View>: View {
    let title: String
    let content: Content

    init(title: String, @ViewBuilder content: () -> Content) {
        self.title = title
        self.content = content()
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text(title.uppercased()).font(.caption.weight(.semibold)).tracking(1).foregroundStyle(Theme.muted)
            content
        }
        .padding(18).frame(maxWidth: .infinity, alignment: .leading)
        .background(Theme.panel, in: RoundedRectangle(cornerRadius: 22))
    }
}

private struct LabSlider: View {
    @AppStorage("machine.haptics") private var haptics = true
    let title: String
    @Binding var value: Double
    let range: ClosedRange<Double>
    var step: Double = 1
    var suffix: String = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text(title).font(.subheadline.weight(.medium))
                Spacer()
                Text(value.formatted(.number.precision(.fractionLength(step < 1 ? 2 : 0))) + suffix)
                    .font(.subheadline.monospacedDigit()).foregroundStyle(Theme.mint)
            }
            Slider(value: $value, in: range, step: step).tint(Theme.mint).frame(minHeight: 44)
                .accessibilityIdentifier("paper-slider-\(title)")
                .accessibilityLabel(title)
                .accessibilityValue(value.formatted() + suffix)
                .sensoryFeedback(.selection, trigger: Int(value / step)) { _, _ in haptics }
        }.foregroundStyle(Theme.ink)
    }
}

private struct Formula: View {
    let text: String
    var body: some View {
        Text(text).font(.system(.subheadline, design: .monospaced)).foregroundStyle(Theme.mint)
            .fixedSize(horizontal: false, vertical: true).padding(.vertical, 4)
    }
}

private struct LabNote: View {
    let text: String
    var body: some View { Text(text).font(.footnote).foregroundStyle(Theme.muted).fixedSize(horizontal: false, vertical: true) }
}

private struct LabReadout: View {
    let title: String
    let value: String
    var color: Color = Theme.mint
    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title).font(.caption).foregroundStyle(Theme.muted)
            Text(value).font(.system(.title3, design: .rounded, weight: .semibold)).monospacedDigit().foregroundStyle(color)
                .accessibilityIdentifier("paper-readout-\(title)")
        }.frame(maxWidth: .infinity, alignment: .leading)
    }
}

private struct ProbabilityBars: View {
    let labels: [String]
    let probabilities: [Double]
    var color: Color = Theme.mint
    var body: some View {
        VStack(spacing: 12) {
            ForEach(Array(labels.enumerated()), id: \.offset) { index, label in
                let value = probabilities[index]
                VStack(alignment: .leading, spacing: 5) {
                    HStack {
                        Text(label).foregroundStyle(Theme.ink)
                        Spacer()
                        Text(value.formatted(.percent.precision(.fractionLength(1)))).foregroundStyle(color).monospacedDigit()
                    }.font(.caption)
                    GeometryReader { geometry in
                        ZStack(alignment: .leading) {
                            Capsule().fill(color.opacity(0.1))
                            Capsule().fill(color).frame(width: max(0, min(1, value)) * geometry.size.width)
                        }
                    }.frame(height: 7)
                }.accessibilityElement(children: .combine)
            }
        }
    }
}

private func probabilities(_ scores: [Double]) -> [Double] {
    TransformerMath.softmax(Matrix([scores])).row(0)
}

struct DotProductExperiment: View {
    @State private var angle = 30.0
    @State private var magnitude = 2.0
    @State private var scaled = true
    @State private var masked = false
    private let names = ["Key A · east", "Key B · north", "Key C · west"]
    private var query: [Double] { [magnitude * cos(angle * .pi / 180), magnitude * sin(angle * .pi / 180)] }
    private var scores: [Double] {
        let divisor = scaled ? sqrt(2.0) : 1
        return [query[0] / divisor, query[1] / divisor, masked ? -.infinity : -query[0] / divisor]
    }
    private var weights: [Double] { probabilities(scores) }
    private var output: Double { zip(weights, [1.0, 0.3, -1.0]).reduce(0) { $0 + $1.0 * $1.1 } }

    var body: some View {
        VStack(spacing: 18) {
            LabPanel(title: "Query / key compass") {
                VectorCompass(angle: angle, magnitude: magnitude)
                LabSlider(title: "Query angle", value: $angle, range: 0...360, suffix: "°")
                LabSlider(title: "Query magnitude", value: $magnitude, range: 0.1...8, step: 0.1)
                Toggle("Divide by √dₖ (√2 here)", isOn: $scaled).tint(Theme.mint).foregroundStyle(Theme.ink)
                Toggle("Mask key C before softmax", isOn: $masked).tint(Theme.mint).foregroundStyle(Theme.ink)
                Formula(text: "softmax(qKᵀ / √dₖ) V")
            }
            LabPanel(title: "Exact attention distribution") {
                ProbabilityBars(labels: names, probabilities: weights)
                HStack {
                    LabReadout(title: "Entropy", value: String(format: "%.3f nats", TransformerMath.entropy(weights)))
                    LabReadout(title: "Value mixture", value: String(format: "%+.3f", output), color: Theme.gold)
                }
                LabNote(text: "Values are [1, 0.3, −1]. A zero-length query would give equal weights; a larger magnitude concentrates softmax. The mask sets a score to −∞, so its weight becomes zero.")
            }
        }
    }
}

private struct VectorCompass: View {
    let angle: Double
    let magnitude: Double
    var body: some View {
        Canvas { context, size in
            let center = CGPoint(x: size.width / 2, y: size.height / 2)
            let radius = min(size.width, size.height) * 0.38
            let circle = Path(ellipseIn: CGRect(x: center.x - radius, y: center.y - radius, width: radius * 2, height: radius * 2))
            context.stroke(circle, with: .color(Theme.muted.opacity(0.2)), lineWidth: 1)
            for (degrees, color) in [(0.0, Theme.violet), (90.0, Theme.violet), (180.0, Theme.violet), (angle, Theme.coral)] {
                let length = degrees == angle && color == Theme.coral ? radius * (0.2 + magnitude / 10) : radius
                let radians = degrees * .pi / 180
                let point = CGPoint(x: center.x + cos(radians) * length, y: center.y - sin(radians) * length)
                var line = Path(); line.move(to: center); line.addLine(to: point)
                context.stroke(line, with: .color(color), lineWidth: color == Theme.coral ? 4 : 2)
                context.fill(Path(ellipseIn: CGRect(x: point.x - 5, y: point.y - 5, width: 10, height: 10)), with: .color(color))
            }
        }
        .frame(height: 180)
        .accessibilityLabel("Query points at \(Int(angle)) degrees with magnitude \(magnitude.formatted()). Keys point east, north, and west.")
    }
}

struct PositionExperiment: View {
    @State private var position = 12.0
    @State private var pair = 0
    private let dimensions = 16
    private var frequency: Double { pow(10_000, Double(2 * pair) / Double(dimensions)) }
    private var encoding: [Double] { TransformerMath.sinusoidalPositions(count: 1, dimension: dimensions, offset: Int(position)).row(0) }

    var body: some View {
        LabPanel(title: "A bank of position waves") {
            Chart {
                ForEach(0..<129, id: \.self) { point in
                    LineMark(x: .value("Position", point), y: .value("Encoding", sin(Double(point) / frequency)))
                        .foregroundStyle(by: .value("Channel", "sine"))
                    LineMark(x: .value("Position", point), y: .value("Encoding", cos(Double(point) / frequency)))
                        .foregroundStyle(by: .value("Channel", "cosine"))
                }
                RuleMark(x: .value("Selected position", position)).foregroundStyle(Theme.gold).lineStyle(StrokeStyle(lineWidth: 1, dash: [4]))
            }
            .chartForegroundStyleScale(["sine": Theme.mint, "cosine": Theme.violet])
            .chartYScale(domain: -1.1...1.1).frame(height: 210)
            .accessibilityLabel("Sine and cosine positional encodings. Selected position \(Int(position)), channel pair \(pair).")
            LabSlider(title: "Token position", value: $position, range: 0...128)
            Picker("Channel pair", selection: $pair) {
                ForEach(0..<8) { index in Text("\(2 * index), \(2 * index + 1)").tag(index) }
            }.pickerStyle(.menu).tint(Theme.mint).frame(minHeight: 44)
            HStack {
                LabReadout(title: "PE[pos, \(2 * pair)]", value: String(format: "%+.4f", encoding[2 * pair]))
                LabReadout(title: "PE[pos, \(2 * pair + 1)]", value: String(format: "%+.4f", encoding[2 * pair + 1]), color: Theme.violet)
            }
            Formula(text: "sin(pos / 10000^(2i/d))\ncos(pos / 10000^(2i/d))")
            LabNote(text: "This wave bank uses d = 16 so every channel is inspectable. The original base model uses d = 512. Higher channel pairs change more slowly.")
        }
    }
}

struct ResidualExperiment: View {
    @State private var strength = 1.0
    private let input = [-1.0, 0.2, 1.2, 0.6]
    private let sublayer = [1.5, -0.8, 0.3, -1.0]
    private var added: [Double] { zip(input, sublayer).map { $0 + strength * $1 } }
    private var normalized: [Double] { TransformerMath.layerNormalize(Matrix([added])).row(0) }

    var body: some View {
        LabPanel(title: "Add, then normalize") {
            Chart {
                ForEach(0..<4, id: \.self) { index in
                    BarMark(x: .value("Channel", "\(index)"), y: .value("Value", added[index]))
                        .foregroundStyle(by: .value("Stage", "Added")).position(by: .value("Stage", "Added"))
                    BarMark(x: .value("Channel", "\(index)"), y: .value("Value", normalized[index]))
                        .foregroundStyle(by: .value("Stage", "Normalized")).position(by: .value("Stage", "Normalized"))
                }
            }.chartForegroundStyleScale(["Added": Theme.coral, "Normalized": Theme.mint]).frame(height: 210)
            LabSlider(title: "Sublayer contribution α", value: $strength, range: 0...2, step: 0.05)
            HStack {
                LabReadout(title: "Added mean", value: String(format: "%+.3f", added.reduce(0, +) / 4), color: Theme.coral)
                LabReadout(title: "Normalized mean", value: String(format: "%+.3f", normalized.reduce(0, +) / 4))
            }
            Formula(text: "LayerNorm(x + α · Sublayer(x))")
            LabNote(text: "α is an experimental control; the paper uses α = 1. Normalization is across the features of one token, with γ = 1 and β = 0 here. The 2017 architecture applies normalization after the residual addition.")
        }
    }
}

struct FeedForwardExperiment: View {
    @State private var input = 0.5
    private var preactivations: [Double] { [input + 0.2, -2 * input + 0.5, 0.5 * input - 0.4] }
    private var hidden: [Double] { preactivations.map { max(0, $0) } }
    private var output: Double { zip(hidden, [0.8, -0.3, 1.2]).reduce(0.1) { $0 + $1.0 * $1.1 } }
    private func transform(_ value: Double) -> Double { 0.8 * max(0, value + 0.2) - 0.3 * max(0, -2 * value + 0.5) + 1.2 * max(0, 0.5 * value - 0.4) + 0.1 }

    var body: some View {
        VStack(spacing: 18) {
            LabPanel(title: "The nonlinear gate") {
                Chart {
                    ForEach(-80...80, id: \.self) { point in
                        let value = Double(point) / 40
                        LineMark(x: .value("Input", value), y: .value("Output", transform(value))).foregroundStyle(Theme.gold)
                    }
                    PointMark(x: .value("Input", input), y: .value("Output", output)).foregroundStyle(Theme.mint).symbolSize(90)
                    RuleMark(y: .value("Zero", 0)).foregroundStyle(Theme.muted.opacity(0.4))
                }.frame(height: 200)
                LabSlider(title: "Signed token feature x", value: $input, range: -2...2, step: 0.05)
                Formula(text: "max(0, xW₁ + b₁)W₂ + b₂")
            }
            LabPanel(title: "Inspect every hidden unit") {
                ForEach(0..<3, id: \.self) { index in
                    HStack {
                        Text("Unit \(index + 1)").foregroundStyle(Theme.ink)
                        Spacer()
                        Text(String(format: "%+.2f → %.2f", preactivations[index], hidden[index])).monospacedDigit().foregroundStyle(hidden[index] == 0 ? Theme.muted : Theme.gold)
                    }.font(.subheadline)
                }
                LabReadout(title: "Projected output", value: String(format: "%+.3f", output), color: Theme.gold)
                LabNote(text: "A one-feature, three-unit teaching network. The same weights process every token independently. Negative preactivations are clamped to zero; positive ones pass through.")
            }
        }
    }
}

struct EmbeddingExperiment: View {
    @State private var angle = 20.0
    @State private var selected = 0
    private let tokens = ["sun", "moon", "ocean", "forest"]
    private let embedding = [[1.0, 0.0], [0.0, 1.0], [-0.8, 0.3], [0.2, -0.9]]
    private var state: [Double] { [2 * cos(angle * .pi / 180), 2 * sin(angle * .pi / 180)] }
    private var logits: [Double] { embedding.map { row in zip(row, state).reduce(0) { $0 + $1.0 * $1.1 } } }

    var body: some View {
        LabPanel(title: "One matrix, two directions") {
            Picker("Input token", selection: $selected) { ForEach(0..<4) { Text(tokens[$0]).tag($0) } }
                .pickerStyle(.segmented).frame(minHeight: 44)
            LabReadout(title: "Embedding row × √2", value: String(format: "[%+.2f, %+.2f]", embedding[selected][0] * sqrt(2), embedding[selected][1] * sqrt(2)), color: Theme.violet)
            LabSlider(title: "Decoder hidden-state angle", value: $angle, range: 0...360, suffix: "°")
            ProbabilityBars(labels: tokens, probabilities: probabilities(logits))
            Formula(text: "Input: E[token] × √d\nOutput: softmax(hidden · Eᵀ)")
            LabNote(text: "This four-token, two-feature matrix is shared by the input lookup and output projection. Selecting an input row does not determine the decoder state: move its angle independently to inspect prediction logits.")
        }
    }
}

struct HeadsExperiment: View {
    @State private var heads = 8
    @State private var selectedHead = 0.0
    @State private var queryAngle = 20.0
    private let counts = [1, 4, 8, 16, 32]
    private var keyDimension: Int { 512 / heads }
    private var scores: [Double] {
        (0..<5).map { token in
            let projectedAngle = queryAngle * .pi / 180 + selectedHead * 2 * .pi / Double(heads)
            return 2 * cos(projectedAngle - Double(token) * 1.1)
        }
    }

    var body: some View {
        LabPanel(title: "Divide width, preserve the model") {
            Picker("Attention heads", selection: $heads) { ForEach(counts, id: \.self) { Text("\($0)").tag($0) } }
                .pickerStyle(.segmented).frame(minHeight: 44)
            HStack {
                LabReadout(title: "Heads × channels", value: "\(heads) × \(keyDimension)")
                LabReadout(title: "Concatenated width", value: "512", color: Theme.violet)
            }
            if heads > 1 { LabSlider(title: "Inspect head", value: $selectedHead, range: 0...Double(heads - 1)) }
            LabSlider(title: "Query angle", value: $queryAngle, range: 0...360, suffix: "°")
            ProbabilityBars(labels: ["Token 1", "Token 2", "Token 3", "Token 4", "Token 5"], probabilities: probabilities(scores))
            Formula(text: "Concat(head₁ … headₕ)Wᴼ\ndₖ = dᵥ = d_model / h")
            LabNote(text: "Fixed illustrative projections rotate the query for each head. Their scores are calculated, not extracted from a trained model. Each choice preserves d_model = 512; Q, K, V, and output projection parameters total 4 × 512² (biases excluded). The paper’s base setting is eight heads.")
        }.onChange(of: heads) { _, newValue in selectedHead = min(selectedHead, Double(newValue - 1)) }
    }
}

struct MaskExperiment: View {
    @State private var mode = 1
    @State private var query = 2.0
    private let labels = ["I", "see", "the", "sea", "."]
    private var matrix: [[Double]] {
        (0..<5).map { row in
            probabilities((0..<5).map { column in
                if mode == 1 && column > row { return -.infinity }
                if mode == 2 { return cos(Double(row + 1) * Double(column + 1)) * 1.5 }
                return -abs(Double(row - column)) * 0.55
            })
        }
    }

    var body: some View {
        LabPanel(title: "Who may read whom?") {
            Picker("Attention mode", selection: $mode) {
                Text("Encoder").tag(0); Text("Causal").tag(1); Text("Cross").tag(2)
            }.pickerStyle(.segmented).frame(minHeight: 44)
            AttentionGrid(values: matrix, selectedRow: Int(query), labels: labels)
            LabSlider(title: "Query row", value: $query, range: 0...4)
            ProbabilityBars(labels: labels, probabilities: matrix[Int(query)])
            Formula(text: mode == 1 ? "mask future scores → −∞ → softmax" : mode == 2 ? "Q from decoder; K,V from encoder" : "Q,K,V from the encoder stream")
            LabNote(text: mode == 1 ? "Future positions are zero before values are mixed. Decoder inputs are shifted right during training, so the target token itself is never provided as its own answer." : mode == 2 ? "Cross-attention has no causal restriction over the source memory. These illustrative scores pair decoder queries with encoder keys." : "Encoder tokens can read the entire input sequence. Each row is normalized separately and sums to one.")
        }
    }
}

private struct AttentionGrid: View {
    let values: [[Double]]
    let selectedRow: Int
    let labels: [String]
    var body: some View {
        VStack(spacing: 6) {
            HStack(spacing: 6) {
                Text("Q / K").font(.caption2).foregroundStyle(Theme.muted).frame(width: 42)
                ForEach(labels.indices, id: \.self) { index in Text(labels[index]).font(.caption2).foregroundStyle(Theme.muted).frame(maxWidth: .infinity) }
            }
            ForEach(values.indices, id: \.self) { row in
                HStack(spacing: 6) {
                    Text(labels[row]).font(.caption).foregroundStyle(row == selectedRow ? Theme.coral : Theme.muted).frame(width: 42)
                    ForEach(values[row].indices, id: \.self) { column in
                        RoundedRectangle(cornerRadius: 5).fill(Theme.mint.opacity(values[row][column] == 0 ? 0.025 : 0.1 + values[row][column] * 0.9))
                            .frame(maxWidth: .infinity).frame(height: 34)
                            .overlay { if values[row][column] == 0 { Image(systemName: "lock.fill").font(.caption2).foregroundStyle(Theme.muted) } }
                            .overlay { if row == selectedRow { RoundedRectangle(cornerRadius: 5).stroke(Theme.coral.opacity(0.7), lineWidth: 1) } }
                            .accessibilityLabel("Query \(labels[row]), key \(labels[column]), \(values[row][column].formatted(.percent))")
                    }
                }
            }
        }
    }
}

struct ComplexityExperiment: View {
    @State private var length = 128.0
    @State private var dimension = 512.0
    private var estimates: [LayerComplexity] { PaperComplexity.estimate(sequenceLength: Int(length), dimension: Int(dimension)) }

    var body: some View {
        VStack(spacing: 18) {
            LabPanel(title: "Table 1 · cost and parallelism") {
                Chart(estimates) { item in
                    BarMark(x: .value("Cost", item.operations), y: .value("Layer", item.name))
                        .foregroundStyle(item.name == "Self-attention" ? Theme.mint : Theme.violet)
                }.frame(height: 190)
                LabSlider(title: "Sequence length n", value: $length, range: 16...1024, step: 16)
                LabSlider(title: "Representation width d", value: $dimension, range: 64...1024, step: 64)
                HStack {
                    LabReadout(title: "Attention / recurrent cost", value: String(format: "%.2f×", length / dimension))
                    LabReadout(title: "Recurrent serial steps", value: "\(Int(length))", color: Theme.coral)
                }
                LabNote(text: "Proportional operation counts from Table 1, with k = 3 and restricted neighborhood r = 4. These omit constants and projections; the bars are not measured latency or a full model FLOP count.")
            }
            LabPanel(title: "The three comparisons") {
                ForEach(estimates) { item in
                    VStack(alignment: .leading, spacing: 6) {
                        Text(item.name).font(.subheadline.weight(.semibold)).foregroundStyle(Theme.ink)
                        Text(item.formula).font(.caption.monospaced()).foregroundStyle(Theme.muted)
                        Text("Serial stages: \(Int(item.sequentialSteps)) · maximum path estimate: \(Int(item.maximumPath))")
                            .font(.caption).foregroundStyle(Theme.mint)
                    }.padding(.vertical, 4)
                }
                LabNote(text: "Training positions in a layer can run together. Autoregressive generation still produces one output token at a time.")
            }
        }
    }
}

struct DependencyExperiment: View {
    @State private var distance = 32.0
    @State private var kernel = 3.0
    private var convolutionSteps: Int { Int(ceil(log(max(2, distance)) / log(kernel))) }

    var body: some View {
        LabPanel(title: "How far can a signal travel?") {
            Chart {
                ForEach(1...8, id: \.self) { layer in
                    LineMark(x: .value("Layers", layer), y: .value("Reach", pow(kernel, Double(layer))))
                        .foregroundStyle(Theme.violet)
                }
                RuleMark(y: .value("Required distance", distance)).foregroundStyle(Theme.coral).lineStyle(StrokeStyle(lineWidth: 2, dash: [5]))
            }.chartYScale(domain: 1...pow(kernel, 8), type: .log).frame(height: 200)
            LabSlider(title: "Token separation", value: $distance, range: 2...128)
            LabSlider(title: "Dilated convolution kernel k", value: $kernel, range: 2...8)
            HStack {
                LabReadout(title: "Self-attention path", value: "1")
                LabReadout(title: "Recurrent path", value: "\(Int(distance))", color: Theme.coral)
                LabReadout(title: "Dilated conv. layers", value: "\(convolutionSteps)", color: Theme.violet)
            }
            LabNote(text: "The curve shows an idealized k-fold receptive-field expansion for dilated convolutions. A full attention layer directly connects every permitted pair of tokens. Shorter routes reduce how many transformations a distant dependency must traverse.")
        }
    }
}

struct ArchitectureExperiment: View {
    @State private var source = 4.0
    @State private var target = 3.0
    @State private var layers = 6.0
    private var routes: [(String, Double)] {
        [("Encoder self", source * source * layers), ("Decoder causal", target * (target + 1) / 2 * layers), ("Cross-attention", source * target * layers)]
    }

    var body: some View {
        VStack(spacing: 18) {
            LabPanel(title: "Two stacks, three attention routes") {
                VStack(spacing: 10) {
                    routeNode("Source tokens", detail: "\(Int(source)) positions → embeddings + positions", color: Theme.violet)
                    Image(systemName: "arrow.down").foregroundStyle(Theme.muted)
                    routeNode("Encoder × \(Int(layers))", detail: "Self-attention → add & norm → FFN → add & norm", color: Theme.mint)
                    Image(systemName: "arrow.down").foregroundStyle(Theme.muted)
                    routeNode("Encoder memory → decoder", detail: "Causal self-attention → cross-attention → FFN", color: Theme.coral)
                    Image(systemName: "arrow.down").foregroundStyle(Theme.muted)
                    routeNode("Next-token probabilities", detail: "Linear projection → softmax → one token", color: Theme.gold)
                }
                LabSlider(title: "Source positions n", value: $source, range: 2...12)
                LabSlider(title: "Known target positions m", value: $target, range: 1...12)
                LabSlider(title: "Layers per stack N", value: $layers, range: 1...6)
            }
            LabPanel(title: "Permitted query–key pairs · one head") {
                Chart {
                    ForEach(routes, id: \.0) { route in
                        BarMark(x: .value("Route", route.0), y: .value("Connections", route.1)).foregroundStyle(Theme.mint)
                    }
                }.frame(height: 180)
                Formula(text: "Encoder: Nn²\nCausal decoder: Nm(m+1)/2\nCross-attention: Nnm")
                LabNote(text: "Counts cover all layers during a training pass, for one head. Change source length to see which routes need more memory connections. The paper’s original stacks each have N = 6.")
            }
        }
    }

    private func routeNode(_ title: String, detail: String, color: Color) -> some View {
        VStack(alignment: .leading, spacing: 5) {
            Text(title).font(.headline).foregroundStyle(color)
            Text(detail).font(.caption).foregroundStyle(Theme.muted)
        }.frame(maxWidth: .infinity, alignment: .leading).padding(13)
            .background(color.opacity(0.055), in: RoundedRectangle(cornerRadius: 12))
            .overlay(RoundedRectangle(cornerRadius: 12).stroke(color.opacity(0.2), lineWidth: 1))
    }
}

struct BatchingExperiment: View {
    @State private var sorted = true
    @State private var batchSize = 4
    private let lengths = [8, 13, 21, 5, 17, 9, 24, 6]
    private var ordered: [Int] { sorted ? lengths.sorted() : lengths }
    private var padded: [Int] {
        ordered.indices.map { index in
            let start = (index / batchSize) * batchSize
            return ordered[start..<min(start + batchSize, ordered.count)].max() ?? 0
        }
    }
    private var padding: Int { padded.reduce(0, +) - ordered.reduce(0, +) }

    var body: some View {
        LabPanel(title: "Packing variable-length sentences") {
            Chart {
                ForEach(ordered.indices, id: \.self) { index in
                    BarMark(x: .value("Sentence", "\(index + 1)"), y: .value("Tokens", ordered[index]))
                        .foregroundStyle(by: .value("Type", "Tokens"))
                    BarMark(x: .value("Sentence", "\(index + 1)"), y: .value("Tokens", padded[index] - ordered[index]))
                        .foregroundStyle(by: .value("Type", "Padding"))
                }
            }.chartForegroundStyleScale(["Tokens": Theme.mint, "Padding": Theme.coral.opacity(0.65)]).frame(height: 200)
            Toggle("Group similar lengths", isOn: $sorted).foregroundStyle(Theme.ink).tint(Theme.mint)
            Picker("Sentences per batch", selection: $batchSize) {
                Text("2 per batch").tag(2); Text("4 per batch").tag(4); Text("8 per batch").tag(8)
            }.pickerStyle(.segmented).frame(minHeight: 44)
            HStack {
                LabReadout(title: "Real tokens", value: "\(ordered.reduce(0, +))")
                LabReadout(title: "Padding tokens", value: "\(padding)", color: Theme.coral)
                LabReadout(title: "Utilization", value: (Double(ordered.reduce(0, +)) / Double(padded.reduce(0, +))).formatted(.percent.precision(.fractionLength(0))))
            }
            LabNote(text: "Eight illustrative sentence lengths, padded to the longest member of each batch. The paper batches by approximate length and uses roughly 25,000 source plus 25,000 target tokens per batch.")
        }
    }
}

struct LearningRateExperiment: View {
    @State private var step = 4000.0
    @State private var warmup = 4000.0
    @State private var dimension = 512
    private var rate: Double { PaperTraining.learningRate(step: Int(step), dimension: dimension, warmup: Int(warmup)) }

    var body: some View {
        VStack(spacing: 18) {
            LabPanel(title: "Equation 3 · warm up, then decay") {
                Chart {
                    ForEach(0...120, id: \.self) { sample in
                        let position = sample * 100
                        LineMark(x: .value("Training step", position), y: .value("Learning rate", PaperTraining.learningRate(step: position, dimension: dimension, warmup: Int(warmup))))
                            .foregroundStyle(Theme.mint)
                    }
                    RuleMark(x: .value("Warmup ends", warmup)).foregroundStyle(Theme.violet).lineStyle(StrokeStyle(lineWidth: 1, dash: [4]))
                    PointMark(x: .value("Selected step", step), y: .value("Learning rate", rate)).foregroundStyle(Theme.gold).symbolSize(75)
                }.frame(height: 210)
                LabSlider(title: "Training step", value: $step, range: 100...12000, step: 100)
                LabSlider(title: "Warmup steps", value: $warmup, range: 500...8000, step: 500)
                Picker("Model width", selection: $dimension) { Text("d = 512").tag(512); Text("d = 1024").tag(1024) }
                    .pickerStyle(.segmented).frame(minHeight: 44)
                LabReadout(title: step <= warmup ? "Warmup learning rate" : "Decay learning rate", value: String(format: "%.7f", rate))
                Formula(text: "d⁻½ min(step⁻½, step · warmup⁻³⁄²)")
            }
            LabPanel(title: "Reported training configuration") {
                Text("Adam · β₁ = 0.9 · β₂ = 0.98 · ε = 10⁻⁹").font(.subheadline.monospaced()).foregroundStyle(Theme.ink)
                LabNote(text: "Original warmup: 4,000 steps. Base: 100,000 steps, roughly 12 hours. Big: 300,000 steps, roughly 3.5 days. Both used eight NVIDIA P100 GPUs. Scrubbing this curve changes the schedule mathematically; it does not train weights.")
            }
        }
    }
}

struct RegularizationExperiment: View {
    @State private var smoothing = 0.1
    @State private var dropout = 0.1
    @State private var seed = 0
    private let tokens = ["correct token", "other A", "other B", "other C", "other D"]
    private var target: [Double] { PaperTraining.smoothedTarget(index: 0, vocabularySize: 5, epsilon: smoothing) }
    private var retained: [Bool] {
        (0..<20).map { index in
            let sample = sin(Double((index + 1) * 97 + seed * 211)) * 43758.5453
            return (sample - floor(sample)) >= dropout
        }
    }

    var body: some View {
        VStack(spacing: 18) {
            LabPanel(title: "Smooth the training target") {
                ProbabilityBars(labels: tokens, probabilities: target)
                LabSlider(title: "Label smoothing ε", value: $smoothing, range: 0...0.4, step: 0.01)
                LabReadout(title: "Target entropy", value: String(format: "%.3f nats", TransformerMath.entropy(target)))
                Formula(text: "correct: 1 − ε\nother: ε / (vocabulary − 1)")
                LabNote(text: "Five classes, with no padding class. This lab distributes smoothing mass over the other classes; the paper states ε = 0.1 without spelling out the distribution convention.")
            }
            LabPanel(title: "Drop a sublayer activation") {
                LazyVGrid(columns: Array(repeating: GridItem(.flexible(), spacing: 6), count: 10), spacing: 6) {
                    ForEach(0..<20, id: \.self) { index in
                        RoundedRectangle(cornerRadius: 5).fill(retained[index] ? Theme.gold : Theme.muted.opacity(0.12)).frame(height: 25)
                            .accessibilityLabel("Unit \(index + 1), \(retained[index] ? "retained" : "dropped")")
                    }
                }
                LabSlider(title: "Dropout probability", value: $dropout, range: 0...0.6, step: 0.05)
                HStack {
                    LabReadout(title: "Kept in this sample", value: "\(retained.filter { $0 }.count) / 20", color: Theme.gold)
                    LabReadout(title: "Kept-unit multiplier", value: String(format: "%.2f×", 1 / (1 - dropout)))
                }
                if #available(iOS 26.0, *) {
                    Button("Draw another mask", systemImage: "arrow.clockwise") { seed += 1 }
                        .buttonStyle(.glass).tint(Theme.mint).frame(minHeight: 44)
                } else {
                    Button("Draw another mask", systemImage: "arrow.clockwise") { seed += 1 }
                        .buttonStyle(.bordered).tint(Theme.mint).frame(minHeight: 44)
                }
                LabNote(text: "A reproducible pseudo-random mask illustrates inverted dropout: retained units are scaled by 1/(1 − p). Dropout is disabled for inference. The base paper setting is p = 0.1.")
            }
        }
    }
}

struct ResultsExperiment: View {
    @State private var language = 0
    @State private var metric = 0
    private var items: [(String, Double)] {
        if metric == 1 { return [("Base", 3.3e18), ("Big", 2.3e19)] }
        return language == 0 ? [("Base", 27.3), ("Big", 28.4)] : [("Base", 38.1), ("Big", 41.8)]
    }

    var body: some View {
        LabPanel(title: "Reported · Table 2") {
            Picker("Evaluation language", selection: $language) { Text("English → German").tag(0); Text("English → French").tag(1) }
                .pickerStyle(.segmented).frame(minHeight: 44)
            Picker("Metric", selection: $metric) { Text("BLEU").tag(0); Text("Training FLOPs").tag(1) }
                .pickerStyle(.segmented).frame(minHeight: 44)
            Chart {
                ForEach(items, id: \.0) { item in
                    BarMark(x: .value("Model", item.0), y: .value("Metric", item.1)).foregroundStyle(item.0 == "Base" ? Theme.mint : Theme.violet)
                }
            }.frame(height: 220)
            HStack {
                ForEach(items, id: \.0) { item in
                    LabReadout(title: item.0, value: metric == 0 ? String(format: "%.1f BLEU", item.1) : String(format: "%.1e", item.1), color: item.0 == "Base" ? Theme.mint : Theme.violet)
                }
            }
            LabReadout(title: metric == 0 ? "Reported big − base" : "Reported big / base training cost", value: metric == 0 ? String(format: "+%.1f BLEU", items[1].1 - items[0].1) : String(format: "%.2f×", items[1].1 / items[0].1), color: Theme.gold)
            LabNote(text: "WMT newstest2014 scores, copied as numbers from Table 2. Its cost column gives 3.3 × 10¹⁸ and 2.3 × 10¹⁹ estimated training FLOPs. This is historical evidence, not a benchmark of the downloaded model. The English–French prose reports 41.0, while the abstract and Table 2 report 41.8; this panel follows Table 2.")
        }
    }
}

struct AblationExperiment: View {
    @State private var family = 0
    @State private var selected = 0.0
    private var rows: [(String, Double, Double)] {
        switch family {
        case 1: [("2 layers", 23.7, 6.11), ("4 layers", 25.3, 5.19), ("6 layers", 25.8, 4.92), ("8 layers", 25.5, 4.88)]
        case 2: [("d = 256", 24.5, 5.75), ("d = 512", 25.8, 4.92), ("d = 1024", 26.0, 4.66)]
        case 3: [("FFN 1024", 25.4, 5.12), ("FFN 2048", 25.8, 4.92), ("FFN 4096", 26.2, 4.75)]
        case 4: [("dₖ = 16", 25.1, 5.16), ("dₖ = 32", 25.4, 5.01), ("dₖ = 64", 25.8, 4.92)]
        case 5: [("p = 0", 24.6, 5.77), ("p = 0.1", 25.8, 4.92), ("p = 0.2", 25.5, 4.95)]
        case 6: [("ε = 0", 25.3, 4.67), ("ε = 0.1", 25.8, 4.92), ("ε = 0.2", 25.7, 5.47)]
        case 7: [("Sinusoidal", 25.8, 4.92), ("Learned", 25.7, 4.92)]
        case 8: [("Base", 25.8, 4.92), ("Big", 26.4, 4.33)]
        default: [("1 head", 24.9, 5.29), ("4 heads", 25.5, 5.00), ("8 heads", 25.8, 4.92), ("16 heads", 25.8, 4.91), ("32 heads", 25.4, 5.01)]
        }
    }
    private var current: (String, Double, Double) { rows[min(Int(selected), rows.count - 1)] }

    var body: some View {
        LabPanel(title: "Reported · Table 3 · newstest2013") {
            Picker("Ablation family", selection: $family) {
                Text("Attention heads").tag(0); Text("Stack depth").tag(1)
                Text("Model width").tag(2); Text("Feed-forward width").tag(3)
                Text("Key dimensions").tag(4); Text("Dropout").tag(5)
                Text("Label smoothing").tag(6); Text("Positions").tag(7)
                Text("Base versus big").tag(8)
            }.pickerStyle(.menu).tint(Theme.mint).frame(minHeight: 44)
            Chart {
                ForEach(rows, id: \.0) { row in
                    BarMark(x: .value("Setting", row.0), y: .value("BLEU", row.1)).foregroundStyle(row.0 == current.0 ? Theme.gold : Theme.mint.opacity(0.6))
                }
            }.chartYScale(domain: 0...28).frame(height: 210)
            LabSlider(title: "Inspect reported setting", value: $selected, range: 0...Double(rows.count - 1))
            Text(current.0).font(.headline).foregroundStyle(Theme.gold)
            HStack {
                LabReadout(title: "Development BLEU", value: String(format: "%.1f", current.1))
                LabReadout(title: "Wordpiece perplexity", value: String(format: "%.2f", current.2), color: Theme.violet)
            }
            LabNote(text: "Discrete observed settings, with no interpolation or predicted score. Table 3 uses development data and no checkpoint averaging; its scores differ from Table 2’s test results. Wordpiece perplexity is not directly comparable to per-word perplexity.")
        }.onChange(of: family) { _, _ in selected = 0 }
    }
}

struct ParsingExperiment: View {
    @State private var split = 2.0
    @State private var semiSupervised = false
    private let tokens = ["the", "small", "machine", "learns", "."]
    private var serialization: String {
        "(S (NP \(tokens.prefix(Int(split) + 1).joined(separator: " "))) (VP \(tokens.dropFirst(Int(split) + 1).joined(separator: " "))))"
    }
    private var outputTokens: Int {
        serialization.replacingOccurrences(of: "(", with: " ( ").replacingOccurrences(of: ")", with: " ) ").split(whereSeparator: \.isWhitespace).count
    }

    var body: some View {
        LabPanel(title: "Tokens become a structured output") {
            HStack(spacing: 5) {
                ForEach(tokens.indices, id: \.self) { index in
                    Text(tokens[index]).font(.caption.weight(.semibold)).foregroundStyle(index <= Int(split) ? Theme.violet : Theme.mint)
                        .padding(.vertical, 14).frame(maxWidth: .infinity)
                        .background((index <= Int(split) ? Theme.violet : Theme.mint).opacity(0.08), in: RoundedRectangle(cornerRadius: 8))
                }
            }
            LabSlider(title: "Hypothetical phrase boundary", value: $split, range: 0...3)
            Formula(text: serialization)
            HStack {
                LabReadout(title: "Input tokens", value: "5")
                LabReadout(title: "Serialized output tokens", value: "\(outputTokens)", color: Theme.gold)
            }
            LabNote(text: "Moving the boundary constructs candidate spans; it does not claim they are grammatically correct. Structural labels and brackets make the output longer than the sentence.")
            Divider().overlay(Theme.muted.opacity(0.2))
            Toggle("Reported semi-supervised setting", isOn: $semiSupervised).foregroundStyle(Theme.ink).tint(Theme.mint)
            LabReadout(title: "Reported Transformer (4 layers) · WSJ 23 F1", value: semiSupervised ? "92.7" : "91.3")
            LabNote(text: "Table 4 reports a four-layer, d = 1024 Transformer. WSJ-only training used about 40,000 sentences; the semi-supervised corpus about 17 million. These are published results, separate from the phrase-boundary toy.")
        }
    }
}

struct RestrictedExperiment: View {
    @State private var length = 128.0
    @State private var radius = 16.0
    private var reach: Double { min(radius, length) }

    var body: some View {
        LabPanel(title: "Local attention · a proposed direction") {
            Chart {
                BarMark(x: .value("Scope", "Dense"), y: .value("Proportional pairs", length * length)).foregroundStyle(Theme.violet)
                BarMark(x: .value("Scope", "Restricted"), y: .value("Proportional pairs", length * reach)).foregroundStyle(Theme.mint)
            }.frame(height: 210)
            LabSlider(title: "Sequence length n", value: $length, range: 32...512, step: 16)
            LabSlider(title: "Neighborhood size r", value: $radius, range: 1...128)
            HStack {
                LabReadout(title: "Dense / local pair cost", value: String(format: "%.1f×", length / reach))
                LabReadout(title: "Local maximum path estimate", value: "\(Int(ceil(length / reach)))", color: Theme.coral)
            }
            Formula(text: "Dense: O(n²d), path O(1)\nRestricted: O(rnd), path O(n/r)")
            LabNote(text: "Table 1’s asymptotic comparison treats r as neighborhood size. Fewer pairs reduce per-layer work but longer dependencies need more hops. The paper identifies restricted attention and non-text modalities as future directions, not measured results here.")
        }
    }
}
