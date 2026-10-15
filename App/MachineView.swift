import SwiftUI
import AttentionCore

enum MachineStage: Int, CaseIterable, Identifiable {
    case embedding, attention, residual, feedForward, masked, cross, output
    var id: Int { rawValue }
    var title: String {
        ["Embedding + position", "Multi-head attention", "Add & normalize", "Feed-forward network", "Masked decoder", "Encoder → decoder", "Next-token distribution"][rawValue]
    }
    var short: String { ["Embed", "Attend", "Normalize", "Transform", "Mask", "Connect", "Predict"][rawValue] }
    var equation: String {
        ["x = √dₘₒdₑₗ · E(token) + PE(position)", "Attention(Q,K,V) = softmax(QKᵀ/√dₖ)V", "LayerNorm(x + Sublayer(x))", "FFN(x) = max(0, xW₁ + b₁)W₂ + b₂", "Attention sees only the past + present", "Queries: decoder · keys & values: encoder", "p(next token) = softmax(hidden · Eᵀ)"][rawValue]
    }
    var description: String {
        ["A token becomes a vector. Sine and cosine add its place in the sequence.",
         "Each query asks how strongly to connect to every key. Those weights mix the values.",
         "A shortcut preserves the input. Normalization recenters and rescales each token vector.",
         "Each token passes through the same two-layer network. ReLU opens only positive gates.",
         "The shifted target sequence predicts one token at a time. Future keys receive zero weight.",
         "The decoder asks questions of the encoder’s final representations, connecting source to target.",
         "The embedding table is reused as an output projection. A probability distribution becomes the next token."][rawValue]
    }
}

struct MachineView: View {
    @Environment(\.scenePhase) private var scenePhase
    @AppStorage("machine.source") private var source = "The animal did not cross the street"
    @AppStorage("machine.target") private var target = "because it was tired"
    @AppStorage("machine.sound") private var sound = false
    @AppStorage("machine.haptics") private var haptics = true
    @AppStorage("machine.soundGain") private var soundGain = 0.5
    @State private var controls = TransformerControls(headCount: 2, layerCount: 2)
    @State private var trace = TransformerEngine().forward(source: "The animal did not cross the street", target: "because it was tired")
    @State private var stage = MachineStage.attention
    @State private var query = 1
    @State private var head = 0
    @State private var layer = 0
    @State private var cellKey = 0
    @State private var playing = false
    @State private var showingInputs = false
    @State private var showingControls = false
    @State private var showingSensory = ProcessInfo.processInfo.arguments.contains("--sensory-check")
    @State private var endSampled = false
    @State private var sensory = SensoryEngine()
    @State private var rebuildTask: Task<Void, Never>?
    @State private var rebuildGeneration = 0
    @State private var isCalculating = false

    private var encoder: EncoderLayerTrace { trace.encoderLayers[min(layer, trace.encoderLayers.count - 1)] }
    private var decoder: DecoderLayerTrace { trace.decoderLayers[min(layer, trace.decoderLayers.count - 1)] }
    private var activeAttention: MultiHeadTrace {
        switch stage {
        case .masked: decoder.selfAttention
        case .cross: decoder.crossAttention
        default: encoder.selfAttention
        }
    }
    private var activeHead: AttentionHeadTrace { activeAttention.heads[min(head, activeAttention.heads.count - 1)] }
    private var queryTokens: [String] { stage == .masked || stage == .cross || stage == .output ? trace.decoderTokens : trace.sourceTokens }
    private var keyTokens: [String] { stage == .masked ? trace.decoderTokens : trace.sourceTokens }
    private var row: Int { min(query, max(0, queryTokens.count - 1)) }
    private var stageEquation: String {
        if stage == .masked && !trace.controls.causalMask { return "Mask disabled: future target keys can receive attention" }
        if stage == .attention && !trace.controls.useScaledAttention { return "Scaling disabled: Attention(Q,K,V) = softmax(QKᵀ)V" }
        if stage == .embedding && trace.controls.positionStrength != 1 { return "x = √8 · E(token) + \(trace.controls.positionStrength.formatted()) · PE(position)" }
        return stage.equation
    }
    private var stageDescription: String {
        if stage == .masked && !trace.controls.causalMask {
            return "The causal gate is open for this experiment. Future target tokens can now influence earlier predictions; restore the mask to recover autoregressive behavior."
        }
        return stage.description
    }

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    header
                    if stage == .output {
                        Text("Prefix: \(trace.decoderTokens.joined(separator: " · "))")
                            .font(.system(.caption, design: .monospaced)).foregroundStyle(Theme.coral)
                            .fixedSize(horizontal: false, vertical: true)
                    } else {
                        TokenStrip(tokens: queryTokens, selected: $query, tint: Theme.coral) { _ in sensory.step(haptics: haptics) }
                    }
                    if trace.inputWasTruncated {
                        Text("Input capped at 24 source and 23 prefix tokens. Edit the sentences to explore the rest.")
                            .font(.caption).foregroundStyle(Theme.gold).accessibilityIdentifier("input-cap-note")
                    }
                    machine
                    explanation
                    HStack {
                        Label("Everything here is calculated on your iPhone.", systemImage: "iphone")
                            .font(.caption).foregroundStyle(Theme.muted)
                            .accessibilityIdentifier("machine-device-note")
                    }
                    .padding(.bottom, 20)
                }.padding(.horizontal, 20).padding(.top, 16)
            }
            .background { Theme.bg.ignoresSafeArea(); GridBackdrop().ignoresSafeArea() }
            .safeAreaInset(edge: .bottom, spacing: 0) {
                stageControls.padding(.horizontal, 20).padding(.vertical, 8)
                    .background(Theme.bg.opacity(0.92))
            }
            .toolbar(.hidden, for: .navigationBar)
            .sheet(isPresented: $showingInputs) { inputsSheet }
            .sheet(isPresented: $showingControls) { controlsSheet }
            .sheet(isPresented: $showingSensory) { SensoryCheckView(engine: sensory) }
            .task { rebuild(); sensory.prepare(haptics: haptics) }
            .onChange(of: controls) { _, _ in rebuild(debounced: true) }
            .onChange(of: stage) { _, _ in
                query = min(query, queryTokens.count - 1)
                cellKey = min(cellKey, keyTokens.count - 1)
                sensory.step(haptics: haptics)
            }
            .onChange(of: scenePhase) { _, value in
                if value != .active { playing = false; sensory.stop() }
                else { sensory.prepare(haptics: haptics) }
            }
            .onDisappear {
                playing = false; sensory.stop()
                rebuildTask?.cancel(); rebuildTask = nil; isCalculating = false
            }
            .task(id: playing) {
                while playing && !Task.isCancelled {
                    try? await Task.sleep(for: .seconds(1.25))
                    guard !Task.isCancelled, playing else { break }
                    stage = MachineStage(rawValue: (stage.rawValue + 1) % MachineStage.allCases.count)!
                }
            }
        }
    }

    private var header: some View {
        HStack(alignment: .center) {
            VStack(alignment: .leading, spacing: 6) {
                SectionEyebrow(text: "Transformer / 2017")
                Text("Attention").font(.system(.largeTitle, design: .rounded).weight(.semibold)).foregroundStyle(Theme.ink)
                    .accessibilityLabel("Attention Is All You Need")
                Text("IS ALL YOU NEED").font(.system(.caption2, design: .monospaced)).tracking(3).foregroundStyle(Theme.muted)
                Text("d = 8 · SEEDED / UNTRAINED").font(.system(.caption2, design: .monospaced)).foregroundStyle(Theme.mint)
            }
            Spacer()
            VStack(spacing: 8) {
                Button { sound.toggle(); if !sound { sensory.stop() }; sensory.step(haptics: haptics) } label: {
                    Image(systemName: sound ? "speaker.wave.2" : "speaker.slash").frame(width: 44, height: 44)
                }.buttonStyle(.glass).accessibilityLabel(sound ? "Turn attention tones off" : "Turn attention tones on")
                Button { playing = false; showingSensory = true } label: {
                    Image(systemName: "waveform.path").frame(width: 44, height: 44)
                }.buttonStyle(.glass).accessibilityLabel("Sound and touch check").accessibilityIdentifier("sensory-check")
            }
        }
    }

    private var machine: some View {
        InstrumentPanel {
            VStack(alignment: .leading, spacing: 14) {
                HStack(alignment: .firstTextBaseline) {
                    Text(stage.title).font(.headline).foregroundStyle(Theme.ink)
                        .accessibilityIdentifier("machine-stage-title")
                    Spacer()
                    Text("\(stage.rawValue + 1) / 7").font(.system(.caption, design: .monospaced)).foregroundStyle(Theme.muted)
                }
                Text(isCalculating ? "Updating calculation…" : "Calculation ready · \(trace.controls.headCount) heads · \(trace.controls.layerCount) layers")
                    .font(.caption2).foregroundStyle(Theme.muted).accessibilityIdentifier("machine-calculation-state")
                    .accessibilityValue(String(format: "Position strength: %.6f", trace.controls.positionStrength))
                if stage == .attention || stage == .masked || stage == .cross {
                    projections
                    AttentionWires(weights: activeHead.weights.row(min(row, activeHead.weights.rows - 1)), selected: cellKey)
                        .frame(height: 28)
                    HStack {
                        Spacer(minLength: 0)
                        HeatmapView(matrix: nested(activeHead.weights), tokens: keyTokens, selectedRow: $query,
                                    causal: stage == .masked && trace.controls.causalMask) { _, k, weight in
                            cellKey = k
                            sensory.touch(weight: weight, pitch: k, sound: sound, haptics: haptics, soundGain: soundGain)
                        }.frame(maxWidth: 240)
                        Spacer(minLength: 0)
                    }
                    attentionReadout
                } else {
                    tensorStage
                }
                Text(stageEquation).font(.system(.caption, design: .monospaced)).foregroundStyle(Theme.mint)
                    .accessibilityIdentifier("machine-equation")
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    private var projections: some View {
        HStack(spacing: 8) {
            port("Q", label: "QUERY", color: Theme.coral, vector: activeHead.q.row(min(row, activeHead.q.rows - 1)))
            port("K", label: "KEY", color: Theme.violet, vector: activeHead.k.row(min(cellKey, activeHead.k.rows - 1)))
            port("V", label: "VALUE", color: Theme.gold, vector: activeHead.v.row(min(cellKey, activeHead.v.rows - 1)))
        }
    }

    private func port(_ symbol: String, label: String, color: Color, vector: [Double]) -> some View {
        VStack(spacing: 4) {
            Text(symbol).font(.system(size: 21, weight: .medium, design: .rounded)).foregroundStyle(color)
                .frame(width: 36, height: 36).background(color.opacity(0.06), in: Circle())
                .overlay(Circle().stroke(color.opacity(0.8), lineWidth: 1.5))
            Text(label).font(.system(.caption2, design: .monospaced)).tracking(2).foregroundStyle(color)
            TinyVector(values: vector, color: color).frame(height: 10)
        }.frame(maxWidth: .infinity).accessibilityElement(children: .ignore)
            .accessibilityLabel("\(label) projection, \(vector.map { String(format: "%.2f", $0) }.joined(separator: ", "))")
    }

    private var attentionReadout: some View {
        let r = min(row, activeHead.weights.rows - 1)
        let k = min(cellKey, activeHead.weights.columns - 1)
        return HStack(alignment: .top) {
            ValueReadout(title: "\(queryTokens[r]) → \(keyTokens[k])", value: String(format: "%.1f%%", activeHead.weights[r, k] * 100), color: Theme.coral)
            Spacer()
            ValueReadout(title: "Head / layer", value: "\(head + 1) / \(layer + 1)")
            Spacer()
            ValueReadout(title: "Row sum", value: String(format: "%.3f", activeHead.weights.row(r).reduce(0, +)))
        }
    }

    @ViewBuilder private var tensorStage: some View {
        switch stage {
        case .embedding:
            TensorRows(title: "TOKEN EMBEDDING × √8", matrix: trace.sourceEmbedding.scaled, row: row, color: Theme.coral)
            TensorRows(title: "POSITION × \(trace.controls.positionStrength.formatted())", matrix: trace.sourceEmbedding.positions.scaled(by: trace.controls.positionStrength), row: row, color: Theme.violet)
            TensorRows(title: "SUM → FIRST ENCODER", matrix: trace.sourceEmbedding.combined, row: row, color: Theme.mint)
        case .residual:
            TensorRows(title: "INPUT / THE SHORTCUT", matrix: encoder.attentionResidual.input, row: row, color: Theme.coral)
            TensorRows(title: "+ ATTENTION OUTPUT", matrix: encoder.attentionResidual.sublayer, row: row, color: Theme.violet)
            TensorRows(title: "LAYER-NORMALIZED SUM", matrix: encoder.attentionResidual.normalized, row: row, color: Theme.mint)
        case .feedForward:
            TensorRows(title: "W₁ + B₁ · \(TransformerEngine.feedForwardDimension) CHANNELS", matrix: encoder.feedForward.preActivation, row: row, color: Theme.violet)
            TensorRows(title: "RELU · NEGATIVES CLOSED", matrix: encoder.feedForward.relu, row: row, color: Theme.gold)
            TensorRows(title: "W₂ + B₂ · BACK TO 8", matrix: encoder.feedForward.output, row: row, color: Theme.mint)
        case .output:
            SectionEyebrow(text: "After \(trace.decoderTokens.last ?? "<bos>")")
            ForEach(Array(trace.nextTokenCandidates.prefix(6))) { candidate in
                HStack {
                    Text(candidate.token).font(.system(.subheadline, design: .monospaced)).frame(width: 80, alignment: .leading)
                    GeometryReader { g in
                        Capsule().fill(Theme.mint.opacity(0.12))
                        Capsule().fill(Theme.mint).frame(width: max(2, g.size.width * candidate.probability))
                    }.frame(height: 8)
                    Text(String(format: "%.1f%%", candidate.probability * 100)).font(.system(.caption, design: .monospaced))
                }.frame(minHeight: 38).foregroundStyle(Theme.ink)
            }
            Button(endSampled ? "End token sampled" : (trace.decoderTokens.count >= TransformerEngine.maximumTokens ? "Prefix full · edit to continue" : "Sample a token"), systemImage: "dice") {
                let token = trace.sample(unitInterval: Double.random(in: 0..<1))
                if token == "<eos>" { endSampled = true }
                else if token != "<bos>" { target += " " + token; rebuild() }
                sensory.step(haptics: haptics)
            }.buttonStyle(.glass).disabled(isCalculating || endSampled || trace.decoderTokens.count >= TransformerEngine.maximumTokens)
                .accessibilityIdentifier("sample-token")
            Text("Untrained weights: these probabilities demonstrate computation, not meaningful language.")
                .font(.caption).foregroundStyle(Theme.muted)
        default: EmptyView()
        }
    }

    private var stageControls: some View {
        GlassEffectContainer(spacing: 8) {
            HStack(spacing: 8) {
                Button {
                    playing = false
                    stage = MachineStage(rawValue: (stage.rawValue + 6) % 7)!
                } label: { Image(systemName: "backward.end.fill").frame(maxWidth: .infinity, minHeight: 44) }
                    .buttonStyle(.glass).accessibilityLabel("Previous machine stage")
                Button { playing.toggle() } label: {
                    Image(systemName: playing ? "pause.fill" : "play.fill").frame(maxWidth: .infinity, minHeight: 44)
                }.buttonStyle(.glassProminent).accessibilityLabel(playing ? "Pause machine" : "Play machine")
                    .accessibilityIdentifier("play-machine")
                Button {
                    playing = false
                    stage = MachineStage(rawValue: (stage.rawValue + 1) % 7)!
                } label: { Image(systemName: "forward.end.fill").frame(maxWidth: .infinity, minHeight: 44) }
                    .buttonStyle(.glass).accessibilityLabel("Next machine stage").accessibilityIdentifier("next-stage")
                Button { showingInputs = true } label: { Image(systemName: "text.cursor").frame(maxWidth: .infinity, minHeight: 44) }
                    .buttonStyle(.glass).accessibilityLabel("Change input sentences")
                    .accessibilityIdentifier("edit-input")
                Button { showingControls = true } label: { Image(systemName: "slider.horizontal.3").frame(maxWidth: .infinity, minHeight: 44) }
                    .buttonStyle(.glass).accessibilityLabel("Machine controls")
                    .accessibilityIdentifier("machine-controls")
            }
        }
    }

    private var explanation: some View {
        VStack(alignment: .leading, spacing: 12) {
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 6) {
                    ForEach(MachineStage.allCases) { item in
                        Button(item.short) { playing = false; stage = item }
                            .font(.caption.weight(.medium)).padding(.horizontal, 12).frame(minHeight: 44)
                            .background(stage == item ? Theme.mint.opacity(0.13) : Color.clear, in: Capsule())
                            .foregroundStyle(stage == item ? Theme.mint : Theme.muted)
                            .buttonStyle(.plain).accessibilityAddTraits(stage == item ? [.isSelected] : [])
                            .accessibilityIdentifier("stage-\(item.rawValue)")
                    }
                }
            }
            Text(stageDescription).font(.subheadline).foregroundStyle(Theme.ink).fixedSize(horizontal: false, vertical: true)
        }
    }

    private var inputsSheet: some View {
        NavigationStack {
            Form {
                Section("Source tokens / encoder") {
                    TextField("Source sentence", text: $source, axis: .vertical).lineLimit(2...4)
                        .accessibilityIdentifier("source-input")
                }
                Section("Target prefix / decoder") {
                    TextField("Target prefix", text: $target, axis: .vertical).lineLimit(2...4)
                        .accessibilityIdentifier("target-input")
                }
                Section {
                    Text("The decoder shifts the target right and inserts a beginning token. This untrained model uses up to \(TransformerEngine.maximumTokens) source tokens and \(TransformerEngine.maximumTokens - 1) prefix tokens. Target words outside the source and small teaching vocabulary become <unk>.")
                }
            }.navigationTitle("Feed the machine").toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Apply") { endSampled = false; rebuild(); showingInputs = false }.accessibilityIdentifier("apply-input")
                }
            }
        }.presentationDetents([.medium, .large])
    }

    private var controlsSheet: some View {
        NavigationStack {
            Form {
                Section("Look through a head") {
                    Picker("Attention head", selection: $head) {
                        ForEach(0..<controls.headCount, id: \.self) { Text("Head \($0 + 1)").tag($0) }
                    }
                    Picker("Stack layer", selection: $layer) {
                        ForEach(0..<controls.layerCount, id: \.self) { Text("Layer \($0 + 1)").tag($0) }
                    }
                }
                Section("Change the architecture") {
                    Picker("Heads", selection: $controls.headCount) {
                        ForEach([1, 2, 4, 8], id: \.self) { Text("\($0)").tag($0) }
                    }.pickerStyle(.segmented).accessibilityIdentifier("teaching-heads")
                    Stepper("\(controls.layerCount) encoder + decoder layers", value: $controls.layerCount, in: 1...6)
                        .accessibilityIdentifier("teaching-layers")
                    Toggle("Scale scores by √dₖ", isOn: $controls.useScaledAttention)
                    Toggle("Mask future decoder tokens", isOn: $controls.causalMask)
                        .accessibilityIdentifier("causal-mask")
                    VStack(alignment: .leading) {
                        Text("Position strength: \(controls.positionStrength, specifier: "%.2f")")
                        Slider(value: $controls.positionStrength, in: 0...2).accessibilityLabel("Position strength")
                    }
                }
                Section("Explore the distributions") {
                    VStack(alignment: .leading) {
                        Text("Attention temperature: \(controls.attentionTemperature, specifier: "%.2f")")
                        Slider(value: $controls.attentionTemperature, in: 0.1...3).accessibilityLabel("Attention temperature")
                    }
                    VStack(alignment: .leading) {
                        Text("Sampling temperature: \(controls.samplingTemperature, specifier: "%.2f")")
                        Slider(value: $controls.samplingTemperature, in: 0.1...2).accessibilityLabel("Sampling temperature")
                    }
                    Text("Temperatures and position strength are teaching ablations. The original paper uses 8 heads, d = 512 and 6 layers per stack.").font(.caption)
                }
                Button("Restore teaching model") { controls = .init(); head = 0; layer = 0 }
            }.navigationTitle("Machine controls").toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { rebuild(); showingControls = false }.accessibilityIdentifier("done-machine-controls")
                }
            }
        }.presentationDetents([.large])
    }

    private func rebuild(debounced: Bool = false) {
        rebuildTask?.cancel()
        rebuildGeneration += 1
        let generation = rebuildGeneration
        let sourceSnapshot = source
        let targetSnapshot = target
        let controlsSnapshot = controls
        isCalculating = true
        rebuildTask = Task { @MainActor in
            if debounced {
                do { try await Task.sleep(for: .milliseconds(100)) } catch { return }
            }
            guard !Task.isCancelled else { return }
            // Only immutable Sendable inputs cross to the CPU worker. Superseded results never publish.
            let computation = Task.detached(priority: .userInitiated) {
                TransformerEngine().forward(source: sourceSnapshot, target: targetSnapshot, controls: controlsSnapshot)
            }
            let result = await withTaskCancellationHandler {
                await computation.value
            } onCancel: {
                computation.cancel()
            }
            guard !Task.isCancelled, generation == rebuildGeneration else { return }
            trace = result
            head = min(head, result.controls.headCount - 1)
            layer = min(layer, result.controls.layerCount - 1)
            query = min(query, queryTokens.count - 1)
            cellKey = min(cellKey, keyTokens.count - 1)
            isCalculating = false
            rebuildTask = nil
        }
    }
    private func nested(_ matrix: Matrix) -> [[Double]] { (0..<matrix.rows).map { matrix.row($0) } }
}

struct AttentionWires: View {
    let weights: [Double]
    let selected: Int
    var body: some View {
        Canvas { context, size in
            for (i, weight) in weights.enumerated() {
                let x = size.width * (CGFloat(i) + 0.5) / CGFloat(max(1, weights.count))
                var path = Path()
                path.move(to: CGPoint(x: size.width / 2, y: 0))
                path.addCurve(to: CGPoint(x: x, y: size.height), control1: CGPoint(x: size.width / 2, y: size.height * 0.55), control2: CGPoint(x: x, y: size.height * 0.45))
                context.stroke(path, with: .color((selected == i ? Theme.coral : Theme.mint).opacity(0.15 + sqrt(weight) * 0.85)), lineWidth: 0.7 + weight * 5)
                context.fill(Path(ellipseIn: CGRect(x: x - 2, y: size.height - 4, width: 4, height: 4)), with: .color(Theme.mint))
            }
        }.accessibilityHidden(true)
    }
}

struct TinyVector: View {
    let values: [Double]
    var color: Color = Theme.mint
    var body: some View {
        Canvas { context, size in
            let bound = max(values.map(abs).max() ?? 1, 0.001)
            let w = size.width / CGFloat(max(values.count, 1))
            for (i, value) in values.enumerated() {
                let h = abs(value) / bound * size.height / 2
                let rect = CGRect(x: CGFloat(i) * w + 1, y: value >= 0 ? size.height / 2 - h : size.height / 2, width: max(1, w - 2), height: max(1, h))
                context.fill(Path(roundedRect: rect, cornerRadius: 2), with: .color(value >= 0 ? color : Theme.coral.opacity(0.7)))
            }
            var axis = Path(); axis.move(to: CGPoint(x: 0, y: size.height / 2)); axis.addLine(to: CGPoint(x: size.width, y: size.height / 2))
            context.stroke(axis, with: .color(.white.opacity(0.1)), lineWidth: 1)
        }.accessibilityHidden(true)
    }
}

struct TensorRows: View {
    let title: String
    let matrix: Matrix
    let row: Int
    var color: Color = Theme.mint
    var body: some View {
        let values = matrix.row(min(row, matrix.rows - 1))
        VStack(alignment: .leading, spacing: 10) {
            SectionEyebrow(text: title)
            TinyVector(values: values, color: color).frame(height: 50)
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 8) {
                    ForEach(Array(values.enumerated()), id: \.offset) { i, v in
                        VStack(spacing: 4) {
                            Text(String(format: "%+.2f", v)).foregroundStyle(v >= 0 ? color : Theme.coral)
                            Text("\(i)").foregroundStyle(Theme.muted)
                        }.font(.system(.caption2, design: .monospaced))
                    }
                }
            }
        }.padding(.vertical, 8).accessibilityElement(children: .combine)
    }
}
