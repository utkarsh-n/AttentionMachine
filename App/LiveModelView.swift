import SwiftUI

struct LiveModelView: View {
    let engine: LocalModelEngine
    @AppStorage("machine.sound") private var sound = false
    @AppStorage("machine.haptics") private var haptics = true
    @AppStorage("machine.soundGain") private var soundGain = 0.5
    @State private var recording = LocalModelTrace.recorded()
    @State private var prompt = "The animal did not cross the street because it was tired. What was tired?"
    @State private var layer = 17
    @State private var head = 0
    @State private var query = 0
    @State private var key = 0
    @State private var sensory = SensoryEngine()
    @State private var showConnections = false
    @State private var initializedSelection = false
    private var trace: LocalModelTrace? { engine.trace ?? recording }

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 24) {
                    header
                    modelControls
                    if let trace { traceView(trace) }
                    architectureNote
                }.padding(20).padding(.bottom, 20)
            }
            .background { Theme.bg.ignoresSafeArea(); GridBackdrop().ignoresSafeArea() }
            .toolbar(.hidden, for: .navigationBar)
            .onChange(of: engine.trace?.capturedAt) { _, _ in
                if let trace = engine.trace { query = max(0, trace.tokens.count - 1); key = query }
            }
            .onDisappear { sensory.stop() }
            .task {
                sensory.prepare(haptics: haptics)
                if !initializedSelection {
                    if let trace { query = max(0, trace.tokens.count - 1); key = query }
                    initializedSelection = true
                }
                if ProcessInfo.processInfo.arguments.contains("--verify-local-model"), DeviceValidationRunGate.begin() {
                    await engine.load()
                    guard !Task.isCancelled else { return }
                    if engine.isReady {
                        await engine.inspect(prompt: prompt, layer: layer, head: head, maxNewTokens: 16)
                    }
                    guard !Task.isCancelled else { return }
                    await saveDeviceValidation()
                }
            }
        }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 9) {
            SectionEyebrow(text: "Theory → measured computation")
            Text("Inside a real model").font(.system(.largeTitle, design: .rounded).weight(.semibold)).foregroundStyle(Theme.ink)
            Text("Download once. Generate and inspect attention entirely on your iPhone.")
                .font(.subheadline).foregroundStyle(Theme.muted)
        }.padding(.top, 8)
    }

    private var modelControls: some View {
        InstrumentPanel {
            VStack(alignment: .leading, spacing: 17) {
                HStack {
                    Image(systemName: "cpu").foregroundStyle(Theme.mint)
                    Text("Qwen3 · 4B Instruct").font(.headline).foregroundStyle(Theme.ink)
                    Spacer()
                    Text("4 BIT").font(.system(.caption2, design: .monospaced)).foregroundStyle(Theme.mint)
                }
                Text(engine.status).font(.caption).foregroundStyle(Theme.muted).fixedSize(horizontal: false, vertical: true)
                if engine.isWorking {
                    ProgressView(value: engine.progress).tint(Theme.mint)
                    Button("Cancel", systemImage: "stop.fill") { engine.cancel() }.buttonStyle(.glass)
                } else if !engine.isReady {
                    Button(engine.hasDownloadedModel ? "Load local model" : "Download model · about 2.3 GB", systemImage: "arrow.down.circle") {
                        Task {
                            if engine.hasDownloadedModel { await engine.load() }
                            else { await engine.download() }
                        }
                    }.buttonStyle(.glassProminent).accessibilityIdentifier("download-model")
                    Text("Weights stay in this app. No account or cloud inference is required.")
                        .font(.caption).foregroundStyle(Theme.muted)
                } else {
                    TextField("Prompt", text: $prompt, axis: .vertical)
                        .lineLimit(2...5).padding(12).background(Theme.bg, in: RoundedRectangle(cornerRadius: 12))
                        .foregroundStyle(Theme.ink).accessibilityIdentifier("model-prompt")
                    HStack {
                        Picker("Layer", selection: $layer) {
                            ForEach(0..<36, id: \.self) { Text("Layer \($0 + 1)").tag($0) }
                        }
                        Picker("Head", selection: $head) {
                            ForEach(0..<32, id: \.self) { Text("Head \($0 + 1)").tag($0) }
                        }
                    }.font(.caption)
                    Button("Generate & inspect", systemImage: "sparkles") {
                        Task { await engine.inspect(prompt: prompt, layer: layer, head: head, maxNewTokens: 32) }
                    }.buttonStyle(.glassProminent).disabled(prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                        .accessibilityIdentifier("inspect-model")
                    Button("Unload to free memory", systemImage: "memorychip") { Task { await engine.unload() } }
                        .font(.caption).buttonStyle(.glass)
                }
                if let error = engine.error {
                    Label(error, systemImage: "exclamationmark.circle").font(.caption).foregroundStyle(Theme.coral)
                        .fixedSize(horizontal: false, vertical: true).accessibilityIdentifier("model-error")
                }
            }
        }
    }

    private func traceView(_ trace: LocalModelTrace) -> some View {
        VStack(alignment: .leading, spacing: 18) {
            HStack(spacing: 7) {
                Circle().fill(trace.mode == "live" ? Theme.mint : Theme.gold).frame(width: 6, height: 6)
                Text(trace.mode == "live" ? "MEASURED ON THIS IPHONE" : "RECORDED REAL MODEL TRACE")
                    .font(.system(.caption2, design: .monospaced)).tracking(1)
                    .accessibilityIdentifier("model-trace-mode")
            }.foregroundStyle(trace.mode == "live" ? Theme.mint : Theme.gold)
            Text(trace.prompt).font(.subheadline).foregroundStyle(Theme.ink).fixedSize(horizontal: false, vertical: true)
            TokenStrip(tokens: trace.tokens.map(\.text), selected: $query, tint: Theme.coral) { _ in sensory.step(haptics: haptics) }
            InstrumentPanel {
                VStack(alignment: .leading, spacing: 16) {
                    HStack {
                        Text("Measured attention").font(.headline)
                        Spacer()
                        Text("L\(trace.layer + 1) / H\(trace.head + 1)")
                            .font(.system(.caption, design: .monospaced)).foregroundStyle(Theme.mint)
                    }.foregroundStyle(Theme.ink)
                    HeatmapView(matrix: trace.attention, tokens: trace.tokens.map(\.text), selectedRow: $query, causal: true) { _, column, value in
                        key = column; sensory.touch(weight: value, pitch: Int((Double(column) * 23 / Double(max(1, trace.tokens.count - 1))).rounded()), sound: sound, haptics: haptics, soundGain: soundGain)
                    }
                    if trace.attention.indices.contains(query), trace.attention[query].indices.contains(key), trace.tokens.indices.contains(key) {
                        HStack {
                            ValueReadout(title: "Query token", value: "\(query + 1)", color: Theme.coral)
                                .accessibilityIdentifier("model-query-readout")
                            Spacer()
                            ValueReadout(title: "Key: \(trace.tokens[key].text)", value: String(format: "%.2f%%", trace.attention[query][key] * 100), color: Theme.violet)
                        }
                    }
                    Button(showConnections ? "Hide connections" : "Read this attention row", systemImage: "list.bullet") {
                        showConnections.toggle()
                    }.buttonStyle(.glass)
                    if showConnections, trace.attention.indices.contains(query) {
                        connectionList(trace)
                    }
                    Text(trace.mode == "recorded" ? "This capture is fixed to the prompt, layer and head shown. Run the downloaded model to measure your own input." : "Rows are queries, columns are keys. Scores use the model’s actual projections, Q/K normalization, rotary positions and causal mask.")
                        .font(.caption).foregroundStyle(Theme.muted).fixedSize(horizontal: false, vertical: true)
                        .accessibilityIdentifier("model-trace-note")
                }
            }
            InstrumentPanel {
                VStack(alignment: .leading, spacing: 14) {
                    SectionEyebrow(text: "Model continuation")
                    Text(trace.generation.isEmpty ? "No continuation in this capture." : trace.generation)
                        .font(.body).foregroundStyle(Theme.ink).textSelection(.enabled)
                    Divider().overlay(Theme.muted.opacity(0.2))
                    SectionEyebrow(text: "Next-token probabilities / original prompt")
                    ForEach(Array(trace.nextTokens.prefix(5).enumerated()), id: \.offset) { _, token in
                        HStack {
                            Text(token.text.replacingOccurrences(of: "\n", with: "↵"))
                                .font(.system(.caption, design: .monospaced)).lineLimit(1)
                            Spacer()
                            Text(String(format: "%.2f%%", token.probability * 100)).font(.system(.caption, design: .monospaced))
                                .foregroundStyle(Theme.mint)
                        }.foregroundStyle(Theme.ink)
                    }
                }
            }
            Text("\(trace.model.id) · \(trace.model.quantization)\nCaptured \(trace.capturedAt)")
                .font(.caption2).foregroundStyle(Theme.muted).textSelection(.enabled)
        }
    }

    private func connectionList(_ trace: LocalModelTrace) -> some View {
        let values = trace.attention[query].enumerated().sorted { $0.element > $1.element }.prefix(8)
        return VStack(spacing: 7) {
            ForEach(Array(values), id: \.offset) { entry in
                HStack {
                    Text("\(entry.offset + 1) · \(trace.tokens[entry.offset].text)").font(.system(.caption, design: .monospaced))
                    Spacer()
                    Text(String(format: "%.2f%%", entry.element * 100)).font(.system(.caption, design: .monospaced))
                }.foregroundStyle(Theme.ink).frame(minHeight: 30).accessibilityElement(children: .combine)
            }
        }
    }

    private var architectureNote: some View {
        VStack(alignment: .leading, spacing: 12) {
            SectionEyebrow(text: "From 2017 to today")
            Text("The same attention equation, a different architecture.").font(.headline).foregroundStyle(Theme.ink)
            Text("Qwen is a trained decoder-only model. It uses rotary positions, RMSNorm before sublayers, grouped query attention, and a gated feed-forward network. The Machine tab shows the original paper’s encoder–decoder, sine/cosine positions, post-layer normalization and ReLU.")
                .font(.subheadline).foregroundStyle(Theme.muted).fixedSize(horizontal: false, vertical: true)
            Text("Attention weights describe a computation, not a complete explanation of a model’s reasoning.")
                .font(.caption).foregroundStyle(Theme.muted)
            HStack {
                Link("Model card", destination: URL(string: "https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507")!)
                Spacer()
                Link("MLX runtime", destination: URL(string: "https://github.com/ml-explore/mlx-swift-lm")!)
            }.font(.caption).foregroundStyle(Theme.mint).frame(minHeight: 44)
        }
    }

    private func saveDeviceValidation() async {
        let directory = URL.documentsDirectory
        if let trace = engine.trace, let data = try? JSONEncoder().encode(trace) {
            try? data.write(to: directory.appendingPathComponent("DeviceValidation.json"), options: .atomic)
            print("ATTENTION_DEVICE_VALIDATION tokens=\(trace.tokens.count) elapsed=\(trace.metrics.elapsedSeconds) rowError=\(trace.metrics.rowSumMaxError) futureWeight=\(trace.metrics.maskedFutureMax) generation=\(trace.generation)")
        } else {
            let error = engine.error ?? "No measured trace"
            try? Data(error.utf8).write(to: directory.appendingPathComponent("DeviceValidationError.txt"), options: .atomic)
            print("ATTENTION_DEVICE_VALIDATION_ERROR \(error)")
        }
    }
}

/// Validation arguments request one experiment per launch, even if tab appearance restarts `.task`.
@MainActor private enum DeviceValidationRunGate {
    private static var started = false

    static func begin() -> Bool {
        guard !started else { return false }
        started = true
        return true
    }
}
