import SwiftUI
import OfflineIntelligence

struct LocalIntelligenceView: View {
    let intelligence: AttentionLocalIntelligence

    var body: some View {
        VStack(spacing: 0) {
            Picker("Local model task", selection: Binding(
                get: { intelligence.mode }, set: { intelligence.select($0) })) {
                ForEach(AttentionLocalIntelligence.Mode.allCases) { mode in Text(mode.rawValue).tag(mode) }
            }
            .pickerStyle(.segmented)
            .padding(.horizontal, 20).padding(.top, 12).padding(.bottom, 8)
            .disabled(intelligence.isSwitching)
            .accessibilityIdentifier("local-model-mode")
            if intelligence.isSwitching {
                HStack {
                    ProgressView().tint(Theme.mint)
                    Text("Releasing the current model…").font(.caption).foregroundStyle(Theme.muted)
                }.padding(.bottom, 8).accessibilityElement(children: .combine)
            }
            Group {
                if intelligence.mode == .assistant { AttentionAssistantView(assistant: intelligence.assistant) }
                else { LiveModelView(engine: intelligence.inspection) }
            }.disabled(intelligence.isSwitching)
        }.background(Theme.bg)
    }
}

private struct AttentionAssistantView: View {
    let assistant: AttentionTextAssistant
    @State private var prompt = "Explain why attention scores are divided by the square root of the key dimension, using a small numerical example."

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 24) {
                    VStack(alignment: .leading, spacing: 9) {
                        SectionEyebrow(text: "Paper → understanding")
                        Text("Ask the paper")
                            .font(.system(.largeTitle, design: .rounded).weight(.semibold)).foregroundStyle(Theme.ink)
                        Text("Explain a concept with a model running entirely on your iPhone.")
                            .font(.subheadline).foregroundStyle(Theme.muted)
                    }.padding(.top, 8)
                    controls
                    if !assistant.output.isEmpty { reply }
                    Text("These are generated text replies. Switch to Attention trace to measure real Q/K attention in the teaching model.")
                        .font(.caption).foregroundStyle(Theme.muted).fixedSize(horizontal: false, vertical: true)
                }.padding(20).padding(.bottom, 20)
            }
            .background { Theme.bg.ignoresSafeArea(); GridBackdrop().ignoresSafeArea() }
            .toolbar(.hidden, for: .navigationBar)
        }
    }

    private var controls: some View {
        InstrumentPanel {
            VStack(alignment: .leading, spacing: 16) {
                Picker("Text model", selection: Binding(get: { assistant.choice }, set: { assistant.select($0) })) {
                    ForEach(OfflineModelChoice.allCases) { choice in Text(choice.title).tag(choice) }
                }.pickerStyle(.segmented).disabled(assistant.isWorking)
                    .accessibilityIdentifier("assistant-model-choice")
                Text(assistant.choice.detail).font(.caption).foregroundStyle(Theme.muted)
                TextField("Question about attention", text: $prompt, axis: .vertical)
                    .lineLimit(3...6).padding(12).background(Theme.bg, in: RoundedRectangle(cornerRadius: 12))
                    .foregroundStyle(Theme.ink).disabled(assistant.isWorking)
                    .accessibilityIdentifier("assistant-prompt")
                Text(assistant.status).font(.caption).foregroundStyle(Theme.muted)
                    .fixedSize(horizontal: false, vertical: true).accessibilityIdentifier("assistant-status")
                if assistant.isWorking {
                    HStack {
                        ProgressView().tint(Theme.mint)
                        Button("Stop", systemImage: "stop.fill") { assistant.cancel() }.buttonStyle(.glass)
                            .accessibilityIdentifier("assistant-stop")
                    }
                } else {
                    Button("Ask on this iPhone", systemImage: "sparkles") { assistant.generate(prompt: prompt) }
                        .buttonStyle(.glassProminent)
                        .disabled(prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                        .accessibilityIdentifier("assistant-generate")
                    if assistant.isReady {
                        Button("Unload to free memory", systemImage: "memorychip") { Task { await assistant.unload() } }
                            .font(.caption).buttonStyle(.glass).accessibilityIdentifier("assistant-unload")
                    }
                }
                if let error = assistant.error {
                    Label(error, systemImage: "exclamationmark.circle")
                        .font(.caption).foregroundStyle(Theme.coral).fixedSize(horizontal: false, vertical: true)
                        .accessibilityIdentifier("assistant-error")
                }
            }
        }
    }

    private var reply: some View {
        InstrumentPanel {
            VStack(alignment: .leading, spacing: 16) {
                SectionEyebrow(text: "Generated on this iPhone")
                Text(assistant.output).font(.body).foregroundStyle(Theme.ink).textSelection(.enabled)
                    .accessibilityIdentifier("assistant-output")
                if let metrics = assistant.metrics {
                    Divider().overlay(Theme.muted.opacity(0.2))
                    HStack {
                        ValueReadout(title: "First token", value: metrics.firstTokenSeconds.map { String(format: "%.2f s", $0) } ?? "—")
                        Spacer()
                        ValueReadout(title: "Decode", value: String(format: "%.1f tok/s", metrics.tokensPerSecond))
                    }
                    Text("\(metrics.promptTokens) input tokens · \(metrics.generatedTokens) output tokens · \(assistant.choice.title)")
                        .font(.caption2).foregroundStyle(Theme.muted)
                }
            }
        }
    }
}
