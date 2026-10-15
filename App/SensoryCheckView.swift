import SwiftUI

struct SensoryCheckView: View {
    let engine: SensoryEngine
    @Environment(\.dismiss) private var dismiss
    @Environment(\.scenePhase) private var scenePhase
    @AppStorage("machine.sound") private var sound = false
    @AppStorage("machine.haptics") private var haptics = true
    @AppStorage("machine.soundGain") private var soundGain = 0.5
    @State private var strength = 0.5
    @State private var sequence: Task<Void, Never>?
    @State private var running = false

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    Text("Sound & touch").font(.system(.title2, design: .rounded).weight(.semibold))
                    Text("A connection’s weight controls the strength of its tap and tone. Pitch follows the key token’s position.")
                        .foregroundStyle(.secondary)
                }
                Section("Your machine") {
                    Toggle("Attention tones", isOn: $sound).accessibilityIdentifier("sound-toggle")
                    Toggle("Haptic feedback", isOn: $haptics).accessibilityIdentifier("haptics-toggle")
                    VStack(alignment: .leading) {
                        Text("Sound level · \(Int(soundGain * 100))%")
                        Slider(value: $soundGain, in: 0...1, step: 0.05)
                            .accessibilityLabel("Sound level").accessibilityIdentifier("sensory-volume")
                            .accessibilityValue(String(format: "%.2f", soundGain))
                    }
                    Text("Tones respect your iPhone’s Silent mode and system volume. These settings apply to the teaching machine and real model; haptics also apply to the paper experiments.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                Section("Feel a connection") {
                    VStack(alignment: .leading) {
                        Text("Attention weight · \(Int(strength * 100))%")
                        Slider(value: $strength, in: 0...1, step: 0.05)
                            .accessibilityLabel("Attention weight")
                            .onChange(of: strength) { _, value in pulse(value) }
                    }
                    HStack(spacing: 12) {
                        Button("Zero") { pulse(0) }.accessibilityIdentifier("sensory-zero")
                        Button("Quiet") { pulse(0.05) }.accessibilityIdentifier("sensory-weak")
                        Button("Strong") { pulse(0.85) }.accessibilityIdentifier("sensory-strong")
                    }.buttonStyle(.bordered).frame(minHeight: 44)
                    Text("Zero weight gives only the small selection cue. It has no connection tone or impact.")
                        .font(.caption).foregroundStyle(.secondary)
                    Button(running ? "Playing five connections…" : "Play five connections", systemImage: "waveform") {
                        playSequence()
                    }.disabled(running).accessibilityIdentifier("sensory-sequence")
                    Button("Stop sound & sequence", systemImage: "stop.fill") { stop() }
                        .accessibilityIdentifier("sensory-stop")
                }
                Section("Playback status") {
                    TimelineView(.periodic(from: .now, by: 0.5)) { _ in
                        let info = engine.diagnostics
                        VStack(alignment: .leading, spacing: 8) {
                            Text(info.status).accessibilityIdentifier("sensory-status")
                            Text("\(info.playingVoiceCount) playing voices · \(info.cachedVoiceCount)/\(info.maxCachedVoiceCount) reusable voices")
                                .accessibilityIdentifier("sensory-voices")
                            Text("\(info.emittedHapticCount) haptic requests · \(info.playedToneCount) accepted tone starts")
                                .accessibilityIdentifier("sensory-diagnostics")
                                .accessibilityValue("playing=\(info.playingVoiceCount);haptics=\(info.emittedHapticCount);tones=\(info.playedToneCount)")
                            Text("Output: \(info.route) · system volume \(Int(info.outputVolume * 100))%")
                            Text(info.hapticsSupported ? "iPhone haptic hardware available" : "Haptic hardware unavailable")
                            if let error = info.lastError { Text(error).foregroundStyle(Theme.coral) }
                        }.font(.caption).foregroundStyle(.secondary)
                    }
                }
            }.navigationTitle("Sound & touch").navigationBarTitleDisplayMode(.inline)
                .toolbar {
                    ToolbarItem(placement: .confirmationAction) {
                        Button("Done") { stop(); dismiss() }.accessibilityIdentifier("sensory-done")
                    }
                }
        }
        .task { engine.prepare(haptics: haptics) }
        .onChange(of: sound) { _, enabled in if !enabled { stop() } }
        .onChange(of: scenePhase) { _, phase in if phase != .active { stop() } }
        .onDisappear { stop() }
    }

    private func pulse(_ weight: Double, pitch: Int = 6) {
        engine.touch(weight: weight, pitch: pitch, sound: sound, haptics: haptics, soundGain: soundGain)
    }

    private func playSequence() {
        stop()
        running = true
        sequence = Task { @MainActor in
            for (index, weight) in [0.05, 0.2, 0.4, 0.65, 0.85].enumerated() {
                guard !Task.isCancelled else { return }
                pulse(weight, pitch: index * 3)
                do { try await Task.sleep(for: .milliseconds(320)) } catch { return }
            }
            running = false
            sequence = nil
        }
    }

    private func stop() {
        sequence?.cancel()
        sequence = nil
        running = false
        engine.stop()
    }
}
