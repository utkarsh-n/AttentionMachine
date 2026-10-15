import UIKit
import AVFoundation
import CoreHaptics
import Observation

struct SensoryDiagnostics: Equatable, Sendable {
    let route: String
    let outputVolume: Double
    let hapticsSupported: Bool
    let isOtherAudioPlaying: Bool
    let cachedVoiceCount: Int
    let playingVoiceCount: Int
    let maxCachedVoiceCount: Int
    let lastPlaySucceeded: Bool?
    let lastError: String?
    let status: String
    let emittedHapticCount: Int
    let playedToneCount: Int
    let suppressedTouchCount: Int
}

@Observable @MainActor
final class SensoryEngine {
    private(set) var lastPlaySucceeded: Bool?
    private(set) var lastError: String?
    private(set) var status = "Ready · optional sound respects Silent Mode"
    private(set) var emittedHapticCount = 0
    private(set) var playedToneCount = 0
    private(set) var suppressedTouchCount = 0

    @ObservationIgnored private let engineID = UUID()
    @ObservationIgnored private let selection = UISelectionFeedbackGenerator()
    @ObservationIgnored private let impact = UIImpactFeedbackGenerator(style: .soft)
    @ObservationIgnored private let hapticsSupported = CHHapticEngine.capabilitiesForHardware().supportsHaptics
    @ObservationIgnored private var lastHapticUptime = -Double.infinity
    @ObservationIgnored private var lastDiagnosticsUptime = -Double.infinity
    @ObservationIgnored private var toneTask: Task<Void, Never>?
    @ObservationIgnored private var toneToken: AudioNoteToken?
    @ObservationIgnored private var toneGeneration: UInt64 = 0
    @ObservationIgnored private var diagnosticsTask: Task<Void, Never>?
    @ObservationIgnored private var lastSnapshotRevision: UInt64 = 0
    private var audioSnapshot = AudioSnapshot.empty

    init() {
        SensoryAudioSession.shared.register(self)
        refreshDiagnostics()
        prepare()
    }

    deinit {
        toneToken?.cancel()
        toneTask?.cancel()
        diagnosticsTask?.cancel()
        let id = engineID
        Task { await SensoryAudioBackend.shared.release(engineID: id) }
    }

    var diagnostics: SensoryDiagnostics {
        // Audio properties are sampled on the worker; this getter never acquires audio hardware.
        refreshDiagnostics()
        return SensoryDiagnostics(route: audioSnapshot.route, outputVolume: audioSnapshot.outputVolume,
            hapticsSupported: hapticsSupported, isOtherAudioPlaying: audioSnapshot.isOtherAudioPlaying,
            cachedVoiceCount: audioSnapshot.cachedVoiceCount, playingVoiceCount: audioSnapshot.playingVoiceCount,
            maxCachedVoiceCount: 48, lastPlaySucceeded: lastPlaySucceeded, lastError: lastError,
            status: status, emittedHapticCount: emittedHapticCount, playedToneCount: playedToneCount,
            suppressedTouchCount: suppressedTouchCount)
    }

    func prepare(haptics: Bool = true) {
        guard haptics, UIApplication.shared.applicationState == .active else { return }
        audioPhase("haptic.prepare.begin")
        selection.prepare()
        impact.prepare()
        audioPhase("haptic.prepare.end")
    }

    func touch(weight: Double = 0.5, pitch: Int = 0, sound: Bool = false,
               haptics: Bool = true, soundGain: Double = 0.5) {
        guard UIApplication.shared.applicationState == .active else { return }
        let strength = SensoryTone.clampedUnit(weight)
        lastError = nil
        lastPlaySucceeded = nil
        if strength == 0 {
            cancelPendingTone()
            SensoryAudioSession.shared.deactivateIfIdle()
            step(haptics: haptics)
            status = "Zero attention · selection cue only · no tone"
            return
        }
        if haptics {
            if allowHaptic() {
                audioPhase("haptic.impact.begin")
                impact.impactOccurred(intensity: CGFloat(SensoryTone.hapticIntensity(weight: strength)))
                audioPhase("haptic.impact.end")
                audioPhase("haptic.reprepare.begin")
                impact.prepare()
                audioPhase("haptic.reprepare.end")
                emittedHapticCount += 1
            } else {
                suppressedTouchCount += 1
            }
        }
        status = haptics ? "Weighted haptic requested" : "Haptics disabled"
        if sound { play(pitch: pitch, strength: strength, gain: soundGain) }
        else {
            cancelPendingTone()
            SensoryAudioSession.shared.deactivateIfIdle()
        }
    }

    func step(haptics: Bool = true) {
        guard haptics, UIApplication.shared.applicationState == .active else { return }
        guard allowHaptic() else { suppressedTouchCount += 1; return }
        audioPhase("haptic.selection.begin")
        selection.selectionChanged()
        selection.prepare()
        audioPhase("haptic.selection.end")
        emittedHapticCount += 1
        status = "Selection haptic requested"
    }

    func stop() {
        stopLocal(status: "Stopped · no tones playing")
        SensoryAudioSession.shared.stop(engineID: engineID)
        refreshDiagnostics(force: true)
    }

    fileprivate func stopLocal(status: String) {
        // Invalidate synchronously before any worker hop. Do not make UIKit wait for player.stop().
        cancelPendingTone()
        lastHapticUptime = -Double.infinity
        self.status = status
    }

    private func cancelPendingTone() {
        toneGeneration &+= 1
        toneToken?.cancel()
        toneToken = nil
        toneTask?.cancel()
        toneTask = nil
    }

    private func allowHaptic() -> Bool {
        let now = ProcessInfo.processInfo.systemUptime
        guard now - lastHapticUptime >= 0.06 else { return false }
        lastHapticUptime = now
        return true
    }

    private func refreshDiagnostics(force: Bool = false) {
        let now = ProcessInfo.processInfo.systemUptime
        guard diagnosticsTask == nil, force || now - lastDiagnosticsUptime >= 0.25 else { return }
        lastDiagnosticsUptime = now
        let id = engineID
        diagnosticsTask = Task { @MainActor [weak self] in
            let snapshot = await SensoryAudioSession.shared.snapshot(engineID: id)
            guard let self, !Task.isCancelled else { return }
            self.apply(snapshot)
            self.diagnosticsTask = nil
        }
    }

    private func apply(_ snapshot: AudioSnapshot) {
        guard snapshot.revision >= lastSnapshotRevision else { return }
        lastSnapshotRevision = snapshot.revision
        if audioSnapshot != snapshot { audioSnapshot = snapshot }
    }

    private func play(pitch: Int, strength: Double, gain: Double) {
        cancelPendingTone()
        let gain = SensoryTone.clampedUnit(gain)
        guard gain > 0 else {
            status = "Sound gain is zero · no tone"
            SensoryAudioSession.shared.deactivateIfIdle()
            return
        }
        let generation = toneGeneration
        let id = engineID
        let token = AudioNoteToken()
        toneToken = token
        toneTask = Task { @MainActor [weak self] in
            let report = await SensoryAudioSession.shared.play(engineID: id, token: token,
                pitch: pitch, strength: strength, gain: gain)
            guard let self, !Task.isCancelled, !token.isCancelled,
                  generation == self.toneGeneration,
                  UIApplication.shared.applicationState == .active else { return }
            self.toneTask = nil
            self.toneToken = nil
            self.apply(report.snapshot)
            switch report.outcome {
            case .started(let frequency):
                self.lastPlaySucceeded = true
                self.playedToneCount += 1
                self.status = "Tone requested · \(frequency) Hz"
            case .throttled:
                self.suppressedTouchCount += 1
                self.status = "Rapid tone request throttled"
            case .ringing:
                self.suppressedTouchCount += 1
                self.status = "Tone still ringing · new onset skipped"
            case .unavailable(let message):
                self.lastPlaySucceeded = false
                self.lastError = message
                self.status = "Audio unavailable"
            case .creationFailed(let message):
                self.lastPlaySucceeded = false
                self.lastError = message
                self.status = "Could not create tone player"
            case .playDeclined:
                self.lastPlaySucceeded = false
                self.lastError = "AVAudioPlayer declined playback. Check interruption state and output route."
                self.status = "Audio playback did not start"
            case .canceled:
                break
            }
        }
    }
}

private struct AudioSnapshot: Equatable, Sendable {
    let revision: UInt64
    let route: String
    let outputVolume: Double
    let isOtherAudioPlaying: Bool
    let cachedVoiceCount: Int
    let playingVoiceCount: Int
    static let empty = AudioSnapshot(revision: 0, route: "Checking audio route", outputVolume: 0,
        isOtherAudioPlaying: false, cachedVoiceCount: 0, playingVoiceCount: 0)

    static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.route == rhs.route && lhs.outputVolume == rhs.outputVolume &&
        lhs.isOtherAudioPlaying == rhs.isOtherAudioPlaying &&
        lhs.cachedVoiceCount == rhs.cachedVoiceCount && lhs.playingVoiceCount == rhs.playingVoiceCount
    }
}

private struct AudioPlaybackReport: Sendable {
    enum Outcome: Sendable {
        case started(Int), throttled, ringing, unavailable(String), creationFailed(String), playDeclined, canceled
    }
    let outcome: Outcome
    let snapshot: AudioSnapshot
}

/// The only unchecked state is a small lock-protected cancellation/eligibility flag.
/// No lock spans an audio operation, so Stop never waits for synchronous hardware work.
private final class AudioNoteToken: @unchecked Sendable {
    private let lock = NSLock()
    private var canceled = false
    var isCancelled: Bool { lock.withLock { canceled } }
    func cancel() { lock.withLock { canceled = true } }
}

private final class AudioEligibility: @unchecked Sendable {
    static let shared = AudioEligibility()
    private let lock = NSLock()
    private var appActive = false
    private var interrupted = false
    var isAllowed: Bool { lock.withLock { appActive && !interrupted } }
    func setAppActive(_ value: Bool) { lock.withLock { appActive = value } }
    func setInterrupted(_ value: Bool) { lock.withLock { interrupted = value } }
}

/// Explicit dispatch executor: default actor isolation alone does not promise a non-main thread.
private final class AudioSerialExecutor: SerialExecutor {
    private let queue = DispatchQueue(label: "com.utkarsh.attentionmachine.audio", qos: .userInitiated)
    func enqueue(_ job: consuming ExecutorJob) {
        let job = UnownedJob(job)
        let executor = asUnownedSerialExecutor()
        queue.async {
            dispatchPrecondition(condition: .onQueue(self.queue))
            job.runSynchronously(on: executor)
        }
    }
}

/// All session and AVAudioPlayer access is confined here, including isPlaying and currentTime.
/// Players never cross the actor boundary; the main actor receives only Sendable value snapshots.
private actor SensoryAudioBackend {
    static let shared = SensoryAudioBackend()
    nonisolated private let executor = AudioSerialExecutor()
    nonisolated var unownedExecutor: UnownedSerialExecutor { executor.asUnownedSerialExecutor() }
    private final class Voice {
        let player: AVAudioPlayer
        init(_ player: AVAudioPlayer) { self.player = player }
    }
    private enum ActivationResult: Sendable { case ready, superseded, unavailable(String) }
    private var voices: [UUID: [Int: [Voice]]] = [:]
    private var pending: [UUID: AudioNoteToken] = [:]
    private var configured = false
    private var active = false
    private var lastSoundUptime = -Double.infinity
    private var activationTask: Task<ActivationResult, Never>?
    private var deactivationTask: Task<Void, Never>?
    private var idleTask: Task<Void, Never>?
    private var lifecycleGeneration: UInt64 = 0
    private var snapshotRevision: UInt64 = 0

    func play(engineID: UUID, token: AudioNoteToken, pitch: Int, strength: Double, gain: Double) async -> AudioPlaybackReport {
        assert(!Thread.isMainThread)
        guard !token.isCancelled, !Task.isCancelled else { return report(.canceled, engineID: engineID) }
        guard AudioEligibility.shared.isAllowed else {
            return report(.unavailable("The app is inactive or an audio interruption is active. Tap again after it ends."), engineID: engineID)
        }
        let now = ProcessInfo.processInfo.systemUptime
        guard now - lastSoundUptime >= 0.06 else { return report(.throttled, engineID: engineID) }
        lastSoundUptime = now
        pending[engineID] = token
        defer {
            if pending[engineID] === token { pending.removeValue(forKey: engineID) }
            deactivateIfIdle()
        }
        let activation = await beginPlayback(token: token)
        guard canPlay(token) else { return report(.canceled, engineID: engineID) }
        switch activation {
        case .unavailable(let message): return report(.unavailable(message), engineID: engineID)
        case .superseded: return report(.canceled, engineID: engineID)
        case .ready: break
        }
        let note = max(0, min(23, pitch))
        do {
            var bank = voices[engineID]?[note] ?? []
            let voice: Voice
            if let idle = bank.first(where: { !$0.player.isPlaying }) {
                voice = idle
            } else if bank.count < 2 {
                audioPhase("player.init.begin")
                voice = Voice(try AVAudioPlayer(data: SensoryTone.wav(pitch: note)))
                audioPhase("player.init.end")
                audioPhase("player.prepare.begin")
                voice.player.prepareToPlay()
                audioPhase("player.prepare.end")
                bank.append(voice)
                voices[engineID, default: [:]][note] = bank
            } else {
                // Never seek a sounding waveform back to its start: that creates a discontinuity.
                return report(.ringing, engineID: engineID)
            }
            guard canPlay(token) else {
                stop(voice.player)
                return report(.canceled, engineID: engineID)
            }
            voice.player.volume = Float(SensoryTone.playbackVolume(weight: strength, gain: gain))
            voice.player.currentTime = 0
            guard canPlay(token) else {
                stop(voice.player)
                return report(.canceled, engineID: engineID)
            }
            audioPhase("player.play.begin")
            let success = voice.player.play()
            audioPhase("player.play.end")
            // Stop can invalidate the request during synchronous play; stop that voice before publishing.
            guard canPlay(token) else {
                stop(voice.player)
                return report(.canceled, engineID: engineID)
            }
            if success {
                scheduleIdleCleanup()
                return report(.started(Int(SensoryTone.frequency(pitch: note).rounded())), engineID: engineID)
            }
            return report(.playDeclined, engineID: engineID)
        } catch {
            return report(.creationFailed(error.localizedDescription), engineID: engineID)
        }
    }

    func snapshot(engineID: UUID) -> AudioSnapshot {
        assert(!Thread.isMainThread)
        let session = AVAudioSession.sharedInstance()
        let outputs = session.currentRoute.outputs
        let route = outputs.isEmpty ? "No output route" : outputs.map { "\($0.portName) (\($0.portType.rawValue))" }.joined(separator: ", ")
        let bank = voices[engineID] ?? [:]
        snapshotRevision &+= 1
        return AudioSnapshot(revision: snapshotRevision, route: route, outputVolume: Double(session.outputVolume),
            isOtherAudioPlaying: session.isOtherAudioPlaying,
            cachedVoiceCount: bank.values.reduce(0) { $0 + $1.count },
            playingVoiceCount: bank.values.flatMap { $0 }.filter { $0.player.isPlaying }.count)
    }

    func stop(engineID: UUID, discardPlayers: Bool = false) {
        pending.removeValue(forKey: engineID)?.cancel()
        for voice in (voices[engineID] ?? [:]).values.flatMap({ $0 }) { stop(voice.player) }
        if discardPlayers { voices.removeValue(forKey: engineID) }
        deactivateIfIdle()
    }

    func release(engineID: UUID) { stop(engineID: engineID, discardPlayers: true) }

    func stopAll(discardPlayers: Bool, sessionInactive: Bool = false, resetSession: Bool = false) {
        lifecycleGeneration &+= 1
        for token in pending.values { token.cancel() }
        pending.removeAll()
        idleTask?.cancel()
        idleTask = nil
        for voice in voices.values.flatMap({ $0.values.flatMap { $0 } }) { stop(voice.player) }
        if discardPlayers { voices.removeAll() }
        if sessionInactive || resetSession { active = false }
        if resetSession { configured = false }
        lastSoundUptime = -Double.infinity
        deactivateIfIdle()
    }

    func deactivateIfIdle() { requestDeactivation(ignoringPendingNotes: false) }

    private func report(_ outcome: AudioPlaybackReport.Outcome, engineID: UUID) -> AudioPlaybackReport {
        AudioPlaybackReport(outcome: outcome, snapshot: snapshot(engineID: engineID))
    }

    private func canPlay(_ token: AudioNoteToken) -> Bool {
        !token.isCancelled && !Task.isCancelled && AudioEligibility.shared.isAllowed
    }

    private func stop(_ player: AVAudioPlayer) {
        assert(!Thread.isMainThread)
        audioPhase("player.stop.begin")
        player.stop()
        player.currentTime = 0
        audioPhase("player.stop.end")
    }

    private func beginPlayback(token: AudioNoteToken) async -> ActivationResult {
        while canPlay(token) {
            if let deactivationTask { await deactivationTask.value; continue }
            if let activationTask {
                switch await activationTask.value {
                case .ready, .superseded: continue
                case .unavailable(let message): return .unavailable(message)
                }
            }
            if active, configured { return .ready }
            startActivation()
        }
        return .superseded
    }

    private func startActivation() {
        assert(!Thread.isMainThread)
        let generation = lifecycleGeneration
        // The native transition has no cancellation contract; it outlives the requesting note.
        activationTask = Task {
            assert(!Thread.isMainThread)
            let result: ActivationResult
            do {
                let session = AVAudioSession.sharedInstance()
                if !configured {
                    audioPhase("session.setCategory.begin")
                    try session.setCategory(.ambient, mode: .default)
                    audioPhase("session.setCategory.end")
                    configured = true
                }
                audioPhase("session.activate.begin")
                active = try await session.activate(options: [])
                assert(!Thread.isMainThread)
                audioPhase("session.activate.end")
                result = generation != lifecycleGeneration ? .superseded :
                    (active ? .ready : .unavailable("The audio session did not activate."))
            } catch {
                active = false
                result = generation != lifecycleGeneration ? .superseded : .unavailable(error.localizedDescription)
            }
            activationTask = nil
            // A canceled last note must release a late activation, even if Stop saw active == false.
            requestDeactivation(ignoringPendingNotes: generation != lifecycleGeneration)
            return result
        }
    }

    private func requestDeactivation(ignoringPendingNotes: Bool) {
        pending = pending.filter { !$0.value.isCancelled }
        guard !voices.values.contains(where: { $0.values.contains { $0.contains { $0.player.isPlaying } } }) else { return }
        guard ignoringPendingNotes || pending.isEmpty else { return }
        guard active, activationTask == nil, deactivationTask == nil else { return }
        deactivationTask = Task {
            assert(!Thread.isMainThread)
            do {
                audioPhase("session.deactivate.begin")
                if try await AVAudioSession.sharedInstance().deactivate(options: .notifyOthersOnDeactivation) { active = false }
                assert(!Thread.isMainThread)
                audioPhase("session.deactivate.end")
            } catch {
                // Keep last known session state; a future request will retry. No stopped player is resumed.
            }
            deactivationTask = nil
        }
    }

    private func scheduleIdleCleanup() {
        idleTask?.cancel()
        idleTask = Task {
            do { try await Task.sleep(for: .milliseconds(150)) } catch { return }
            guard !Task.isCancelled else { return }
            idleTask = nil
            deactivateIfIdle()
        }
    }
}

/// UIKit lifecycle handling stays immediate on MainActor; all hardware cleanup is ordered on the worker.
@MainActor private final class SensoryAudioSession {
    static let shared = SensoryAudioSession()
    private final class WeakEngine {
        weak var value: SensoryEngine?
        init(_ value: SensoryEngine) { self.value = value }
    }
    private var engines: [WeakEngine] = []
    private var observers: [NSObjectProtocol] = []
    private var commandTask: Task<Void, Never>?
    private var commandGeneration: UInt64 = 0

    private init() {
        AudioEligibility.shared.setAppActive(UIApplication.shared.applicationState == .active)
        let center = NotificationCenter.default
        observers.append(center.addObserver(forName: UIApplication.willResignActiveNotification, object: nil, queue: .main) { [weak self] _ in
            MainActor.assumeIsolated {
                AudioEligibility.shared.setAppActive(false)
                self?.stopAll(status: "Stopped · app inactive", discardPlayers: false)
            }
        })
        observers.append(center.addObserver(forName: UIApplication.didEnterBackgroundNotification, object: nil, queue: .main) { [weak self] _ in
            MainActor.assumeIsolated {
                AudioEligibility.shared.setAppActive(false)
                self?.stopAll(status: "Stopped · app backgrounded · tones stopped", discardPlayers: false)
            }
        })
        observers.append(center.addObserver(forName: UIApplication.didBecomeActiveNotification, object: nil, queue: .main) { _ in
            MainActor.assumeIsolated { AudioEligibility.shared.setAppActive(true) }
        })
        observers.append(center.addObserver(forName: AVAudioSession.didBecomeInactiveNotification, object: nil, queue: .main) { [weak self] notification in
            let causedBySystem = (notification.userInfo?[AVAudioSession.deactivationContextKey] as? AVAudioSession.DeactivationContext)?.source == .system
            MainActor.assumeIsolated {
                // App transition completions own session state. A delayed own notification cannot clear a new activation.
                guard causedBySystem else { return }
                AudioEligibility.shared.setInterrupted(true)
                self?.stopAll(status: "Audio session interrupted · tones stopped", discardPlayers: false, sessionInactive: true)
            }
        })
        observers.append(center.addObserver(forName: AVAudioSession.resumptionRecommendationNotification, object: nil, queue: .main) { _ in
            MainActor.assumeIsolated {
                AudioEligibility.shared.setInterrupted(false)
                // A recommendation permits a fresh touch; it never replays the previous one.
            }
        })
        observers.append(center.addObserver(forName: AVAudioSession.routeChangeNotification, object: nil, queue: .main) { [weak self] notification in
            let raw = notification.userInfo?[AVAudioSessionRouteChangeReasonKey] as? UInt
            MainActor.assumeIsolated {
                if raw == AVAudioSession.RouteChangeReason.oldDeviceUnavailable.rawValue {
                    self?.stopAll(status: "Output disconnected · tones stopped", discardPlayers: false)
                }
            }
        })
        observers.append(center.addObserver(forName: AVAudioSession.mediaServicesWereResetNotification, object: nil, queue: .main) { [weak self] _ in
            MainActor.assumeIsolated {
                AudioEligibility.shared.setInterrupted(false)
                self?.stopAll(status: "Audio service reset · next touch rebuilds tone players", discardPlayers: true, resetSession: true)
            }
        })
        observers.append(center.addObserver(forName: UIApplication.didReceiveMemoryWarningNotification, object: nil, queue: .main) { [weak self] _ in
            MainActor.assumeIsolated { self?.stopAll(status: "Memory pressure · tone cache released", discardPlayers: true) }
        })
    }

    func register(_ engine: SensoryEngine) {
        engines.removeAll { $0.value == nil }
        engines.append(WeakEngine(engine))
    }

    func play(engineID: UUID, token: AudioNoteToken, pitch: Int, strength: Double, gain: Double) async -> AudioPlaybackReport {
        // Fresh touches await prior Stop/reset commands, even if actor scheduling is not FIFO.
        let preceding = commandTask
        await preceding?.value
        return await SensoryAudioBackend.shared.play(engineID: engineID, token: token, pitch: pitch, strength: strength, gain: gain)
    }

    func snapshot(engineID: UUID) async -> AudioSnapshot {
        let preceding = commandTask
        await preceding?.value
        return await SensoryAudioBackend.shared.snapshot(engineID: engineID)
    }

    func stop(engineID: UUID) { enqueue { await $0.stop(engineID: engineID) } }
    func deactivateIfIdle() { enqueue { await $0.deactivateIfIdle() } }

    private func stopAll(status: String, discardPlayers: Bool, sessionInactive: Bool = false, resetSession: Bool = false) {
        engines.removeAll { $0.value == nil }
        for engine in engines.compactMap(\.value) { engine.stopLocal(status: status) }
        enqueue { await $0.stopAll(discardPlayers: discardPlayers, sessionInactive: sessionInactive, resetSession: resetSession) }
    }

    private func enqueue(_ operation: @escaping @Sendable (SensoryAudioBackend) async -> Void) {
        let preceding = commandTask
        commandGeneration &+= 1
        let generation = commandGeneration
        commandTask = Task { @MainActor [weak self] in
            await preceding?.value
            await operation(.shared)
            if let self, self.commandGeneration == generation { self.commandTask = nil }
        }
    }
}

/// Opt-in timing markers contain only public phase names, thread identity and time.
private func audioPhase(_ phase: String) {
    guard ProcessInfo.processInfo.arguments.contains("--diagnose-audio") else { return }
    print("ATTENTION_AUDIO_PHASE \(phase) epoch=\(String(format: "%.6f", Date.timeIntervalSinceReferenceDate + 978307200)) main=\(Thread.isMainThread)")
}

// MARK: - Pure PCM synthesis (also exercised by Tools/check-sensory.py)
enum SensoryTone {
    static let sampleRate = 22_050
    static let duration = 0.11
    static let peakAmplitude = 9_000.0

    static func clampedUnit(_ value: Double) -> Double {
        value.isFinite ? min(1, max(0, value)) : 0
    }

    static func hapticIntensity(weight: Double) -> Double {
        sqrt(clampedUnit(weight))
    }

    static func playbackVolume(weight: Double, gain: Double) -> Double {
        let weight = clampedUnit(weight)
        guard weight > 0 else { return 0 }
        return clampedUnit(gain) * (0.12 + 0.68 * sqrt(weight))
    }

    static func frequency(pitch: Int) -> Double {
        220 * pow(2, Double(max(0, min(23, pitch))) / 12)
    }

    static func wav(pitch: Int) -> Data {
        let frequency = frequency(pitch: pitch)
        let count = Int(Double(sampleRate) * duration)
        let attack = Int(Double(sampleRate) * 0.004)
        var samples = Data(capacity: count * 2)
        for index in 0..<count {
            let envelope: Double
            if index <= attack {
                envelope = 0.5 - 0.5 * cos(.pi * Double(index) / Double(attack))
            } else {
                envelope = 0.5 + 0.5 * cos(.pi * Double(index - attack) / Double(count - 1 - attack))
            }
            let phase = 2 * Double.pi * frequency * Double(index) / Double(sampleRate)
            var sample = Int16((sin(phase) * envelope * peakAmplitude).rounded()).littleEndian
            withUnsafeBytes(of: &sample) { samples.append(contentsOf: $0) }
        }
        var wav = Data(capacity: samples.count + 44)
        func ascii(_ value: String) { wav.append(contentsOf: value.utf8) }
        func u32(_ value: UInt32) { var value = value.littleEndian; withUnsafeBytes(of: &value) { wav.append(contentsOf: $0) } }
        func u16(_ value: UInt16) { var value = value.littleEndian; withUnsafeBytes(of: &value) { wav.append(contentsOf: $0) } }
        ascii("RIFF"); u32(UInt32(36 + samples.count)); ascii("WAVEfmt "); u32(16)
        u16(1); u16(1); u32(UInt32(sampleRate)); u32(UInt32(sampleRate * 2)); u16(2); u16(16)
        ascii("data"); u32(UInt32(samples.count)); wav.append(samples)
        return wav
    }
}
