import Foundation
import Observation

/// The teaching probe and shared runtime both use MLX's process-global allocator.
/// A mode transition drains the current engine before exposing controls for the other engine.
@Observable @MainActor
final class AttentionLocalIntelligence {
    enum Mode: String, CaseIterable, Identifiable {
        case assistant = "Text assistant"
        case attention = "Attention trace"
        var id: Self { self }
    }
    let inspection = LocalModelEngine()
    let assistant = AttentionTextAssistant()
    private(set) var mode: Mode
    private(set) var isSwitching = false
    @ObservationIgnored private var transitionTask: Task<Void, Never>?

    init() {
        let arguments = ProcessInfo.processInfo.arguments
        mode = arguments.contains("--verify-local-model") || arguments.contains("--verify-stability")
            || arguments.contains("--verify-long-stability") ? .attention : .assistant
    }

    func select(_ next: Mode) {
        guard next != mode, !isSwitching else { return }
        isSwitching = true
        transitionTask = Task { [self] in
            await unloadCurrentMode()
            if !Task.isCancelled { mode = next }
            isSwitching = false
            transitionTask = nil
        }
    }

    func cancel() {
        transitionTask?.cancel()
        if mode == .assistant { assistant.cancel() }
        else { inspection.cancel() }
    }

    func unload() async {
        transitionTask?.cancel()
        await transitionTask?.value
        await unloadCurrentMode()
    }

    private func unloadCurrentMode() async {
        if mode == .assistant { await assistant.unload() }
        else { await inspection.unload() }
    }
}
